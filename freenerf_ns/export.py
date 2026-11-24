"""Render a trained run to disk as float32 arrays (the only thing evalkit reads). Requires nerfstudio + torch.

    python -m freenerf_ns.export --config outputs/<run>/<method>/main/config.yml --scene lego \
        --manifest manifests/lego_n8.json --data-root data/blender400 --out outputs/<run>/export [--orbit 60]

Writes  test_{rgb,depth,acc}.npy [V,H,W,(3)], train_{rgb,depth,acc}.npy (unless --no-train),
        train_rays_w.npy [R,S] (fine-pass weights of fixed training rays), meta.json, optional orbit PNG frames.
Test *images* are never read here - only test camera poses.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .geometry import orbit_poses


def trust_own_checkpoints() -> None:
    """torch>=2.6 defaults torch.load(weights_only=True), which rejects the numpy scalars nerfstudio stores in its
    checkpoints. The checkpoints loaded here are the ones this pipeline just wrote, so full unpickling is safe."""
    import torch

    if getattr(torch.load, "_freenerf_patched", False):
        return
    orig = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig(*args, **kwargs)

    load._freenerf_patched = True
    torch.load = load


def cameras_from_transforms(tj: Path, indices, width: int, height: int):
    import torch
    from nerfstudio.cameras.cameras import Cameras, CameraType

    meta = json.loads(Path(tj).read_text())
    focal = 0.5 * width / math.tan(0.5 * float(meta["camera_angle_x"]))
    poses = torch.tensor(np.array([meta["frames"][i]["transform_matrix"] for i in indices]), dtype=torch.float32)[:, :3, :4]
    return Cameras(camera_to_worlds=poses, fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0, width=width,
                   height=height, camera_type=CameraType.PERSPECTIVE)


def cameras_from_c2w(c2w: np.ndarray, focal: float, width: int, height: int):
    import torch
    from nerfstudio.cameras.cameras import Cameras, CameraType

    poses = torch.tensor(c2w[:, :3, :4], dtype=torch.float32)
    return Cameras(camera_to_worlds=poses, fx=focal, fy=focal, cx=width / 2.0, cy=height / 2.0, width=width,
                   height=height, camera_type=CameraType.PERSPECTIVE)


def render_cameras(model, cameras, device):
    import torch

    rgb, depth, acc = [], [], []
    with torch.no_grad():
        for i in range(int(cameras.camera_to_worlds.shape[0])):
            bundle = cameras.generate_rays(camera_indices=i, keep_shape=True).to(device)
            out = model.get_outputs_for_camera_ray_bundle(bundle)
            rgb.append(out["rgb_fine"].clamp(0, 1).cpu().numpy())
            depth.append(out["depth_fine"][..., 0].cpu().numpy())
            acc.append(out["accumulation_fine"][..., 0].cpu().numpy())
    return np.stack(rgb).astype(np.float32), np.stack(depth).astype(np.float32), np.stack(acc).astype(np.float32)


def train_ray_weights(model, cameras, device, n_rays: int, seed: int):
    """Fine-pass weights [R,S] of a fixed random set of training rays (for Omega_M)."""
    import torch
    from nerfstudio.cameras.rays import RayBundle

    rng = np.random.default_rng(seed)
    n_cams = int(cameras.camera_to_worlds.shape[0])
    per = max(1, n_rays // n_cams)
    O, D = [], []
    for i in range(n_cams):
        b = cameras.generate_rays(camera_indices=i, keep_shape=True)
        o, d = b.origins.reshape(-1, 3), b.directions.reshape(-1, 3)
        idx = torch.from_numpy(rng.choice(o.shape[0], per, replace=False))
        O.append(o[idx]), D.append(d[idx])
    O, D = torch.cat(O), torch.cat(D)
    ws = []
    for s in range(0, O.shape[0], 4096):
        bundle = RayBundle(origins=O[s:s + 4096], directions=D[s:s + 4096], pixel_area=torch.ones(len(O[s:s + 4096]), 1)).to(device)
        w, _ = model.ray_stats(bundle)
        ws.append(w.cpu().numpy())
    return np.concatenate(ws).astype(np.float32)


def save_orbit(model, train_c2w, focal, width, height, device, n_frames, out: Path):
    from PIL import Image

    cams = cameras_from_c2w(orbit_poses(train_c2w, n_frames), focal, width, height)
    rgb, depth, _ = render_cameras(model, cams, device)
    (out / "orbit_rgb").mkdir(parents=True, exist_ok=True)
    (out / "orbit_depth").mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        Image.fromarray((rgb[i] * 255).round().astype(np.uint8)).save(out / "orbit_rgb" / f"{i:03d}.png")
        d = np.clip((depth[i] - 2.0) / 4.0, 0, 1)             # fixed near/far range -> comparable across arms
        Image.fromarray((d * 255).round().astype(np.uint8)).save(out / "orbit_depth" / f"{i:03d}.png")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="data/blender400")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=400)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--orbit", type=int, default=0)
    ap.add_argument("--no-train", action="store_true", help="skip train-view renders (dense reference runs)")
    ap.add_argument("--n-ray-stats", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="render only the first N test views (smoke tests)")
    ap.add_argument("--chunk", type=int, default=4096,
                    help="rays rendered per forward pass; 1<<15 needs ~8 GB for vanilla-nerf and OOMs a 15 GB T4")
    a = ap.parse_args(argv)

    from nerfstudio.utils.eval_utils import eval_setup

    trust_own_checkpoints()
    t0 = time.time()
    config, pipeline, ckpt, step = eval_setup(Path(a.config), test_mode="inference")
    model, device = pipeline.model.eval(), pipeline.device
    model.config.eval_num_rays_per_chunk = a.chunk
    for r in (getattr(model, "renderer_rgb", None),):      # random-background training renders on white for eval
        if r is not None and getattr(r, "background_color", None) == "random":
            r.background_color = "white"
    import torch
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    man = json.loads(Path(a.manifest).read_text())
    if a.limit:
        man = {**man, "test": man["test"][:a.limit]}
    root = Path(a.data_root) / a.scene
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cams_test = cameras_from_transforms(root / "transforms_test.json", man["test"], a.width, a.height)
    rgb, depth, acc = render_cameras(model, cams_test, device)
    np.save(out / "test_rgb.npy", rgb), np.save(out / "test_depth.npy", depth), np.save(out / "test_acc.npy", acc)

    cams_train = cameras_from_transforms(root / "transforms_train.json", man["train"], a.width, a.height)
    if not a.no_train:
        rgb, depth, acc = render_cameras(model, cams_train, device)
        np.save(out / "train_rgb.npy", rgb), np.save(out / "train_depth.npy", depth), np.save(out / "train_acc.npy", acc)
        np.save(out / "train_rays_w.npy", train_ray_weights(model, cams_train, device, a.n_ray_stats, a.seed))
    if a.orbit:
        focal = 0.5 * a.width / math.tan(0.5 * float(json.loads((root / "transforms_train.json").read_text())["camera_angle_x"]))
        c2w = np.concatenate([cams_train.camera_to_worlds.numpy(), np.tile([[[0, 0, 0, 1.0]]], (len(man["train"]), 1, 1))], 1)
        save_orbit(model, c2w, focal, a.width, a.height, device, a.orbit, out)

    meta = dict(scene=a.scene, manifest=a.manifest, n_train=len(man["train"]), n_test=len(man["test"]),
                step=int(step), config=str(a.config), checkpoint=str(ckpt), export_seconds=time.time() - t0,
                has_train=not a.no_train)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"exported {len(man['test'])} test views to {out}")


if __name__ == "__main__":
    main()
