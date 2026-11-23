"""Dataset preparation (pure Python: NumPy + Pillow). Used by the scripts and unit-tested on a synthetic mini-dataset.

Layout produced
    data/blender/<scene>/{train,test}/r_i.png + transforms_{train,test}.json    original NeRF-synthetic (800x800)
    data/blender400/<scene>/...                                                 same, downscaled to 400x400 (DietNeRF protocol)
    data/sparse/<scene>_n<N>/train/r_i.png + transforms_{train,val,test}.json   what the *trainer* sees: only the N training views
    manifests/<scene>_n<N>.json, index.json, HASHES.txt                         the single source of truth for every split

Isolation: the trainer's folder contains no test image at all (and val/test json point at training images), so a training run
physically cannot read test data. This replaces a custom data parser with something simpler and stronger.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .geometry import fps, fps_extend

# 8-view DietNeRF Blender protocol (used by FreeNeRF): 8 training views, 25 test images (testskip=8), half resolution.
# Verified against github.com/ajayjain/DietNeRF, dietnerf/run_nerf.py: np.random.seed(args.seed=0) (line 722), then
# np.random.choice(i_train, size=8, replace=False) (line 799), with no random draw in between, i.e.
# np.random.seed(0); np.random.choice(np.arange(100), 8, replace=False) == [26, 86, 2, 55, 75, 93, 16, 73]  (every scene)
DIETNERF_8 = [26, 86, 2, 55, 75, 93, 16, 73]
SYNTHETIC_SCENES = ["lego", "chair", "drums", "ficus"]
SWEEP = [4, 8, 16, 32, 100]


def _load(p):
    return json.loads(Path(p).read_text())


def _dump(obj, p):
    Path(p).write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _src(root: Path, tj: Path, frame) -> Path:
    rel = frame["file_path"]
    rel = rel[2:] if rel.startswith("./") else rel
    p = tj.parent / rel
    return p if p.suffix else p.with_suffix(".png")


# ------------------------------------------------------------------------------------------------ downscale
def downscale_scene(src_root, dst_root, scene: str, size: int = 400, splits=("train", "test")) -> None:
    """Area-downscale a Blender scene to size x size (RGBA resized as-is, composited on white later, as in DietNeRF)."""
    src, dst = Path(src_root) / scene, Path(dst_root) / scene
    for split in splits:
        tj = src / f"transforms_{split}.json"
        meta = _load(tj)
        (dst / split).mkdir(parents=True, exist_ok=True)
        for fr in meta["frames"]:
            sp = _src(src, tj, fr)
            rel = fr["file_path"][2:] if fr["file_path"].startswith("./") else fr["file_path"]
            dp = (dst / rel).with_suffix(".png")
            im = Image.open(sp).convert("RGBA")
            if im.size != (size, size):
                im = im.resize((size, size), Image.BOX)
            im.save(dp)
        _dump(meta, dst / f"transforms_{split}.json")


# ------------------------------------------------------------------------------------------------ manifests
def write_hashes(manifest_dir) -> None:
    d = Path(manifest_dir)
    files = sorted(list(d.glob("*_n*.json")) + [d / "index.json"])
    lines = [f"{p.name} {hashlib.sha256(p.read_bytes()).hexdigest()}" for p in files if p.exists()]
    (d / "HASHES.txt").write_text("\n".join(lines) + "\n")


def update_index(manifest_dir, scene: str, entry: dict) -> None:
    p = Path(manifest_dir) / "index.json"
    idx = _load(p) if p.exists() else {}
    idx[scene] = entry
    _dump(idx, p)


def synthetic_manifests(data_root, manifest_dir, scenes=SYNTHETIC_SCENES, eight=None, n_test_stride: int = 8) -> list:
    """Nested training subsets (4 < 8 < 16 < 32 < 100) and the 25-image test split for each synthetic scene."""
    eight = list(eight or DIETNERF_8)
    out = []
    Path(manifest_dir).mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        tj = Path(data_root) / scene / "transforms_train.json"
        meta = _load(tj)
        pos = np.array([np.asarray(f["transform_matrix"])[:3, 3] for f in meta["frames"]])
        n_all = len(pos)
        n_test = len(_load(Path(data_root) / scene / "transforms_test.json")["frames"])
        test = list(range(0, n_test, n_test_stride))
        s8 = eight
        s16 = fps_extend(pos, s8, 16)
        s32 = fps_extend(pos, s16, 32)
        s4 = [s8[i] for i in fps_extend(pos[s8], [0], 4)]
        subsets = {4: s4, 8: s8, 16: s16, 32: s32, n_all: list(range(n_all))}
        for n, train in subsets.items():
            note = ("DietNeRF 8-view indices (np.random.seed(0); choice(100, 8)), verified against ajayjain/DietNeRF" if n == 8 else
                    "all training views" if n == n_all else "farthest-point sampling, nested in the 8-view set")
            _dump({"scene": scene, "n": n, "train": [int(i) for i in train], "test": test, "source": note},
                  Path(manifest_dir) / f"{scene}_n{n}.json")
            out.append((scene, n))
        update_index(manifest_dir, scene, {"views": sorted(subsets), "dense": n_all, "test": len(test)})
    write_hashes(manifest_dir)
    return out


# ------------------------------------------------------------------------------------------------ sparse datasets
def make_sparse(data_root, manifest_dir, scene: str, n, out_root, permute_seed=None) -> Path:
    """Materialise the trainer's folder for (scene, n): only the manifest's training images, poses in transforms_*.json.
    permute_seed shuffles the image-pose pairing (negative control C1)."""
    from evalkit import controls

    problems = controls.verify_hashes(manifest_dir)
    if problems:
        raise RuntimeError(f"manifest integrity check failed: {problems}")
    if str(n) == "dense":
        n = _load(Path(manifest_dir) / "index.json")[scene]["dense"]
    man = _load(Path(manifest_dir) / f"{scene}_n{n}.json")
    root = Path(data_root) / scene
    tj = root / "transforms_train.json"
    meta = _load(tj)
    dest = Path(out_root) / (f"{scene}_n{n}" + ("_perm" if permute_seed is not None else ""))
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "train").mkdir(parents=True)
    frames = []
    for i in man["train"]:
        fr = dict(meta["frames"][i])
        sp = _src(root, tj, fr)
        shutil.copyfile(sp, dest / "train" / f"r_{i}.png")
        fr["file_path"] = f"./train/r_{i}"
        frames.append(fr)
    if permute_seed is not None and len(frames) > 1:
        rng = np.random.default_rng(permute_seed)
        poses = [f["transform_matrix"] for f in frames]
        perm = rng.permutation(len(frames))
        while (perm == np.arange(len(frames))).any() and len(frames) > 2:     # no fixed points
            perm = rng.permutation(len(frames))
        for f, j in zip(frames, perm):
            f["transform_matrix"] = poses[int(j)]
    out_meta = {k: v for k, v in meta.items() if k != "frames"}
    out_meta["frames"] = frames
    for split in ("train", "val", "test"):                                       # val/test point at training images only
        _dump(out_meta, dest / f"transforms_{split}.json")
    bad = controls.check_sparse_dir(dest, man)
    if bad:
        raise RuntimeError(f"sparse dataset isolation check failed: {bad}")
    return dest


# ------------------------------------------------------------------------------------------------ real scene
def convert_real_scene(ns_dir, out_root, manifest_dir, scene: str = "real", width: int = 400, n_test: int = 14,
                       n_train: int = 8, seed: int = 0, radius: float = 4.0) -> dict:
    """Convert a nerfstudio `ns-process-data images` folder (transforms.json + images/) into the Blender-format layout so the
    real scene runs through exactly the same pipeline as Lego and Chair.

    - re-centres at the least-squares intersection of the optical axes and rescales so cameras sit at distance `radius`
      (default 4, like Blender: the default near/far of 2..6 then apply unchanged);
    - holds out `n_test` views evenly spread in angle as test; the rest is the dense reference pool;
    - picks the `n_train` sparse views by farthest-point sampling on the pool.
    """
    ns_dir = Path(ns_dir)
    meta = _load(ns_dir / "transforms.json")
    frames = meta["frames"]
    W, H = int(meta["w"]), int(meta["h"])
    fl = float(meta.get("fl_x") or meta["fl_y"])
    warn = []
    if abs(meta.get("k1", 0)) > 0.05 or abs(meta.get("k2", 0)) > 0.05:
        warn.append("noticeable lens distortion (k1/k2); the pinhole NeRF ignores it - consider undistorting the photos")
    c2w = np.array([f["transform_matrix"] for f in frames], dtype=np.float64)
    P, Fw = c2w[:, :3, 3].copy(), -c2w[:, :3, 2]
    A = np.zeros((3, 3)); b = np.zeros(3)
    for p, f in zip(P, Fw):
        m = np.eye(3) - np.outer(f, f)
        A += m; b += m @ p
    center = np.linalg.solve(A, b)
    scale = radius / np.mean(np.linalg.norm(P - center, axis=1))
    c2w[:, :3, 3] = (P - center) * scale
    rel = c2w[:, :3, 3]
    _, _, vt = np.linalg.svd(rel - rel.mean(0))
    xy = (rel - rel.mean(0)) @ vt[:2].T
    order = np.argsort(np.arctan2(xy[:, 1], xy[:, 0]))
    pick = np.unique(np.round(np.linspace(0, len(order) - 1, n_test)).astype(int))
    test_ids = [int(order[i]) for i in pick]
    pool_ids = [i for i in range(len(frames)) if i not in set(test_ids)]
    train_local = fps(rel[pool_ids], n_train, seed)

    out = Path(out_root) / scene
    for split in ("train", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)
    h = int(round(H * width / W))
    angle_x = 2 * math.atan(W / (2 * fl))

    def write_split(split, ids):
        fr_out = []
        for k, i in enumerate(ids):
            im = Image.open(ns_dir / frames[i]["file_path"]).convert("RGB").resize((width, h), Image.LANCZOS)
            im.save(out / split / f"r_{k}.png")
            fr_out.append({"file_path": f"./{split}/r_{k}", "transform_matrix": c2w[i].tolist()})
        _dump({"camera_angle_x": angle_x, "frames": fr_out}, out / f"transforms_{split}.json")

    write_split("train", pool_ids)
    write_split("test", test_ids)
    md = Path(manifest_dir)
    md.mkdir(parents=True, exist_ok=True)
    dense = len(pool_ids)
    note = "farthest-point sampling on the camera centres of the non-test views"
    _dump({"scene": scene, "n": n_train, "train": [int(i) for i in train_local], "test": list(range(len(test_ids))), "source": note},
          md / f"{scene}_n{n_train}.json")
    _dump({"scene": scene, "n": dense, "train": list(range(dense)), "test": list(range(len(test_ids))),
           "source": "all non-test views"}, md / f"{scene}_n{dense}.json")
    update_index(md, scene, {"views": [n_train, dense], "dense": dense, "test": len(test_ids)})
    write_hashes(md)
    return {"n_frames": len(frames), "n_test": len(test_ids), "dense": dense, "train": train_local, "warnings": warn}
