#!/usr/bin/env python
"""Side-by-side RGB + depth videos: vanilla | FreeNeRF | depth smoothness (| ground truth on the test path). Needs ffmpeg.
   python scripts/make_videos.py --scene lego --seed 0
Reads outputs/<run_id>/export (written by run_arm.py with --orbit) and writes results/videos/<scene>_{orbit,test}.mp4."""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evalkit.io import load_gt, load_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arms", nargs="+", default=["A", "D", "E"])
ap.add_argument("--outputs", default="outputs")
ap.add_argument("--data-root", default="data/blender400")
ap.add_argument("--manifests", default="manifests")
ap.add_argument("--out", default="results/videos")
ap.add_argument("--fps", type=int, default=12)
a = ap.parse_args()
NAMES = {"A": "vanilla NeRF", "D": "FreeNeRF", "E": "depth smoothness", "B": "frequency only", "C": "occlusion only"}


def turbo(d01):
    import matplotlib
    return (matplotlib.colormaps["turbo"](np.clip(d01, 0, 1))[..., :3] * 255).astype(np.uint8)


def label(img, text):
    im = Image.fromarray(img)
    ImageDraw.Draw(im).text((6, 4), text, fill=(0, 0, 0))
    return np.asarray(im)


def encode(frames, name):
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        for i, f in enumerate(frames):
            Image.fromarray(f).save(Path(t) / f"f_{i:04d}.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(Path(t) / "f_%04d.png"),
                        "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(Path(a.out) / name)], check=True)
    print("wrote", Path(a.out) / name)


rid = lambda arm: f"{a.scene}_{arm}_n8_s{a.seed}"
ex = lambda arm: Path(a.outputs) / rid(arm) / "export"

# orbit: novel camera path, no ground truth exists
if all((ex(m) / "orbit_rgb").exists() for m in a.arms):
    n = len(list((ex(a.arms[0]) / "orbit_rgb").glob("*.png")))
    frames = []
    for i in range(n):
        top = [label(np.asarray(Image.open(ex(m) / "orbit_rgb" / f"{i:03d}.png").convert("RGB")), NAMES.get(m, m)) for m in a.arms]
        bot = [turbo(np.asarray(Image.open(ex(m) / "orbit_depth" / f"{i:03d}.png"), dtype=np.float32) / 255.0) for m in a.arms]
        frames.append(np.vstack([np.hstack(top), np.hstack(bot)]))
    encode(frames, f"{a.scene}_orbit.mp4")

# held-out test path: includes ground truth
if all((ex(m) / "test_rgb.npy").exists() for m in a.arms):
    man = load_json(Path(a.manifests) / f"{a.scene}_n8.json")
    gt, _ = load_gt(a.data_root, a.scene, "test", man["test"])
    R = {m: np.load(ex(m) / "test_rgb.npy") for m in a.arms}
    D = {m: np.load(ex(m) / "test_depth.npy") for m in a.arms}
    u8 = lambda x: (np.clip(x, 0, 1) * 255).astype(np.uint8)
    frames = []
    for i in range(len(gt)):
        top = [label(u8(R[m][i]), NAMES.get(m, m)) for m in a.arms] + [label(u8(gt[i]), "ground truth")]
        bot = [turbo((D[m][i] - 2.0) / 4.0) for m in a.arms] + [np.full_like(top[-1], 255)]
        frames.append(np.vstack([np.hstack(top), np.hstack(bot)]))
    encode(frames, f"{a.scene}_testpath.mp4")
