"""Shared test fixtures: a tiny fake NeRF-synthetic scene and a fake nerfstudio real-scene folder (no network, no GPU)."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from freenerf_ns.geometry import look_at_c2w


def _poses(n, seed, radius=4.0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        az, el = rng.uniform(0, 2 * np.pi), rng.uniform(0.15, 1.2)
        p = radius * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        out.append(look_at_c2w(p))
    return out


def make_fake_blender(root, scenes=("lego",), size=32, n_train=100, n_test=200, seed=0):
    root = Path(root)
    rng = np.random.default_rng(seed)
    for scene in scenes:
        for split, n in (("train", n_train), ("test", n_test)):
            d = root / scene / split
            d.mkdir(parents=True, exist_ok=True)
            frames = []
            for i, pose in enumerate(_poses(n, seed + hash(split) % 7)):
                yy, xx = np.mgrid[:size, :size]
                alpha = (((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (size / 3) ** 2).astype(np.uint8) * 255
                rgb = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
                Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(d / f"r_{i}.png")
                frames.append({"file_path": f"./{split}/r_{i}", "rotation": 0.0, "transform_matrix": pose.tolist()})
            (root / scene / f"transforms_{split}.json").write_text(
                json.dumps({"camera_angle_x": 0.69, "frames": frames}))


def make_fake_nerfstudio_scene(folder, n=30, w=64, h=48, seed=0):
    folder = Path(folder)
    (folder / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = []
    for i, pose in enumerate(_poses(n, seed, radius=1.3)):
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(folder / "images" / f"frame_{i:05d}.png")
        frames.append({"file_path": f"images/frame_{i:05d}.png", "transform_matrix": pose.tolist()})
    (folder / "transforms.json").write_text(json.dumps(
        {"camera_model": "OPENCV", "fl_x": 55.0, "fl_y": 55.0, "cx": w / 2, "cy": h / 2, "w": w, "h": h,
         "k1": 0.0, "k2": 0.0, "frames": frames}))
