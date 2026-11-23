"""Loading ground truth, manifests and camera positions. NumPy + Pillow only."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def frame_image_path(transforms_json, idx: int) -> Path:
    """Path of frame idx's image next to its transforms file (Blender convention: './train/r_0' -> 'train/r_0.png')."""
    tj = Path(transforms_json)
    meta = load_json(tj)
    rel = meta["frames"][idx]["file_path"]
    rel = rel[2:] if rel.startswith("./") else rel
    p = tj.parent / rel
    return p if p.suffix else p.with_suffix(".png")


def load_rgba_on_white(path):
    """-> (rgb float32 [H,W,3] composited on white, alpha float32 [H,W])."""
    im = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float32) / 255.0
    a = im[..., 3]
    rgb = im[..., :3] * a[..., None] + (1.0 - a[..., None])
    return rgb.astype(np.float32), a.astype(np.float32)


def load_gt(data_root, scene: str, split: str, indices):
    """Ground-truth images of `split` ('train'|'test') for the given frame indices: rgb [V,H,W,3], alpha [V,H,W]."""
    tj = Path(data_root) / scene / f"transforms_{split}.json"
    rgb, alpha = [], []
    for i in indices:
        r, a = load_rgba_on_white(frame_image_path(tj, i))
        rgb.append(r), alpha.append(a)
    return np.stack(rgb), np.stack(alpha)


def camera_positions(transforms_json, indices) -> np.ndarray:
    meta = load_json(transforms_json)
    return np.array([np.asarray(meta["frames"][i]["transform_matrix"])[:3, 3] for i in indices], dtype=np.float64)


def load_manifest(manifest_dir, scene: str, n) -> dict:
    return load_json(Path(manifest_dir) / f"{scene}_n{n}.json")


def resolve_n(manifest_dir, scene: str, n):
    """'dense' -> the number of views of the dense reference for that scene (from manifests/index.json)."""
    if str(n) != "dense":
        return int(n)
    return int(load_json(Path(manifest_dir) / "index.json")[scene]["dense"])
