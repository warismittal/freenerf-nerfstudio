#!/usr/bin/env python
"""Poses for the self-captured scene with COLMAP, written as a nerfstudio-style folder (transforms.json + images/) that
scripts/real_scene.py turns into the Blender layout used by every other scene.

    python scripts/colmap_scene.py --src /kaggle/input/<capture> --out data/real/capture [--size 800]

Phone photos come in both orientations, so every image is centre-cropped to a square and resized; the principal point
stays at the centre and, because all photos share one sensor, one SIMPLE_RADIAL camera fits all of them.
Uses pycolmap if it imports, else the `colmap` command line (apt-get install colmap). CPU only.
Prints the acceptance test of the proposal (Section 6.3): fraction of images registered and mean reprojection error.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

EXTS = (".jpg", ".jpeg", ".png", ".heic", ".JPG", ".JPEG", ".PNG")


def prepare_images(src: Path, img_dir: Path, size: int) -> list[str]:
    img_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.suffix in EXTS and not p.name.startswith("."))
    names = []
    for k, p in enumerate(files):
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.LANCZOS)
        name = f"frame_{k:05d}.jpg"
        im.save(img_dir / name, quality=95)
        names.append(name)
    (img_dir.parent / "source_files.json").write_text(json.dumps({n: str(p.name) for n, p in zip(names, files)}, indent=1))
    return names


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def run_pycolmap(img_dir: Path, work: Path):
    import pycolmap

    db = work / "database.db"
    sparse = work / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)
    pycolmap.extract_features(db, img_dir, camera_mode=pycolmap.CameraMode.SINGLE, camera_model="SIMPLE_RADIAL")
    pycolmap.match_exhaustive(db)
    recs = pycolmap.incremental_mapping(db, img_dir, sparse)
    if not recs:
        raise RuntimeError("COLMAP registered no images")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    cam = next(iter(rec.cameras.values()))
    f, cx, cy, k1 = (list(cam.params) + [0.0])[:4]
    poses = {}
    for img in rec.images.values():
        cfw = img.cam_from_world
        cfw = cfw() if callable(cfw) else cfw
        m = np.eye(4)
        m[:3, :4] = np.asarray(cfw.matrix())
        poses[img.name] = m
    try:
        err = float(rec.compute_mean_reprojection_error())
    except Exception:
        err = float("nan")
    return poses, (float(f), float(cx), float(cy), float(k1)), int(cam.width), int(cam.height), err


def run_cli(img_dir: Path, work: Path):
    db, sparse, txt = work / "database.db", work / "sparse", work / "txt"
    for d in (sparse, txt):
        d.mkdir(parents=True, exist_ok=True)
    sh = lambda *a: subprocess.run([str(x) for x in a], check=True)  # noqa: E731
    sh("colmap", "feature_extractor", "--database_path", db, "--image_path", img_dir, "--ImageReader.single_camera", 1,
       "--ImageReader.camera_model", "SIMPLE_RADIAL", "--SiftExtraction.use_gpu", 0)
    sh("colmap", "exhaustive_matcher", "--database_path", db, "--SiftMatching.use_gpu", 0)
    sh("colmap", "mapper", "--database_path", db, "--image_path", img_dir, "--output_path", sparse)
    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    best = max(models, key=lambda p: (p / "images.bin").stat().st_size)
    sh("colmap", "model_converter", "--input_path", best, "--output_path", txt, "--output_type", "TXT")
    cam_line = [l for l in (txt / "cameras.txt").read_text().splitlines() if l and not l.startswith("#")][0].split()
    w, h = int(cam_line[2]), int(cam_line[3])
    f, cx, cy, k1 = (list(map(float, cam_line[4:])) + [0.0])[:4]
    lines = [l for l in (txt / "images.txt").read_text().splitlines() if l and not l.startswith("#")]
    poses = {}
    for l in lines[::2]:
        e = l.split()
        R, t = qvec2rotmat(list(map(float, e[1:5]))), np.array(list(map(float, e[5:8])))
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = R, t
        poses[e[9]] = m
    err = float("nan")
    pts = [l.split() for l in (txt / "points3D.txt").read_text().splitlines() if l and not l.startswith("#")]
    if pts:
        err = float(np.mean([float(p[7]) for p in pts]))
    return poses, (f, cx, cy, k1), w, h, err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder (searched recursively) with the photos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=800)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if (out / "transforms.json").exists():
        print(f"{out}/transforms.json exists - reusing")
        return 0
    img_dir = out / "images"
    names = prepare_images(Path(a.src), img_dir, a.size)
    work = out / "colmap"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        poses, (f, cx, cy, k1), w, h, err = run_pycolmap(img_dir, work)
        tool = "pycolmap"
    except ImportError:
        poses, (f, cx, cy, k1), w, h, err = run_cli(img_dir, work)
        tool = "colmap CLI"
    frames = []
    for name in names:
        if name not in poses:
            continue
        c2w = np.linalg.inv(poses[name])
        c2w[:3, 1:3] *= -1                     # OpenCV (x right, y down, z forward) -> OpenGL/nerfstudio (y up, z back)
        frames.append({"file_path": f"images/{name}", "transform_matrix": c2w.tolist()})
    meta = {"w": w, "h": h, "fl_x": f, "fl_y": f, "cx": cx, "cy": cy, "k1": k1, "k2": 0.0, "p1": 0.0, "p2": 0.0,
            "camera_model": "OPENCV", "frames": frames}
    (out / "transforms.json").write_text(json.dumps(meta, indent=1))
    frac = len(frames) / max(len(names), 1)
    report = {"tool": tool, "images": len(names), "registered": len(frames), "fraction": round(frac, 3),
              "mean_reprojection_error_px": round(err, 3), "focal_px": round(f, 1), "k1": round(k1, 4),
              "acceptance": bool(frac >= 0.95 and err < 1.0)}
    (out / "colmap_report.json").write_text(json.dumps(report, indent=2))
    print("[colmap_scene]", json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
