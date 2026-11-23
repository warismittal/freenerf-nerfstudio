#!/usr/bin/env python
"""Prepare the self-captured scene from an `ns-process-data images` output folder.

   ns-process-data images --data captures/scene1 --output-dir data/real/scene1
   python scripts/real_scene.py --ns-dir data/real/scene1 --out-root data/blender400
Acceptance test first (proposal Section 6.3): >=95% of images registered and mean reprojection error <1 px (check the COLMAP log).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freenerf_ns import datasets  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ns-dir", required=True)
ap.add_argument("--out-root", default="data/blender400")
ap.add_argument("--manifests", default="manifests")
ap.add_argument("--n-test", type=int, default=14)
ap.add_argument("--n-train", type=int, default=8)
ap.add_argument("--width", type=int, default=400)
a = ap.parse_args()
info = datasets.convert_real_scene(a.ns_dir, a.out_root, a.manifests, "real", a.width, a.n_test, a.n_train)
print(f"{info['n_frames']} registered views -> {info['dense']} dense, {info['n_test']} test, sparse train ids {info['train']}")
for w in info["warnings"]:
    print("WARNING:", w)
print("Next: open the first training image and an orbit render in the viewer; confirm the object sits inside the default near/far (2..6).")
