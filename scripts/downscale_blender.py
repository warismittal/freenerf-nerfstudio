#!/usr/bin/env python
"""Downscale NeRF-synthetic scenes to 400x400 (area filter). Usage: python scripts/downscale_blender.py --src data/blender --dst data/blender400"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freenerf_ns.datasets import SYNTHETIC_SCENES, downscale_scene  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="data/blender")
ap.add_argument("--dst", default="data/blender400")
ap.add_argument("--scenes", nargs="+", default=SYNTHETIC_SCENES)
ap.add_argument("--size", type=int, default=400)
a = ap.parse_args()
for s in a.scenes:
    print(f"downscaling {s} -> {a.size}x{a.size}")
    downscale_scene(a.src, a.dst, s, a.size)
