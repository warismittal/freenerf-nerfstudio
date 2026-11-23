#!/usr/bin/env python
"""Materialise the trainer's isolated dataset folders.  python scripts/make_sparse_datasets.py --scene lego --n 8 [--permute-poses]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freenerf_ns import datasets  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--n", required=True, help="number of views, or 'dense'")
ap.add_argument("--data-root", default="data/blender400")
ap.add_argument("--manifests", default="manifests")
ap.add_argument("--out", default="data/sparse")
ap.add_argument("--permute-poses", action="store_true", help="negative control C1")
a = ap.parse_args()
d = datasets.make_sparse(a.data_root, a.manifests, a.scene, a.n, a.out, 0 if a.permute_poses else None)
print("trainer dataset:", d)
