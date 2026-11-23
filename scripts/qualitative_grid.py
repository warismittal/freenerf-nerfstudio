#!/usr/bin/env python
"""R5: RGB, depth and opacity of one held-out view for every arm next to the ground truth.
   python scripts/qualitative_grid.py --scene lego --view 3"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evalkit.io import load_gt, load_json  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--view", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arms", nargs="+", default=["A", "B", "C", "D", "E"])
ap.add_argument("--outputs", default="outputs")
ap.add_argument("--data-root", default="data/blender400")
ap.add_argument("--manifests", default="manifests")
ap.add_argument("--out", default="results/figures/R5_qualitative.png")
a = ap.parse_args()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

man = load_json(Path(a.manifests) / f"{a.scene}_n8.json")
gt, _ = load_gt(a.data_root, a.scene, "test", [man["test"][a.view]])
arms = [m for m in a.arms if (Path(a.outputs) / f"{a.scene}_{m}_n8_s{a.seed}" / "export" / "test_rgb.npy").exists()]
fig, axes = plt.subplots(3, len(arms) + 1, figsize=(2.4 * (len(arms) + 1), 7.4), squeeze=False)
axes[0, 0].imshow(gt[0]), axes[0, 0].set_title("ground truth", fontsize=8)
for j, m in enumerate(arms, 1):
    e = Path(a.outputs) / f"{a.scene}_{m}_n8_s{a.seed}" / "export"
    axes[0, j].imshow(np.load(e / "test_rgb.npy")[a.view]), axes[0, j].set_title(f"arm {m}", fontsize=8)
    axes[1, j].imshow(np.load(e / "test_depth.npy")[a.view], cmap="turbo", vmin=2, vmax=6)
    axes[2, j].imshow(np.load(e / "test_acc.npy")[a.view], cmap="gray", vmin=0, vmax=1)
axes[1, 0].axis("off"), axes[2, 0].axis("off")
for ax in axes.ravel():
    ax.set_xticks([]), ax.set_yticks([])
axes[0, 0].set_ylabel("RGB"), axes[1, 1].set_ylabel("expected depth"), axes[2, 1].set_ylabel("opacity")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout(), fig.savefig(a.out, dpi=170)
print("wrote", a.out)
