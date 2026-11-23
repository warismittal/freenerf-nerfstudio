"""Score one exported run: per-view metrics, failure signatures and controls -> results/runs/<run_id>.json

    python -m evalkit.run_eval --run-dir outputs/<run_id> --run-id lego_D_n8_s0 --scene lego --arm D --n 8 --seed 0 \
        --manifest manifests/lego_n8.json --data-root data/blender400 --out results/runs
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import controls, diagnostics, metrics
from .io import camera_positions, load_gt, load_json
from freenerf_ns.geometry import angular_distance_deg


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def evaluate_export(export_dir, data_root, scene: str, manifest: dict, *, M: int = 20, lpips_nets=("alex", "vgg"),
                    device: str = "cpu") -> dict:
    ex = Path(export_dir)
    test_idx, train_idx = manifest["test"], manifest["train"]
    pred = np.load(ex / "test_rgb.npy")
    acc = np.load(ex / "test_acc.npy")
    gt, alpha = load_gt(data_root, scene, "test", test_idx)
    assert pred.shape == gt.shape, f"render {pred.shape} vs ground truth {gt.shape}"

    per_view = [metrics.view_metrics(pred[i], gt[i], alpha[i], lpips_nets, device) for i in range(len(test_idx))]
    test = {k: [v[k] for v in per_view] for k in per_view[0]}
    const_bg = [metrics.psnr(np.ones_like(g), g) for g in gt]

    result = {"test": test, "const_bg_psnr": const_bg}
    root = Path(data_root) / scene
    result["ang_dist"] = angular_distance_deg(camera_positions(root / "transforms_train.json", train_idx),
                                              camera_positions(root / "transforms_test.json", test_idx))

    diag = {"B": diagnostics.background_opacity(acc, alpha), "R_hi": diagnostics.r_hi(pred, gt)}
    if (ex / "train_rgb.npy").exists():
        tr_pred = np.load(ex / "train_rgb.npy")
        tr_gt, _ = load_gt(data_root, scene, "train", train_idx)
        tr_psnr = [min(metrics.psnr(tr_pred[i], tr_gt[i]), metrics.PSNR_CAP) for i in range(len(train_idx))]
        result["train"] = {"psnr": tr_psnr}
        diag["G"] = diagnostics.generalisation_gap(float(np.mean(tr_psnr)), float(np.mean(test["psnr"])))
        if (ex / "train_rays_w.npy").exists():
            diag["omega_M"] = diagnostics.omega_M(np.load(ex / "train_rays_w.npy"), M)
    else:
        result["train"] = None
    result["diag"] = diag
    result["controls"] = {
        "C3_train_psnr_ge_30": controls.c3_pass(float(np.mean(result["train"]["psnr"]))) if result["train"] else None,
    }
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--train-time-s", type=float, default=float("nan"))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="data/blender400")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--M", type=int, default=20)
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    manifest = load_json(a.manifest)
    nets = () if a.no_lpips else ("alex", "vgg")
    res = evaluate_export(Path(a.run_dir) / "export", a.data_root, a.scene, manifest, M=a.M, lpips_nets=nets, device=a.device)
    res.update(run_id=a.run_id, scene=a.scene, arm=a.arm, n=a.n, seed=a.seed, iters=a.iters, train_time_s=a.train_time_s)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.run_id}.json").write_text(json.dumps(_clean(res), indent=1))
    print(f"{a.run_id}: test PSNR {np.mean(res['test']['psnr']):.2f} dB over {len(res['test']['psnr'])} views")


if __name__ == "__main__":
    main()
