"""Measure the final checkpoint on every held-out, calibrated RGB image."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from tempfile import NamedTemporaryFile

import torch

from .data import SceneDataset
from .run import load_run, render_image, select_device


def image_scores(predicted, target, perceptual, device):
    """Full-image, unquantized sRGB PSNR and LPIPS AlexNet v0.1."""
    mse = (predicted.double() - target.double()).square().mean().item()
    psnr = -10 * math.log10(max(mse, 1e-12))

    def lpips_input(rgb):
        return rgb.permute(2, 0, 1).unsqueeze(0).to(device) * 2 - 1

    distance = perceptual(lpips_input(predicted), lpips_input(target)).item()
    if not (math.isfinite(psnr) and math.isfinite(distance)):
        raise ValueError("A nonfinite render or metric was encountered")
    return psnr, distance


def lpips_model():
    import lpips
    return lpips.LPIPS(net="alex", version="0.1", verbose=False).eval()


@torch.inference_mode()
def evaluate(run, device, chunk, perceptual=None):
    _, renderer, meta = load_run(run, device)
    test = SceneDataset(meta["manifest_path"], "test", downscale=meta["downscale"])
    if min(test.width, test.height) < 31:
        raise ValueError("AlexNet LPIPS requires each image dimension to be at least 31 pixels. "
                         "Choose a smaller downscale factor when training.")
    if chunk <= 0:
        raise ValueError("chunk must be a positive number of rays")
    if perceptual is None:
        perceptual = lpips_model()
    perceptual.to(device)
    frames = []
    targets_hash = hashlib.sha256()
    for i in range(len(test)):
        origins, directions, target = test.frame(i)
        predicted = render_image(
            renderer, origins, directions, test.near, test.far, device=device, chunk=chunk,
            white_background=test.white_background, regularized=meta["protocol"] == "sparse-regularized",
        )
        psnr, distance = image_scores(predicted, target, perceptual, device)
        frame_id = test.frame_id(i)
        targets_hash.update(json.dumps(frame_id).encode("utf-8"))
        targets_hash.update((target * 255).round().byte().contiguous().numpy().tobytes())
        frames.append({"frame": frame_id, "psnr": psnr, "lpips": distance})
        print(f"[{i+1}/{len(test)}] {frame_id}  PSNR {psnr:.3f} dB  LPIPS {distance:.4f}", flush=True)
    comparison = {
        "manifest_hash": meta["manifest_hash"], "target_hash": targets_hash.hexdigest(),
        "training_config": meta["training_config"], "width": test.width, "height": test.height,
        "white_background": test.white_background, "near": test.near, "far": test.far,
        "metric_definition": "full-srgb-mean-per-view-psnr-mse-floor-1e-12-lpips-alex-v0.1",
    }
    comparison_hash = hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest()
    return {
        "version": 1, "run_id": meta["run_id"], "scene": meta["scene"], "protocol": meta["protocol"], "seed": meta["seed"],
        "train_views": meta["train_views"], "comparison_hash": comparison_hash,
        "comparison": comparison, "frames": frames,
        "mean": {metric: statistics.mean(f[metric] for f in frames) for metric in ("psnr", "lpips")},
    }


def evaluate_and_save(run, device, chunk, perceptual=None):
    result = evaluate(run, device, chunk, perceptual)
    output = run / "metrics.json"
    with NamedTemporaryFile("w", dir=run, delete=False, encoding="utf-8") as temp:
        temp_path = Path(temp.name)
    try:
        temp_path.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Wrote {output}  PSNR {result['mean']['psnr']:.3f} dB  LPIPS {result['mean']['lpips']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=2048, help="rays per evaluation batch")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    evaluate_and_save(args.run, select_device(args.device), args.chunk)


if __name__ == "__main__":
    main()
