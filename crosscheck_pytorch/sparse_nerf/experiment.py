"""Run all three protocols per scene, evaluate them, render both sparse movies and report."""

import argparse
import json
from pathlib import Path

from .data import image_downscale
from .evaluate import evaluate_and_save, lpips_model
from .report import build_tables, collect, validate_metrics, write_results
from .run import read_manifest, select_device
from .train import PROTOCOLS, add_settings, train
from .video import encoder_executable, make_video, movie_is_current


def cached_evaluation(scores, test_ids):
    try:
        record = json.loads(scores.read_text(encoding="utf-8"))
        if validate_metrics(record) == test_ids:
            return record["run_id"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark"))
    parser.add_argument("--chunk", type=int, default=1024, help="rays per rendering batch")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--refresh", action="store_true", help="repeat held-out evaluation and movie rendering")
    add_settings(parser)
    args = parser.parse_args()
    if min(args.chunk, args.frames, args.fps) <= 0:
        parser.error("chunk, frames and fps must be positive")
    device = select_device(args.device)
    scenes = [read_manifest(m)[0] for m in args.manifests]
    names = {s["name"] for s in scenes}
    if len(names) != len(scenes) or len({m.stem.casefold() for m in args.manifests}) != len(scenes):
        parser.error("Each scene needs a distinct scene name and manifest filename")
    if any(len(s["splits"]["dense"]) <= 8 for s in scenes):
        parser.error("The dense reference must have more than eight training images")
    for scene in scenes:
        camera = scene["camera"]
        factor = image_downscale(camera, args.downscale)
        if min(camera["w"], camera["h"]) // factor < 31:
            parser.error(f"{scene['name']}: LPIPS needs at least 31 pixels along both axes. Reduce --downscale.")
    encoder_executable()
    perceptual = None
    for path, scene in zip(args.manifests, scenes):
        test_ids = tuple(scene["frames"][index]["file_path"] for index in scene["splits"]["test"])
        for protocol in PROTOCOLS:
            run = args.output / path.stem / f"seed-{args.seed}" / protocol
            scores = run / "metrics.json"
            cached_id = cached_evaluation(scores, test_ids)
            if perceptual is None and (args.refresh or not cached_id):
                # Check metric weights before a potentially long first training run.
                perceptual = lpips_model()
            training = argparse.Namespace(**vars(args), run=run, manifest=path, protocol=protocol, resume=True)
            trained = train(training)
            run_id = json.loads((run / "run.json").read_text(encoding="utf-8"))["run_id"]
            if args.refresh or trained or cached_id != run_id:
                if perceptual is None:
                    perceptual = lpips_model()
                evaluate_and_save(run, device, args.chunk, perceptual)
                perceptual.cpu()
            if protocol.startswith("sparse-"):
                movie = run / f"novel_{args.frames}f_{args.fps}fps.mp4"
                if args.refresh or trained or not movie_is_current(movie, run_id, args.frames, args.fps):
                    make_video(run, movie, args.frames, args.fps, args.chunk, device)
                    print(f"Wrote {movie}")
    report = args.output / "reports" / f"seed-{args.seed}"
    # Include previously completed scenes when the capture is added in a later run.
    metrics = []
    for seed_dir in args.output.glob(f"*/seed-{args.seed}"):
        group = [seed_dir / protocol / "metrics.json" for protocol in PROTOCOLS]
        if all(path.is_file() for path in group):
            metrics.extend(group)
    write_results(report, *build_tables(collect(metrics)))
    print(f"Results: {report / 'report.md'}")


if __name__ == "__main__":
    main()
