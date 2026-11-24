"""Compare matched dense and sparse runs for each scene and optimization seed."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics


PROTOCOLS = ("dense-vanilla", "sparse-vanilla", "sparse-regularized")
# Avoid reporting an unstable percentage when the observed reference gap is tiny.
GAP_FLOORS = {"psnr": 0.01, "lpips": 0.0001}


def recovery(metric, dense, sparse, regularized):
    direction = 1 if metric == "psnr" else -1
    gap = direction * (dense - sparse)
    improvement = direction * (regularized - sparse)
    floor = GAP_FLOORS[metric]
    # Subtracting two reported scores can place an exact boundary a few ulps above the floor.
    percent = None if gap <= floor or math.isclose(gap, floor, rel_tol=1e-11) else 100 * improvement / gap
    return gap, improvement, percent


def collect(paths):
    files = set()
    for path in paths:
        found = [path] if path.is_file() else list(path.rglob("metrics.json"))
        if not found:
            raise ValueError(f"No metrics.json found under {path}. Evaluate each checkpoint first.")
        files.update(f.resolve() for f in found)
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(files)]


def validate_metrics(run):
    """Check the comparison fingerprint and reconstruct each mean from its frames."""
    label = f"{run['scene']}, seed {run['seed']}, {run['protocol']}"
    digest = hashlib.sha256(json.dumps(run["comparison"], sort_keys=True).encode("utf-8")).hexdigest()
    if digest != run["comparison_hash"]:
        raise ValueError(f"{label}: comparison hash differs from the stored configuration")
    frames = run["frames"]
    ids = tuple(frame["frame"] for frame in frames)
    if not ids or any(not isinstance(name, str) or not name for name in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{label}: expected a nonempty list of unique test frame IDs")
    for metric in GAP_FLOORS:
        values = [frame[metric] for frame in frames]
        stored = run["mean"][metric]
        if not all(type(value) in (float, int) and math.isfinite(value) for value in [stored, *values]):
            raise ValueError(f"{label}: invalid or nonfinite {metric} value")
        if not math.isclose(stored, statistics.mean(values), rel_tol=1e-8, abs_tol=1e-9):
            raise ValueError(f"{label}: stored {metric} mean differs from per-frame scores")
    return ids


def build_tables(records):
    groups = defaultdict(dict)
    scene_hashes = {}
    scene_frames = {}
    scene_settings = {}
    scene_scores = defaultdict(list)
    paired = []
    for run in records:
        scene, seed, protocol = run["scene"], run["seed"], run["protocol"]
        if run["version"] != 1 or protocol not in PROTOCOLS:
            raise ValueError(f"Unsupported metrics or protocol for {scene}: {protocol}")
        frame_ids = validate_metrics(run)
        if scene_frames.setdefault(scene, frame_ids) != frame_ids:
            raise ValueError(f"{scene}: ordered test frame IDs differ between runs")
        if scene_hashes.setdefault(scene, run["comparison_hash"]) != run["comparison_hash"]:
            raise ValueError(f"{scene}: comparison hash differs. Check manifest, targets, training settings and resolution.")
        config = run["comparison"]
        scene_settings[scene] = {
            "width": config["width"], "height": config["height"],
            "steps": config["training_config"]["steps"], "rays_per_update": config["training_config"]["batch_size"],
        }
        if protocol in groups[scene, seed]:
            raise ValueError(f"Duplicate run for {scene}, seed {seed}, {protocol}")
        groups[scene, seed][protocol] = run

    if not groups:
        raise ValueError("No evaluated runs supplied")
    for (scene, seed), runs in sorted(groups.items()):
        missing = set(PROTOCOLS) - runs.keys()
        if missing:
            raise ValueError(f"{scene}, seed {seed}: missing {', '.join(sorted(missing))}")
        dense, sparse, regularized = (runs[p] for p in PROTOCOLS)
        if sparse["train_views"] != 8 or regularized["train_views"] != 8 or dense["train_views"] <= 8:
            raise ValueError(f"{scene}, seed {seed}: expected two 8-view runs and a larger dense reference")
        for run in runs.values():
            scene_scores[scene, run["protocol"]].append(run)
        row = {
            "scene": scene, "seed": seed, "dense_views": dense["train_views"],
            "sparse_views": sparse["train_views"], "test_views": len(dense["frames"]),
            "comparison_hash": dense["comparison_hash"], **scene_settings[scene],
        }
        for metric in GAP_FLOORS:
            scores = [run["mean"][metric] for run in (dense, sparse, regularized)]
            for protocol, score in zip(PROTOCOLS, scores):
                row[f"{metric}_{protocol}"] = score
            gap, improvement, percent = recovery(metric, *scores)
            row[f"{metric}_gap"] = gap
            row[f"{metric}_improvement"] = improvement
            row[f"{metric}_recovered_pct"] = percent
        paired.append(row)
    summary = []
    for (scene, protocol), runs in sorted(scene_scores.items()):
        row = {"scene": scene, "protocol": protocol, "train_views": runs[0]["train_views"],
               "test_views": len(runs[0]["frames"]), "seeds": len(runs),
               "comparison_hash": runs[0]["comparison_hash"], **scene_settings[scene]}
        for metric in GAP_FLOORS:
            scores = [run["mean"][metric] for run in runs]
            row[f"{metric}_mean"] = statistics.mean(scores)
            row[f"{metric}_sd"] = statistics.stdev(scores) if len(scores) > 1 else None
        summary.append(row)
    return summary, paired


def write_results(output, summary, paired):
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("summary.csv", summary), ("paired.csv", paired)):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    lines = ["# Held-out novel-view results", "",
             "Each run averages full-image sRGB PSNR and LPIPS (AlexNet v0.1) over the complete test split. "
             "With multiple seeds, the table gives the mean ± sample standard deviation of those run means.", "",
             "| Scene | Training protocol | Train views | Test views | Seeds | PSNR ↑ (dB) | LPIPS ↓ |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        values = []
        for metric, digits in (("psnr", 3), ("lpips", 4)):
            value = f"{row[metric + '_mean']:.{digits}f}"
            sd = row[metric + "_sd"]
            if sd is not None:
                value += f" ± {sd:.{digits}f}"
            values.append(value)
        lines.append(f"| {row['scene']} | {row['protocol']} | {row['train_views']} | {row['test_views']} "
                     f"| {row['seeds']} | {values[0]} | {values[1]} |")
    lines.extend(["", "## Experiment settings", "",
                  "| Scene | Evaluation resolution (width × height) | Training steps per run | Rays per update |",
                  "| --- | ---: | ---: | ---: |"])
    for row in summary:
        if row["protocol"] == "dense-vanilla":
            lines.append(f"| {row['scene']} | {row['width']} × {row['height']} | {row['steps']:,} "
                         f"| {row['rays_per_update']:,} |")
    lines.extend(["", "## Paired gap recovery", "",
                  "A positive improvement means the regularised model scored better than sparse vanilla. "
                  "Recovery is 100 × improvement / dense-reference gap, computed separately for each seed. "
                  "A negative percentage indicates deterioration. Values above 100% exceed the dense reference. "
                  "A dash marks a gap at or below 0.01 dB PSNR or 0.0001 LPIPS, including reversed gaps.", "",
                  "| Scene | Seed | PSNR gap | PSNR improvement | PSNR recovered | LPIPS gap | LPIPS improvement | LPIPS recovered |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for row in paired:
        values = []
        for metric, digits in (("psnr", 3), ("lpips", 4)):
            percent = row[metric + "_recovered_pct"]
            values.extend([f"{row[metric + '_gap']:.{digits}f}", f"{row[metric + '_improvement']:+.{digits}f}",
                           "—" if percent is None else f"{percent:+.1f}%"])
        lines.append("| " + " | ".join([row["scene"], str(row["seed"]), *values]) + " |")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="metrics.json files or directories to search")
    parser.add_argument("--output", type=Path, required=True, help="destination directory for CSVs and Markdown")
    args = parser.parse_args()
    try:
        summary, paired = build_tables(collect(args.runs))
    except (ValueError, KeyError) as exc:
        parser.exit(1, f"Cannot compare runs: {exc}\n")
    write_results(args.output, summary, paired)
    print(f"Wrote {args.output / 'report.md'}")


if __name__ == "__main__":
    main()
