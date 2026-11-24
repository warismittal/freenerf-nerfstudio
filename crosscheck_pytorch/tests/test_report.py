import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sparse_nerf.report import PROTOCOLS, build_tables, recovery, write_results


def triplet(seed):
    runs = []
    for index, method in enumerate(PROTOCOLS):
        comparison = {"width": 400, "height": 300, "training_config": {"steps": 50000, "batch_size": 512}}
        scores = {"psnr": (32., 24., 28.)[index], "lpips": (.1, .3, .25)[index]}
        runs.append({
            "version": 1, "scene": "test_scene", "seed": seed, "protocol": method,
            "train_views": 100 if index == 0 else 8,
            "comparison_hash": hashlib.sha256(json.dumps(comparison, sort_keys=True).encode()).hexdigest(),
            "comparison": comparison, "frames": [{"frame": "image001", **scores}], "mean": scores,
        })
    return runs


class GapReportTests(unittest.TestCase):
    def test_paired_signs_and_scene_aggregation(self):
        second = triplet(2)
        for run in second:
            run["mean"]["psnr"] += 1
            run["frames"][0]["psnr"] += 1
        summary, paired = build_tables(triplet(1) + second)
        self.assertEqual(len(paired), 2)
        self.assertEqual(paired[0]["psnr_recovered_pct"], 50.)
        self.assertAlmostEqual(paired[0]["lpips_recovered_pct"], 25.)
        dense = next(row for row in summary if row["protocol"] == "dense-vanilla")
        self.assertEqual(dense["psnr_mean"], 32.5)
        self.assertAlmostEqual(dense["psnr_sd"], 0.5 ** 0.5)
        with TemporaryDirectory() as directory:
            output = Path(directory)
            write_results(output, summary, paired)
            markdown = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("LPIPS (AlexNet v0.1)", markdown)
            self.assertEqual(markdown.count("| test_scene | 400 × 300 | 50,000 | 512 |"), 1)

    def test_gap_edge_cases_and_unclipped_percent(self):
        self.assertIsNone(recovery("psnr", 20., 20., 21.)[2])
        self.assertIsNone(recovery("psnr", 20.01, 20., 21.)[2])
        self.assertIsNone(recovery("psnr", 20., 21., 25.)[2])
        self.assertIsNone(recovery("lpips", .2, .20001, .1)[2])
        self.assertAlmostEqual(recovery("lpips", .2, .4, .5)[2], -50.)
        self.assertAlmostEqual(recovery("psnr", 24., 20., 26.)[2], 150.)

    def test_rejects_incomplete_duplicate_and_incomparable_runs(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            build_tables(triplet(1)[:2])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_tables(triplet(1) * 2)
        runs = triplet(1)
        runs[-1]["comparison_hash"] = "different-targets-or-settings"
        with self.assertRaisesRegex(ValueError, "hash differs"):
            build_tables(runs)

    def test_rejects_mislabelled_frames_and_inconsistent_aggregates(self):
        runs = triplet(1)
        runs[0]["frames"].append(runs[0]["frames"][0])
        with self.assertRaisesRegex(ValueError, "unique test frame IDs"):
            build_tables(runs)
        runs = triplet(1)
        runs[0]["frames"][0]["frame"] = "other"
        with self.assertRaisesRegex(ValueError, "frame IDs differ"):
            build_tables(runs)
        runs = triplet(1)
        runs[0]["mean"]["psnr"] = 99.
        with self.assertRaisesRegex(ValueError, "mean differs"):
            build_tables(runs)
        runs[0]["frames"][0]["psnr"] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            build_tables(runs)
