#!/usr/bin/env python
"""Run every control that can be checked and print a PASS/FAIL table (Gate 1 / Gate 3).  python scripts/run_controls.py"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evalkit import controls  # noqa: E402
from evalkit.aggregate import cell, load_runs, views  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--manifests", default="manifests")
ap.add_argument("--sparse", default="data/sparse")
a = ap.parse_args()

rows = []


def add(name, ok, note=""):
    rows.append({"control": name, "result": "PASS" if ok else ("FAIL" if ok is False else "n/a"), "note": note})


add("C5 manifest hashes", not controls.verify_hashes(a.manifests), str(controls.verify_hashes(a.manifests)))
for p in sorted(Path(a.sparse).glob("*_n*")) if Path(a.sparse).exists() else []:
    if p.name.endswith("_perm"):
        continue
    scene, n = p.name.rsplit("_n", 1)
    man = json.loads((Path(a.manifests) / f"{scene}_n{n}.json").read_text())
    bad = controls.check_sparse_dir(p, man)
    add(f"C5 isolation {p.name}", not bad, "; ".join(bad))
add("C8 metric identities", all(controls.c8_metric_identities().values()), str(controls.c8_metric_identities()))
add("C9 diagnostic identities", all(controls.c9_diagnostic_identities().values()), str(controls.c9_diagnostic_identities()))

runs = load_runs(a.results) if (Path(a.results) / "runs").exists() else []
idx = json.loads((Path(a.manifests) / "index.json").read_text()) if (Path(a.manifests) / "index.json").exists() else {}
for r in [r for r in runs if r["arm"] == "A" and r["n"] == 8 and r.get("train")]:
    add(f"C3 overfit {r['scene']} s{r['seed']}", controls.c3_pass(float(np.mean(r["train"]["psnr"]))), f"train {np.mean(r['train']['psnr']):.1f} dB")
if idx.get("lego"):
    d = cell(runs, "lego", "A", idx["lego"]["dense"])
    add("C4 dense Lego within 2 dB of 32.54", controls.c4_pass(float(np.mean(views(d)))) if d else None,
        f"{np.mean(views(d)):.2f} dB (400x400, fewer iterations than the paper: expect a gap; state it)" if d else "run E3 first")
for s in ("lego", "chair"):
    ns = sorted({r["n"] for r in runs if r["scene"] == s and r["arm"] == "A"})
    if len(ns) >= 3:
        add(f"C4 PSNR non-decreasing with N ({s})", controls.c4_monotone({n: float(np.mean(views(cell(runs, s, "A", n)))) for n in ns}))
for r in [r for r in runs if r["run_id"].endswith("_perm")]:
    add(f"C1 permuted poses {r['scene']}", controls.c1_pass(float(np.mean(r["test"]["psnr"])), float(np.mean(r["const_bg_psnr"]))),
        f"test {np.mean(r['test']['psnr']):.1f} dB vs constant background {np.mean(r['const_bg_psnr']):.1f} dB")
for s in ("drums", "lego"):
    a_seeds, c2 = cell(runs, s, "A", 8), cell(runs, s, "C2", 8)
    if a_seeds and c2:
        same = float(np.mean(views([r for r in a_seeds if r["seed"] == c2[0]["seed"]][:1]))) if [r for r in a_seeds if r["seed"] == c2[0]["seed"]] else None
        if same is not None:
            add(f"C2 identity ({s})", abs(same - float(np.mean(views(c2[:1])))) < 0.05, f"A {same:.3f} vs C2 {np.mean(views(c2[:1])):.3f} dB")
for s in ("drums", "ficus"):
    seeds = cell(runs, s, "A", 8)
    if len(seeds) >= 3:
        ps = [float(np.mean(r["test"]["psnr"])) for r in seeds]
        add(f"C7 seed noise {s}", True, f"sd {np.std(ps, ddof=1):.2f} dB -> gains below {2*np.std(ps, ddof=1):.2f} dB are 'unresolved'")

w = max(len(r["control"]) for r in rows)
for r in rows:
    print(f"{r['control']:{w}s}  {r['result']:4s}  {r['note']}")
Path(a.results).mkdir(exist_ok=True)
(Path(a.results) / "controls.json").write_text(json.dumps(rows, indent=1))
sys.exit(1 if any(r["result"] == "FAIL" for r in rows) else 0)
