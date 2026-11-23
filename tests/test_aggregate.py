import json
import zlib
import math
import tempfile
from pathlib import Path

import numpy as np

from evalkit import aggregate

V = 25


def make_run(scene, arm, n, seed, psnr, lpips=0.3, diag=None, time_s=1000.0, ang=None):
    rng = np.random.default_rng(zlib.crc32(f"{scene}{arm}{n}{seed}".encode()))   # deterministic across processes
    e = rng.normal(0, 0.3, V)
    p = psnr + (e - e.mean())                                                  # per-run mean is exactly `psnr`
    d = {"G": 10.0, "omega_M": 0.3, "B": 0.05, "R_hi": 0.8}
    d.update(diag or {})
    return {"run_id": f"{scene}_{arm}_n{n}_s{seed}", "scene": scene, "arm": arm, "n": n, "seed": seed, "iters": 1000,
            "train_time_s": time_s, "ang_dist": list(ang if ang is not None else np.linspace(5, 60, V)),
            "test": {"psnr": p.tolist(), "fg_psnr": p.tolist(), "ssim": np.full(V, 0.8).tolist(),
                     "lpips_alex": (lpips + rng.normal(0, 0.01, V)).tolist(), "lpips_vgg": np.full(V, lpips).tolist(),
                     "average": np.full(V, 0.1).tolist()},
            "train": {"psnr": [38.0] * 8}, "diag": d, "controls": {}}


def write(tmp, runs, index):
    res = Path(tmp) / "results"; (res / "runs").mkdir(parents=True)
    for r in runs:
        (res / "runs" / f"{r['run_id']}.json").write_text(json.dumps(r))
    man = Path(tmp) / "manifests"; man.mkdir()
    (man / "index.json").write_text(json.dumps(index))
    return res, man


INDEX = {s: {"views": [4, 8, 16, 32, 100], "dense": 100, "test": 25} for s in ("lego", "chair", "drums", "ficus")}
INDEX["real"] = {"views": [8, 40], "dense": 40, "test": 14}


def scenario(freenerf_gain=6.0):
    runs = []
    for s in ("lego", "chair", "drums", "ficus"):
        for sd in range(3):
            runs.append(make_run(s, "A", 8, sd, 20.0, lpips=0.30, diag={"G": 14.0, "omega_M": 0.40, "R_hi": 0.7}))
            runs.append(make_run(s, "D", 8, sd, 20.0 + freenerf_gain, lpips=0.18, diag={"G": 6.0, "omega_M": 0.10, "R_hi": 0.9}, time_s=1020))
        runs.append(make_run(s, "A", 100, 0, 31.0, lpips=0.08, diag={}))
    for s in ("lego", "chair"):
        for n, v in ((4, 14.0), (16, 24.0), (32, 27.0)):
            runs.append(make_run(s, "A", n, 0, v))
        for sd in range(2):
            runs.append(make_run(s, "B", 8, sd, 20 + 0.75 * freenerf_gain, diag={"omega_M": 0.38}))
            runs.append(make_run(s, "C", 8, sd, 20.4, diag={"omega_M": 0.12}))
            runs.append(make_run(s, "E", 8, sd, 22.0, diag={"R_hi": 0.6}, time_s=1700))
        runs.append(make_run(s, "Aprime", 8, 0, 17.0))
    return runs


def test_gap_recovery_matches_hand_computation():
    with tempfile.TemporaryDirectory() as t:
        res, man = write(t, scenario(), INDEX)
        out = aggregate.aggregate(res, man, n_boot=300)
        import pandas as pd
        gap = pd.read_csv(res / "tables" / "gap_recovery.csv")
        d = gap[(gap.scene == "lego") & (gap.arm == "D")].iloc[0]
        assert abs(d.rho_psnr - 6 / 11) < 0.03 and d.rho_psnr_lo < d.rho_psnr < d.rho_psnr_hi and d.gap_ok
        assert abs(d.rho_lpips - (0.30 - 0.18) / (0.30 - 0.08)) < 0.05
        e = gap[(gap.scene == "lego") & (gap.arm == "E")].iloc[0]
        assert abs(e.rho_psnr - 2 / 11) < 0.04
        ap = gap[(gap.scene == "lego") & (gap.arm == "Aprime")].iloc[0]
        assert ap.rho_psnr < 0                                                  # untuned baseline is *worse* than tuned vanilla


def test_view_equivalence_and_hypotheses_when_freenerf_works():
    with tempfile.TemporaryDirectory() as t:
        res, man = write(t, scenario(6.0), INDEX)
        out = aggregate.aggregate(res, man, n_boot=300)
        import pandas as pd
        neq = pd.read_csv(res / "tables" / "view_equivalence.csv")
        r = neq[(neq.scene == "lego") & (neq.arm == "D")].iloc[0]
        assert r.range == "in" and 16 < r.N_eq < 32                              # 26 dB lies between the 16-view and 32-view points
        st = {h["id"]: h["status"] for h in out["hypotheses"]}
        assert st["H1"] == "supported" and st["H2"] == "supported" and st["H3"] == "supported"
        assert st["H4"] == "unresolved"                                          # X1 was not run
        assert st["H5"] == "supported" and st["H6"] == "supported"
        assert out["verdict"]["verdict"] == "reproduced"
        for f in ("R1_gap_recovery", "R2_failure_signatures", "R3_quality_vs_views", "R6_distance_to_training", "R7_cost_vs_benefit"):
            assert (res / "figures" / f"{f}.png").stat().st_size > 2000, f


def test_hypotheses_and_verdict_when_freenerf_does_nothing():
    with tempfile.TemporaryDirectory() as t:
        res, man = write(t, scenario(0.0), INDEX)
        out = aggregate.aggregate(res, man, n_boot=300)
        st = {h["id"]: h["status"] for h in out["hypotheses"]}
        assert st["H2"] == "refuted" and st["H6"] == "refuted"
        assert out["verdict"]["verdict"] == "not reproduced"


def test_partial_reproduction_when_gain_is_real_but_below_threshold():
    with tempfile.TemporaryDirectory() as t:
        res, man = write(t, scenario(1.5), INDEX)
        out = aggregate.aggregate(res, man, n_boot=300)
        assert out["verdict"]["verdict"] == "partially reproduced"              # positive, CI excludes 0, but < 3 dB on dev


def test_rho_not_reported_as_ok_when_there_is_no_gap_to_recover():
    runs = scenario(6.0)
    for r in runs:                                                              # make dense no better than sparse vanilla
        if r["arm"] == "A" and r["n"] == 100:
            r["test"]["psnr"] = [20.5] * V
    with tempfile.TemporaryDirectory() as t:
        res, man = write(t, runs, INDEX)
        aggregate.aggregate(res, man, n_boot=200)
        import pandas as pd
        gap = pd.read_csv(res / "tables" / "gap_recovery.csv")
        assert not gap[gap.scene == "lego"].gap_ok.any()                        # gap < 3 dB -> "no gap to recover"


def test_stratified_bins_are_ordered_by_distance():
    with tempfile.TemporaryDirectory() as t:
        res, man = write(t, scenario(), INDEX)
        aggregate.aggregate(res, man, n_boot=200)
        import pandas as pd
        s = pd.read_csv(res / "tables" / "stratified.csv")
        g = s[(s.scene == "lego") & (s.arm == "A")].sort_values("ang_deg")
        assert list(g.bin) == ["near", "mid", "far"]


def test_main_table_markdown_is_written():
    with tempfile.TemporaryDirectory() as t:
        res, man = write(t, scenario(6.0), INDEX)
        aggregate.aggregate(res, man, n_boot=200)
        md = (res / "tables" / "main_table.md").read_text()
        assert md.startswith("| Scene | Arm |") and "| lego | D | 3 |" in md and "dense" in md
