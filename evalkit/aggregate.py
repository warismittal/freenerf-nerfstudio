"""Turn results/runs/*.json into the report: tables, gap-recovery ratios, view-equivalence, hypotheses H1-H6,
the reproduction verdict and figures R1-R4, R6, R7.   python -m evalkit.aggregate [--results results] [--manifests manifests]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import stats
from .io import load_json

REPORTED = ["lego", "chair"]
DEV = ["drums", "ficus"]
ARM_ORDER = ["A", "Aprime", "B", "C", "D", "E", "X1", "X2"]


# ----------------------------------------------------------------------------------------------------------- loading
def load_runs(results_dir) -> list:
    d = Path(results_dir) / "runs"
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def cell(runs, scene, arm, n):
    return [r for r in runs if r["scene"] == scene and r["arm"] == arm and r["n"] == n]


def views(cell_runs, key="psnr", group="test") -> np.ndarray:
    """[seeds, views] array of per-view values."""
    return np.array([r[group][key] for r in cell_runs], dtype=np.float64)


def scalar(cell_runs, section, key):
    vals = [r[section].get(key) if section != "_" else r.get(key) for r in cell_runs]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def dense_n(index, scene):
    return int(index[scene]["dense"])


# ----------------------------------------------------------------------------------------------------------- tables
def summary_table(runs) -> pd.DataFrame:
    rows = []
    keys = sorted({(r["scene"], r["arm"], r["n"]) for r in runs})
    for scene, arm, n in keys:
        c = cell(runs, scene, arm, n)
        row = {"scene": scene, "arm": arm, "n": n, "seeds": len(c)}
        for k in ("psnr", "fg_psnr", "ssim", "lpips_alex", "lpips_vgg", "average"):
            if k in c[0]["test"]:
                per_seed = [np.mean(r["test"][k]) for r in c]
                row[k] = float(np.mean(per_seed))
                row[k + "_sd"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else float("nan")
        for k in ("G", "omega_M", "B", "R_hi"):
            row[k] = scalar(c, "diag", k)
        row["train_time_s"] = float(np.nanmean([r["train_time_s"] if r.get("train_time_s") is not None else np.nan for r in c]))
        rows.append(row)
    return pd.DataFrame(rows)


def gap_recovery_table(runs, index, n_boot=10_000) -> pd.DataFrame:
    rows = []
    for scene in REPORTED + ["real"]:
        dn = dense_n(index, scene) if scene in index else None
        dense, van = (cell(runs, scene, "A", dn) if dn else []), cell(runs, scene, "A", 8)
        if not dense or not van:
            continue
        gap = stats.diff_with_ci(views(dense), views(van), n_boot)
        gap_ok = bool(gap["diff"] >= 3.0 and gap["lo"] > 0)
        for arm in [a for a in ARM_ORDER if a != "A"]:
            reg = cell(runs, scene, arm, 8)
            if not reg:
                continue
            row = {"scene": scene, "arm": arm, "gap_dB": gap["diff"], "gap_ok": gap_ok}
            p = stats.rho_with_ci(views(reg), views(van), views(dense), "psnr", n_boot)
            row.update(rho_psnr=p["rho"], rho_psnr_lo=p["lo"], rho_psnr_hi=p["hi"])
            if "lpips_alex" in reg[0]["test"]:
                q = stats.rho_with_ci(views(reg, "lpips_alex"), views(van, "lpips_alex"), views(dense, "lpips_alex"), "lpips", n_boot)
                row.update(rho_lpips=q["rho"], rho_lpips_lo=q["lo"], rho_lpips_hi=q["hi"])
            row["dpsnr_vs_A"] = float(np.mean(views(reg)) - np.mean(views(van)))
            rows.append(row)
    return pd.DataFrame(rows)


def view_equivalence_table(runs, index) -> pd.DataFrame:
    rows = []
    for scene in REPORTED:
        if scene not in index:
            continue
        ns = [n for n in sorted({r["n"] for r in runs if r["scene"] == scene and r["arm"] == "A"})]
        if len(ns) < 3:
            continue
        van = [float(np.mean(views(cell(runs, scene, "A", n)))) for n in ns]
        for arm in [a for a in ARM_ORDER if a != "A"]:
            reg = cell(runs, scene, arm, 8)
            if reg:
                target = float(np.mean(views(reg)))
                neq, status = stats.n_equivalent(target, ns, van)
                rows.append({"scene": scene, "arm": arm, "psnr_8view": target, "N_eq": neq, "range": status})
    return pd.DataFrame(rows)


def stratified_table(runs) -> pd.DataFrame:
    """Per-view PSNR in three equal-count bins of angular distance to the nearest training camera (8-view runs)."""
    rows = []
    for scene, arm in sorted({(r["scene"], r["arm"]) for r in runs if r["n"] == 8}):
        c = cell(runs, scene, arm, 8)
        ang = np.array(c[0]["ang_dist"])
        p = views(c).mean(axis=0)
        order = np.argsort(ang)
        for b, idx in enumerate(np.array_split(order, 3)):
            rows.append({"scene": scene, "arm": arm, "bin": ["near", "mid", "far"][b], "ang_deg": float(ang[idx].mean()),
                         "psnr": float(p[idx].mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------- hypotheses
def _arm_mean(runs, scene, arm, key="psnr", n=8):
    c = cell(runs, scene, arm, n)
    return float(np.mean(views(c, key))) if c else float("nan")


def _diag(runs, scene, arm, key, n=8):
    c = cell(runs, scene, arm, n)
    return scalar(c, "diag", key) if c else float("nan")


def _status(flags):
    flags = [f for f in flags]
    if any(f is None for f in flags):
        return "unresolved"
    return "supported" if all(flags) else "refuted"


def evaluate_hypotheses(runs, index, gap_df, neq_df) -> list:
    out = []
    # H1 collapse
    f = []
    for s in REPORTED + DEV:
        if cell(runs, s, "A", 8):
            g = _diag(runs, s, "A", "G")
            f.append(bool(g >= 8 and _arm_mean(runs, s, "A") <= 20))
    out.append({"id": "H1", "text": "vanilla collapses at 8 views: G>=8 dB and test PSNR<=20 dB on at least one synthetic scene",
                "status": ("supported" if any(f) else "refuted") if f else "unresolved"})
    # H2 primary
    f = []
    for s in REPORTED:
        r = gap_df[(gap_df.scene == s) & (gap_df.arm == "D")] if len(gap_df) else gap_df
        if len(r) and cell(runs, s, "A", 8):
            row = r.iloc[0]
            lp = _arm_mean(runs, s, "A", "lpips_alex") - _arm_mean(runs, s, "D", "lpips_alex")
            f.append(bool(row.gap_ok and row.rho_psnr >= 0.30 and row.rho_psnr_lo > 0 and lp >= 0.05))
        else:
            f.append(None)
    out.append({"id": "H2", "primary": True,
                "text": "FreeNeRF rescues: rho_PSNR>=0.30 (CI excludes 0) and LPIPS lower by >=0.05 on both Lego and Chair",
                "status": _status(f)})
    # H3 dissociation
    f = []
    for s in REPORTED:
        arms = {a: _arm_mean(runs, s, a) for a in "ABCD"}
        if any(np.isnan(v) for v in arms.values()):
            f.append(None)
            continue
        dgain, bgain, cgain = arms["D"] - arms["A"], arms["B"] - arms["A"], arms["C"] - arms["A"]
        om_a, om_c = _diag(runs, s, "A", "omega_M"), _diag(runs, s, "C", "omega_M")
        f.append(bool(bgain >= 0.6 * dgain and cgain < 1.0 and om_c <= 0.5 * om_a))
    out.append({"id": "H3", "text": "dissociation: B gives >=60% of D's gain; C cuts Omega_M by >=50% but gains <1 dB", "status": _status(f)})
    # H4 order
    b, x1 = _arm_mean(runs, "lego", "B"), _arm_mean(runs, "lego", "X1")
    out.append({"id": "H4", "text": "order matters: X1 at least 1 dB below B",
                "status": "unresolved" if np.isnan(b) or np.isnan(x1) else ("supported" if x1 <= b - 1.0 else "refuted")})
    # H5 smoothness blurs and costs
    rd, re_ = _diag(runs, "lego", "D", "R_hi"), _diag(runs, "lego", "E", "R_hi")
    cd, ce = cell(runs, "lego", "D", 8), cell(runs, "lego", "E", 8)
    if cd and ce and not (np.isnan(rd) or np.isnan(re_)):
        tD = np.nanmean([r.get("train_time_s") or np.nan for r in cd])
        tE = np.nanmean([r.get("train_time_s") or np.nan for r in ce])
        ok_time = None if np.isnan(tD) or np.isnan(tE) else bool(tE >= 1.3 * tD)
        st = _status([bool(re_ < rd), ok_time])
    else:
        st = "unresolved"
    out.append({"id": "H5", "text": "smoothness blurs and costs: R_hi(E)<R_hi(D) on Lego and E takes >=1.3x D's training time", "status": st})
    # H6 view-equivalence
    f = []
    for s in REPORTED:
        r = neq_df[(neq_df.scene == s) & (neq_df.arm == "D")] if len(neq_df) else neq_df
        f.append(bool(r.iloc[0].N_eq >= 16) if len(r) else None)
    out.append({"id": "H6", "text": "view-equivalence: N_eq(D, 8) >= 16 on both synthetic scenes", "status": _status(f)})
    return out


def reproduction_verdict(runs, index, gap_df, hyps) -> dict:
    """Verdict scale fixed in the proposal (Section 12)."""
    d_a = []
    for s in DEV:
        a, d = cell(runs, s, "A", 8), cell(runs, s, "D", 8)
        if a and d:
            d_a.append(float(np.mean(views(d)) - np.mean(views(a))))
    h2 = next(h for h in hyps if h["id"] == "H2")["status"]
    if len(d_a) < 2:
        return {"verdict": "pending", "reason": "development-scene runs (E1) missing"}
    if all(x >= 3.0 for x in d_a) and h2 == "supported":
        return {"verdict": "reproduced", "dev_gain_dB": d_a}
    pos = []
    for s in REPORTED:
        a, d = cell(runs, s, "A", 8), cell(runs, s, "D", 8)
        if a and d:
            ci = stats.diff_with_ci(views(d), views(a), 2000)
            pos.append(ci["lo"] > 0)
    if any(pos):
        return {"verdict": "partially reproduced", "dev_gain_dB": d_a}
    return {"verdict": "not reproduced", "dev_gain_dB": d_a}


# ----------------------------------------------------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_r1(gap_df, out):
    if not len(gap_df):
        return
    plt = _plt()
    scenes = list(gap_df.scene.unique())
    arms = [a for a in ARM_ORDER if a in set(gap_df.arm)]
    fig, ax = plt.subplots(figsize=(1.6 * len(scenes) * len(arms) / 2 + 3, 3.6))
    w = 0.8 / len(arms)
    for j, arm in enumerate(arms):
        d = gap_df[gap_df.arm == arm].set_index("scene").reindex(scenes)
        x = np.arange(len(scenes)) + j * w
        ax.bar(x, d.rho_psnr, w, yerr=[d.rho_psnr - d.rho_psnr_lo, d.rho_psnr_hi - d.rho_psnr], label=arm, capsize=2)
    ax.axhline(0, c="k", lw=.8), ax.axhline(1, c="k", lw=.8, ls=":")
    ax.set_xticks(np.arange(len(scenes)) + 0.4 - w / 2, scenes), ax.set_ylabel("gap-recovery ratio rho (PSNR)")
    ax.legend(ncol=len(arms), fontsize=8), fig.tight_layout(), fig.savefig(out, dpi=180), plt.close(fig)


def fig_r2(summary, out):
    plt = _plt()
    keys = [("G", "gen. gap G (dB)"), ("omega_M", "near-camera opacity"), ("B", "background opacity B"), ("R_hi", "R_hi")]
    scenes = [s for s in REPORTED if s in set(summary.scene)]
    if not scenes:
        return
    fig, axes = plt.subplots(len(scenes), 4, figsize=(11, 2.6 * len(scenes)), squeeze=False)
    for i, s in enumerate(scenes):
        sub = summary[(summary.scene == s) & (summary.n == 8)].set_index("arm")
        for j, (k, lab) in enumerate(keys):
            if "A" not in sub.index or k not in sub:
                continue
            d = (sub[k] - sub.loc["A", k]).drop("A").dropna()
            axes[i, j].bar(d.index, d.values), axes[i, j].axhline(0, c="k", lw=.6)
            axes[i, j].set_title(f"{s}: change in {lab}", fontsize=8)
    fig.tight_layout(), fig.savefig(out, dpi=180), plt.close(fig)


def fig_r3(runs, index, out):
    plt = _plt()
    scenes = [s for s in REPORTED if s in index]
    fig, axes = plt.subplots(1, max(len(scenes), 1), figsize=(4.6 * max(len(scenes), 1), 3.4), squeeze=False)
    for ax, s in zip(axes[0], scenes):
        ns = sorted({r["n"] for r in runs if r["scene"] == s and r["arm"] == "A"})
        if len(ns) < 2:
            continue
        ax.plot(ns, [np.mean(views(cell(runs, s, "A", n))) for n in ns], "o-", label="vanilla (A)")
        dn = [n for n in ns if cell(runs, s, "D", n)]
        if dn:
            ax.plot(dn, [np.mean(views(cell(runs, s, "D", n))) for n in dn], "s--", label="FreeNeRF (D)")
        d8 = cell(runs, s, "D", 8)
        if d8:
            ax.axhline(np.mean(views(d8)), c="C1", lw=.8, ls=":")
        ax.set_xscale("log"), ax.set_xlabel("views N"), ax.set_ylabel("test PSNR (dB)"), ax.set_title(s), ax.legend(fontsize=8)
    fig.tight_layout(), fig.savefig(out, dpi=180), plt.close(fig)


def fig_r4(results_dir, out):
    tdir = Path(results_dir) / "tuning"
    files = sorted(tdir.glob("*.json")) if tdir.exists() else []
    rows = [json.loads(p.read_text()) for p in files]
    rows = [r for r in rows if r.get("arm") == "D" and r.get("lpips_alex") is not None]
    if not rows:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    sc = ax.scatter([r["lpips_alex"] for r in rows], [r["psnr"] for r in rows], c=[r["params"].get("freq_end_frac", 0) for r in rows])
    fig.colorbar(sc, label="curriculum end T / I"), ax.set_xlabel("LPIPS (alex)"), ax.set_ylabel("PSNR (dB)")
    fig.tight_layout(), fig.savefig(out, dpi=180), plt.close(fig)


def fig_r6(strat, out):
    if not len(strat):
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for (s, a), g in strat.groupby(["scene", "arm"]):
        if s in REPORTED and a in ("A", "D"):
            ax.plot(g.ang_deg, g.psnr, "o-", label=f"{s} {a}")
    ax.set_xlabel("angular distance to nearest training camera (deg)"), ax.set_ylabel("test PSNR (dB)"), ax.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(out, dpi=180), plt.close(fig)


def fig_r7(summary, gap_df, out):
    if not len(gap_df):
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for s in [x for x in REPORTED if x in set(gap_df.scene)]:
        sub = summary[(summary.scene == s) & (summary.n == 8)].set_index("arm")
        if "A" not in sub.index or np.isnan(sub.loc["A", "train_time_s"]):
            continue
        for _, r in gap_df[gap_df.scene == s].iterrows():
            if r.arm in sub.index:
                ax.scatter(sub.loc[r.arm, "train_time_s"] / sub.loc["A", "train_time_s"], r.rho_psnr)
                ax.annotate(f"{s[:2]}-{r.arm}", (sub.loc[r.arm, "train_time_s"] / sub.loc["A", "train_time_s"], r.rho_psnr), fontsize=6)
    ax.set_xlabel("training-time multiplier vs A"), ax.set_ylabel("rho (PSNR)")
    fig.tight_layout(), fig.savefig(out, dpi=180), plt.close(fig)


# ----------------------------------------------------------------------------------------------------------- markdown
def _pm(row, k, nd=2):
    v, sd = row.get(k), row.get(k + "_sd")
    if v is None or v != v:
        return "-"
    return f"{v:.{nd}f}" + (f" ± {sd:.{nd}f}" if sd == sd and sd is not None else "")


def main_table_md(summary, gap, neq) -> str:
    """The report's main table: 8 input views, mean ± sd over seeds, gap-recovery ratio with 95% CI."""
    lines = ["| Scene | Arm | Seeds | PSNR ↑ | SSIM ↑ | LPIPS (Alex) ↓ | ρ_PSNR [95% CI] | N_eq |",
             "|---|---|---|---|---|---|---|---|"]
    rows = summary[summary.n == 8].copy()
    rows["_o"] = rows.arm.map({a: i for i, a in enumerate(ARM_ORDER)})
    for _, r in rows.sort_values(["scene", "_o"]).iterrows():
        rho, ne = "-", "-"
        if len(gap):
            g = gap[(gap.scene == r.scene) & (gap.arm == r.arm)]
            if len(g):
                g = g.iloc[0]
                rho = f"{g.rho_psnr:.2f} [{g.rho_psnr_lo:.2f}, {g.rho_psnr_hi:.2f}]" + ("" if g.gap_ok else " (no gap)")
        if len(neq):
            q = neq[(neq.scene == r.scene) & (neq.arm == r.arm)]
            if len(q):
                ne = f"{q.iloc[0].N_eq:.1f}" + ("" if q.iloc[0].range == "in" else f" ({q.iloc[0].range} range)")
        d = r.to_dict()
        lines.append(f"| {r.scene} | {r.arm} | {int(r.seeds)} | {_pm(d, 'psnr')} | {_pm(d, 'ssim', 3)} | "
                     f"{_pm(d, 'lpips_alex', 3)} | {rho} | {ne} |")
    dense = summary[(summary.arm == "A") & (summary.n > 32)]
    for _, r in dense.iterrows():
        d = r.to_dict()
        lines.append(f"| {r.scene} | A, all {int(r.n)} views (dense) | {int(r.seeds)} | {_pm(d, 'psnr')} | "
                     f"{_pm(d, 'ssim', 3)} | {_pm(d, 'lpips_alex', 3)} | - | - |")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------------------- main
def aggregate(results_dir="results", manifests="manifests", n_boot=10_000) -> dict:
    results = Path(results_dir)
    runs = load_runs(results)
    if not runs:
        raise SystemExit(f"no runs found in {results}/runs")
    index = load_json(Path(manifests) / "index.json")
    # make 'dense' runs addressable by their real view count
    summary = summary_table(runs)
    gap = gap_recovery_table(runs, index, n_boot)
    neq = view_equivalence_table(runs, index)
    strat = stratified_table(runs)
    hyps = evaluate_hypotheses(runs, index, gap, neq)
    verdict = reproduction_verdict(runs, index, gap, hyps)

    (results / "tables").mkdir(parents=True, exist_ok=True)
    (results / "figures").mkdir(parents=True, exist_ok=True)
    for name, df in [("summary", summary), ("gap_recovery", gap), ("view_equivalence", neq), ("stratified", strat)]:
        df.to_csv(results / "tables" / f"{name}.csv", index=False)
    fig_r1(gap, results / "figures" / "R1_gap_recovery.png")
    fig_r2(summary, results / "figures" / "R2_failure_signatures.png")
    fig_r3(runs, index, results / "figures" / "R3_quality_vs_views.png")
    fig_r4(results, results / "figures" / "R4_curriculum_tradeoff.png")
    fig_r6(strat, results / "figures" / "R6_distance_to_training.png")
    fig_r7(summary, gap, results / "figures" / "R7_cost_vs_benefit.png")
    (results / "tables" / "main_table.md").write_text(main_table_md(summary, gap, neq))
    out = {"hypotheses": hyps, "verdict": verdict}
    (results / "hypotheses.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--manifests", default="manifests")
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    out = aggregate(a.results, a.manifests, a.n_boot)
    for h in out["hypotheses"]:
        print(f"{h['id']}: {h['status']:11s} {h['text']}")
    print("reproduction verdict:", out["verdict"])


if __name__ == "__main__":
    main()
