"""Turn results/tables/*.csv + results/hypotheses.json into LaTeX tables, LaTeX number macros and a numbers.json.

The report (report/main.tex) and the slides (scripts/make_slides.py) read ONLY these generated files, so every number
in the PDF, the PPTX and the README comes from the same run JSONs and cannot drift.

    python scripts/make_report_assets.py --results results --out report/generated
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

ARM_NAME = {"A": "Vanilla (tuned)", "Aprime": "Vanilla (ns defaults)", "B": "Freq. mask only", "C": "Occ. loss only",
            "D": "FreeNeRF (mask+occ)", "E": "RegNeRF depth-smooth", "A_perm": "Vanilla, permuted poses"}
SCENE_NAME = {"lego": "Lego", "chair": "Chair", "drums": "Drums", "ficus": "Ficus", "real": "Real"}


def ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def f(x, nd=2) -> str:
    return f"{x:.{nd}f}" if ok(x) else "--"


def pm(v, sd, nd=2) -> str:
    if not ok(v):
        return "--"
    return f"{v:.{nd}f}" + (f"$\\pm${sd:.{nd}f}" if ok(sd) else "")


def macro_name(*parts) -> str:
    s = "".join(p.capitalize() for p in parts)
    for a, b in (("0", "Zero"), ("1", "One"), ("2", "Two"), ("3", "Three"), ("4", "Four"), ("5", "Five"),
                 ("6", "Six"), ("7", "Seven"), ("8", "Eight"), ("9", "Nine"), ("_", ""), (".", "")):
        s = s.replace(a, b)
    return s


def build(results: Path, out: Path) -> dict:
    t = results / "tables"
    summary = pd.read_csv(t / "summary.csv")
    gap = pd.read_csv(t / "gap_recovery.csv") if (t / "gap_recovery.csv").stat().st_size > 1 else pd.DataFrame()
    neq_p = t / "view_equivalence.csv"
    neq = pd.read_csv(neq_p) if neq_p.exists() and neq_p.stat().st_size > 1 else pd.DataFrame()
    hyp = json.loads((results / "hypotheses.json").read_text()) if (results / "hypotheses.json").exists() else {}
    out.mkdir(parents=True, exist_ok=True)

    nums: dict = {"cells": [], "gap": [], "neq": [], "hypotheses": hyp.get("hypotheses", []),
                  "verdict": hyp.get("verdict", {})}
    for _, r in summary.iterrows():
        nums["cells"].append({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.to_dict().items()})
    for _, r in gap.iterrows():
        nums["gap"].append({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.to_dict().items()})
    for _, r in neq.iterrows():
        nums["neq"].append({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.to_dict().items()})

    def cell(scene, arm, n):
        m = summary[(summary.scene == scene) & (summary.arm == arm) & (summary.n == n)]
        return m.iloc[0] if len(m) else None

    def g(scene, arm):
        if not len(gap):
            return None
        m = gap[(gap.scene == scene) & (gap.arm == arm)]
        return m.iloc[0] if len(m) else None

    # ---------------------------------------------------------------- main table: 8 views, A vs D (+ dense reference)
    scenes = [s for s in ("lego", "chair", "drums", "ficus", "real") if s in set(summary.scene)]
    rows = []
    for s in scenes:
        dense = summary[(summary.scene == s) & (summary.arm == "A") & (summary.n > 32)]
        for arm in ("A", "D"):
            c = cell(s, arm, 8)
            if c is None:
                continue
            gg = g(s, arm) if arm != "A" else None
            rho = (f"{gg.rho_psnr:.2f} [{gg.rho_psnr_lo:.2f}, {gg.rho_psnr_hi:.2f}]" if gg is not None else "--")
            rows.append(f"{SCENE_NAME[s]} & {ARM_NAME[arm]} & {int(c.seeds)} & {pm(c.psnr, c.get('psnr_sd'))} & "
                        f"{pm(c.ssim, c.get('ssim_sd'), 3)} & {pm(c.get('lpips_alex'), c.get('lpips_alex_sd'), 3)} & {rho} \\\\")
        for _, d in dense.iterrows():
            rows.append(f"{SCENE_NAME[s]} & Vanilla, all {int(d.n)} views & {int(d.seeds)} & {f(d.psnr)} & {f(d.ssim, 3)} & "
                        f"{f(d.get('lpips_alex'), 3)} & -- \\\\")
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    (out / "main_table.tex").write_text(
        "\\begin{tabular}{llccccc}\n\\toprule\nScene & Method & Seeds & PSNR$\\uparrow$ & SSIM$\\uparrow$ & "
        "LPIPS$\\downarrow$ & $\\rho_\\mathrm{PSNR}$ [95\\% CI] \\\\\n\\midrule\n" + "\n".join(rows) +
        "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------------------------------------------------------- ablation: A / B / C / D (+E, Aprime) at 8 views, seed 0
    arows = []
    for s in scenes:
        for arm in ("Aprime", "A", "B", "C", "D", "E"):
            c = cell(s, arm, 8)
            if c is None:
                continue
            gg = g(s, arm) if arm != "A" else None
            arows.append(f"{SCENE_NAME[s]} & {ARM_NAME[arm]} & {f(c.psnr)} & {f(c.get('lpips_alex'), 3)} & "
                         f"{f(gg.dpsnr_vs_A) if gg is not None else '--'} & {f(c.get('omega_M'), 3)} & {f(c.get('B'), 3)} \\\\")
        arows.append("\\midrule")
    if arows and arows[-1] == "\\midrule":
        arows.pop()
    (out / "ablation_table.tex").write_text(
        "\\begin{tabular}{llccccc}\n\\toprule\nScene & Arm & PSNR & LPIPS & $\\Delta$PSNR vs A & $\\Omega_M$ & $B$ \\\\\n"
        "\\midrule\n" + "\n".join(arows) + "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------------------------------------------------------- views sweep (Lego 4/8/16/dense)
    vrows = []
    for n in sorted({int(x) for x in summary[(summary.scene == "lego")].n}):
        a, d = cell("lego", "A", n), cell("lego", "D", n)
        if a is None and d is None:
            continue
        vrows.append(f"{n} & {f(a.psnr) if a is not None else '--'} & {f(d.psnr) if d is not None else '--'} \\\\")
    (out / "views_table.tex").write_text(
        "\\begin{tabular}{ccc}\n\\toprule\nViews & Vanilla PSNR & FreeNeRF PSNR \\\\\n\\midrule\n" + "\n".join(vrows) +
        "\n\\bottomrule\n\\end{tabular}\n")

    # ---------------------------------------------------------------- macros for prose
    macros = []
    for s in scenes:
        for arm in ("A", "D", "B", "C", "E", "Aprime"):
            c = cell(s, arm, 8)
            if c is not None:
                macros.append(f"\\newcommand{{\\{macro_name(s, arm, 'psnr')}}}{{{f(c.psnr)}}}")
                macros.append(f"\\newcommand{{\\{macro_name(s, arm, 'lpips')}}}{{{f(c.get('lpips_alex'), 3)}}}")
        gg = g(s, "D")
        if gg is not None:
            macros.append(f"\\newcommand{{\\{macro_name(s, 'rho')}}}{{{f(gg.rho_psnr)}}}")
            macros.append(f"\\newcommand{{\\{macro_name(s, 'gap')}}}{{{f(gg.gap_dB)}}}")
            macros.append(f"\\newcommand{{\\{macro_name(s, 'dpsnr')}}}{{{f(gg.dpsnr_vs_A)}}}")
    (out / "numbers.tex").write_text("\n".join(macros) + "\n")
    (out / "numbers.json").write_text(json.dumps(nums, indent=2, default=float))
    return nums


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="report/generated")
    a = ap.parse_args(argv)
    build(Path(a.results), Path(a.out))
    print(f"wrote {a.out}/main_table.tex, ablation_table.tex, views_table.tex, numbers.tex, numbers.json")


if __name__ == "__main__":
    main()
