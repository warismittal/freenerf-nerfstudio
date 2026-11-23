"""Outputs that exist after EVERY finished run, so a job that is cut short still leaves something to show.

    python scripts/stage_outputs.py preview  --run-id lego_D_n8_s0     # results/previews/<run_id>.png
    python scripts/stage_outputs.py progress                          # results/PROGRESS.md (+ stage snapshot)

preview : ground truth | render | expected depth for 3 held-out views, with per-view PSNR, from the run's export.
progress: a table of every finished run (scene, arm, views, seed, PSNR, SSIM, LPIPS, train minutes) and, when the
          aggregate succeeds on the runs so far, a copy of the main table in results/stages/<k>_runs_main_table.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def preview(run_id: str, outputs="outputs", results="results", data_root="data/blender400", manifests="manifests", k=3):
    from PIL import Image, ImageDraw

    from evalkit.io import load_gt, load_json

    res = load_json(Path(results) / "runs" / f"{run_id}.json")
    ex = Path(outputs) / run_id / "export"
    rgb, depth = np.load(ex / "test_rgb.npy"), np.load(ex / "test_depth.npy")
    scene, n = res["scene"], res["n"]
    man = load_json(Path(manifests) / f"{scene}_n{n}.json")
    psnr = np.asarray(res["test"]["psnr"])
    views = np.linspace(0, len(rgb) - 1, k).round().astype(int)
    gt, _ = load_gt(data_root, scene, "test", [man["test"][i] for i in views])
    try:
        import matplotlib
        cmap = matplotlib.colormaps["turbo"]
    except Exception:
        cmap = None
    rows = []
    for j, v in enumerate(views):
        d = np.clip((depth[v] - 2.0) / 4.0, 0, 1)
        dimg = (cmap(d)[..., :3] * 255).astype(np.uint8) if cmap else (np.stack([d] * 3, -1) * 255).astype(np.uint8)
        row = np.concatenate([(gt[j] * 255).astype(np.uint8), (rgb[v] * 255).astype(np.uint8), dimg], 1)
        im = Image.fromarray(row)
        dr = ImageDraw.Draw(im)
        w = rgb.shape[2]
        dr.text((6, 4), f"ground truth (test view {v})", fill=(0, 0, 0))
        dr.text((w + 6, 4), f"render  PSNR {psnr[v]:.2f} dB", fill=(0, 0, 0))
        dr.text((2 * w + 6, 4), "expected depth", fill=(255, 255, 255))
        rows.append(np.asarray(im))
    head = Image.new("RGB", (rows[0].shape[1], 28), "white")
    ImageDraw.Draw(head).text((6, 8), f"{run_id}: mean test PSNR {psnr.mean():.2f} dB, "
                                      f"LPIPS {np.mean(res['test'].get('lpips_alex', [np.nan])):.3f}", fill=(0, 0, 0))
    out = Path(results) / "previews"
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([np.asarray(head)] + rows, 0)).save(out / f"{run_id}.png")
    return out / f"{run_id}.png"


def progress(results="results", manifests="manifests"):
    runs = sorted((Path(results) / "runs").glob("*.json"))
    lines = ["# Progress", "", f"Updated {time.strftime('%Y-%m-%d %H:%M:%S')} UTC - {len(runs)} runs finished.", "",
             "| run | scene | arm | views | seed | PSNR | SSIM | LPIPS | train min |", "|---|---|---|---|---|---|---|---|---|"]
    for p in runs:
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        t = r["test"]
        m = lambda k: f"{np.mean(t[k]):.3f}" if k in t else "-"  # noqa: E731
        tm = f"{r['train_time_s'] / 60:.0f}" if r.get("train_time_s") else "-"
        lines.append(f"| {r['run_id']} | {r['scene']} | {r['arm']} | {r['n']} | {r['seed']} | {np.mean(t['psnr']):.2f} | "
                     f"{m('ssim')} | {m('lpips_alex')} | {tm} |")
    it = Path(results) / "iters.json"
    if it.exists():
        d = json.loads(it.read_text())
        lines += ["", f"Iterations per run: {d['iters']} ({d['s_per_iter']} s/iter, amp={d['amp']}, {d['rays']} rays)."]
    rep = Path(results) / "real_colmap_report.json"
    if rep.exists():
        lines += [f"Self-captured scene, COLMAP: {rep.read_text().strip()}"]
    (Path(results) / "PROGRESS.md").write_text("\n".join(lines) + "\n")
    for scene in ("lego", "chair", "real"):           # novel-view videos as soon as vanilla + FreeNeRF seed 0 exist
        have = all((ROOT / "outputs" / f"{scene}_{m}_n8_s0" / "export" / "test_rgb.npy").exists() for m in ("A", "D"))
        if have and not list((Path(results) / "videos").glob(f"{scene}_*.mp4")):
            subprocess.run([sys.executable, "scripts/make_videos.py", "--scene", scene, "--seed", "0", "--arms", "A", "D"],
                           check=False, cwd=ROOT)
    p = subprocess.run([sys.executable, "-m", "evalkit.aggregate", "--results", results, "--manifests", manifests,
                        "--n-boot", "2000"], capture_output=True, text=True, cwd=ROOT)
    mt = Path(results) / "tables" / "main_table.md"
    if p.returncode == 0 and mt.exists():
        st = Path(results) / "stages"
        st.mkdir(parents=True, exist_ok=True)
        shutil.copy(mt, st / f"{len(runs):02d}_runs_main_table.md")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["preview", "progress"])
    ap.add_argument("--run-id")
    ap.add_argument("--results", default="results")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--data-root", default="data/blender400")
    ap.add_argument("--manifests", default="manifests")
    a = ap.parse_args(argv)
    if a.what == "preview":
        print(preview(a.run_id, a.outputs, a.results, a.data_root, a.manifests))
    else:
        progress(a.results, a.manifests)


if __name__ == "__main__":
    main()
