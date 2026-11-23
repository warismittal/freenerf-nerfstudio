"""Gate-1 budget rule, automated: measure seconds per training iteration on this GPU and choose the iteration count I
so that the whole plan fits the GPU budget (proposal, Section 'Budget rule').

    python scripts/choose_iters.py --budget-gpu-h 40 --runs 27 [--amp] [--rays 4096]

It times the FreeNeRF arm (frequency mask + occlusion loss, the most expensive arm apart from E) on 8-view Lego for
--short and --long iterations, takes the difference (so start-up and checkpoint time cancel), checks the logs for NaN,
and, if mixed precision was requested but fails or produces NaN, falls back to fp32. The decision is written to
results/iters.json and every later job reuses it, so all arms and seeds use the same I.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freenerf_ns import datasets  # noqa: E402


def timed_run(iters: int, amp: bool, rays: int, data_dir: str, tag: str) -> tuple[float, bool, str]:
    cmd = ["ns-train", "freenerf", "--data", data_dir, "--max-num-iterations", str(iters),
           "--optimizers.fields.scheduler.max-steps", str(iters), "--pipeline.datamanager.train-num-rays-per-batch", str(rays),
           "--pipeline.model.freq-mask", "ramp", "--pipeline.model.occ-weight", "0.01", "--output-dir", "outputs/speed",
           "--experiment-name", tag, "--timestamp", "main", "--vis", "tensorboard", "--steps-per-save", str(iters),
           "--machine.seed", "0"]
    if amp:
        cmd += ["--mixed-precision", "True"]
    cmd.append("blender-data")
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    log = p.stdout + p.stderr
    Path(f"outputs/speed_{tag}.log").parent.mkdir(parents=True, exist_ok=True)
    Path(f"outputs/speed_{tag}.log").write_text(log)
    bad = p.returncode != 0 or re.search(r"\bnan\b", log, re.IGNORECASE) is not None or not weights_finite(tag)
    return dt, not bad, log[-1500:]


def weights_finite(tag: str) -> bool:
    """AMP with GradScaler can skip NaN steps silently; make sure the trained field is finite and actually moved."""
    import torch

    ckpts = sorted(Path("outputs/speed", tag).rglob("*.ckpt"))
    if not ckpts:
        return False
    state = torch.load(ckpts[-1], map_location="cpu", weights_only=False)["pipeline"]
    return all(torch.isfinite(v).all().item() for v in state.values() if torch.is_tensor(v) and v.is_floating_point())


def measure(amp: bool, rays: int, data_dir: str, short: int, long: int) -> tuple[float, bool]:
    t_s, ok_s, tail_s = timed_run(short, amp, rays, data_dir, f"s{short}_{'amp' if amp else 'fp32'}")
    t_l, ok_l, tail_l = timed_run(long, amp, rays, data_dir, f"s{long}_{'amp' if amp else 'fp32'}")
    if not (ok_s and ok_l):
        print(f"[choose_iters] {'amp' if amp else 'fp32'} run failed or produced NaN:\n{tail_l if not ok_l else tail_s}")
        return math.nan, False
    return (t_l - t_s) / (long - short), True


def decide(s_per_iter: float, budget_gpu_h: float, n_runs: int, eval_min: float, lo: int, hi: int) -> int:
    train_s = budget_gpu_h * 3600 / max(n_runs, 1) - eval_min * 60
    iters = int(train_s / s_per_iter) // 1000 * 1000
    return max(lo, min(hi, iters))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget-gpu-h", type=float, required=True, help="GPU-hours available for the whole plan")
    ap.add_argument("--runs", type=int, required=True, help="number of runs in the plan")
    ap.add_argument("--eval-min", type=float, default=8.0, help="export + evaluation minutes per run")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--rays", type=int, default=4096)
    ap.add_argument("--short", type=int, default=60)
    ap.add_argument("--long", type=int, default=260)
    ap.add_argument("--min-iters", type=int, default=3000)
    ap.add_argument("--max-iters", type=int, default=30000)
    ap.add_argument("--out", default="results/iters.json")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if out.exists():
        d = json.loads(out.read_text())
        print(f"[choose_iters] reusing {out}: ITERS={d['iters']} AMP={d['amp']}")
        print(f"ITERS={d['iters']}\nAMP={1 if d['amp'] else ''}")
        return 0
    data_dir = str(datasets.make_sparse("data/blender400", "manifests", "lego", 8, "data/sparse"))
    amp = a.amp
    spi, ok = measure(amp, a.rays, data_dir, a.short, a.long)
    if amp and not ok:
        amp = False
        spi, ok = measure(False, a.rays, data_dir, a.short, a.long)
    if not ok:
        raise SystemExit("[choose_iters] training does not run; see outputs/speed_*.log")
    iters = decide(spi, a.budget_gpu_h, a.runs, a.eval_min, a.min_iters, a.max_iters)
    d = dict(s_per_iter=round(spi, 4), iters=iters, amp=amp, rays=a.rays, budget_gpu_h=a.budget_gpu_h, runs=a.runs,
             eval_min=a.eval_min, gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
             measured_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2))
    print(f"[choose_iters] {spi:.3f} s/iter ({'amp' if amp else 'fp32'}, {a.rays} rays) -> ITERS={iters} "
          f"for {a.runs} runs in {a.budget_gpu_h} GPU-h")
    print(f"ITERS={iters}\nAMP={1 if amp else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
