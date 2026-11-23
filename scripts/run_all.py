#!/usr/bin/env python
"""Run the proposal's experiment matrix (Table 8). Re-runnable: finished runs are skipped.

   python scripts/run_all.py --list                       # the matrix with costs
   python scripts/run_all.py --only E1 --iters 30000      # one experiment (E1, E2a, E2b, E3, E4)
   python scripts/run_all.py --only E2a,E2b --dry-run     # print the commands only
   python scripts/run_all.py --extensions --only EXT      # third seeds for B,C,E and X1,X2
Tuning (E0) is separate: scripts/tune.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from freenerf_ns import plan  # noqa: E402
from run_arm import execute  # noqa: E402


def stage_outputs(run_id: str, lead: bool, dry: bool) -> None:
    """After every finished run: a preview image of it, and (one shard only) PROGRESS.md + a main-table snapshot,
    so a job that runs out of GPU time still leaves results to show. Never fatal."""
    if dry or not (Path("results/runs") / f"{run_id}.json").exists():
        return
    import subprocess
    for what in (["preview", "--run-id", run_id], ["progress"] if lead else None):
        if what:
            subprocess.run([sys.executable, "scripts/stage_outputs.py", *what], check=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=None, help="comma-separated experiment ids")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--iters", type=int, default=30000, help="iterations I fixed at Gate 1 by the budget rule")
    ap.add_argument("--frozen", default="configs/frozen.json")
    ap.add_argument("--extensions", action="store_true")
    ap.add_argument("--plan", default="core", choices=["core", "submission"],
                    help="core = full proposal matrix; submission = the reduced matrix actually run (see README)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--shard", default=None, help="k/n: run every n-th run starting at k (one shard per GPU)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--deadline-hours", type=float, default=0,
                    help="stop starting new runs once elapsed + mean run time would exceed this (Kaggle 12 h limit)")
    ap.add_argument("--rays", type=int, default=4096, help="rays per training batch (same for every arm)")
    ap.add_argument("--amp", action="store_true", help="mixed-precision training for every arm")
    a = ap.parse_args(argv)

    base = plan.submission_runs() if a.plan == "submission" else plan.core_runs(a.extensions)
    runs = plan.select(base, a.only, a.scene)
    if not a.list and not a.dry_run:     # drop finished runs before sharding so both GPUs get equal new work
        runs = [r for r in runs if not (Path("results/runs") / f"{r.run_id}.json").exists()]
    if a.shard:
        k, n = (int(x) for x in a.shard.split("/"))
        runs = runs[k::n]
    if a.list:
        for r in runs:
            print(f"{r.exp:4s} {r.run_id:28s} cost {r.cost:.2f} SRE")
        print(f"\n{len(runs)} runs, {plan.total_cost(runs):.1f} standard-run equivalents (SRE).  GPU-hours ~= SRE x tau(I).")
        return 0
    frozen = json.loads(Path(a.frozen).read_text()) if Path(a.frozen).exists() else None
    failed, skipped, durations, t_start = [], [], [], time.time()
    for r in runs:
        if a.deadline_hours and not a.dry_run:
            est = (sum(durations) / len(durations)) if durations else 0.0
            if (time.time() - t_start + est) / 3600 > a.deadline_hours:
                skipped.append(r.run_id)
                continue
        t_run = time.time()
        orbit = 60 if (r.exp in ("E2a", "E2b", "S-main", "S-opt") and r.arm in ("A", "D", "E") and r.n == 8
                       and r.seed == 0 and r.scene != "real") else 0
        try:
            execute(r.scene, r.arm, r.n, r.seed, a.iters, frozen=frozen, orbit=orbit, dry_run=a.dry_run,
                    lpips=not a.no_lpips, device=a.device, rays=a.rays, amp=a.amp)
            stage_outputs(r.run_id, lead=(not a.shard or a.shard.startswith("0/")), dry=a.dry_run)
        except Exception:
            traceback.print_exc()
            failed.append(r.run_id)
        durations.append(time.time() - t_run)
    print(f"\ndone: {len(runs) - len(failed) - len(skipped)}/{len(runs)} ok" + (f"; failed: {failed}" if failed else "")
          + (f"; not started (deadline): {skipped}" if skipped else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
