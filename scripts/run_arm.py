#!/usr/bin/env python
"""One experimental run end to end: sparse dataset -> ns-train -> export -> evalkit -> results/runs/<run_id>.json

   python scripts/run_arm.py --scene lego --arm D --n 8 --seed 0 --iters 30000 [--frozen configs/frozen.json] [--orbit 60] [--dry-run]
Skips runs whose result already exists (safe to re-launch after a lost Kaggle/Colab session) unless --force.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evalkit.io import resolve_n  # noqa: E402
from freenerf_ns import arms, datasets  # noqa: E402


def sh(cmd, dry, log=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    if dry:
        return
    if log is None:
        subprocess.run(list(map(str, cmd)), check=True)
        return
    with open(log, "a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        p = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        print(Path(log).read_text()[-4000:], flush=True)
        raise subprocess.CalledProcessError(p.returncode, cmd)


def provenance() -> dict:
    import platform
    info = {"python": platform.python_version(), "host": platform.node()}
    try:
        info["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        info["git_commit"] = None
    try:
        import torch
        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        info["gpu"] = "unknown"
    return info


def lite_kwargs(device_type=None):
    """Settings that let one run finish on an 8 GB laptop without an NVIDIA GPU. Results are for learning and demos:
    200x200 numbers are NOT comparable with the 400x400 protocol in the proposal."""
    import platform
    dt = device_type or ("mps" if platform.machine() == "arm64" and platform.system() == "Darwin" else "cpu")
    return dict(rays=512, device_type=dt, size=200, eval_chunk=4096, data_root="data/blender200",
                sparse_root="data/sparse200", out_dir="outputs_lite", results_dir="results_lite/runs")


def execute(scene, arm, n, seed, iters, *, frozen=None, run_id=None, results_dir="results/runs", out_dir="outputs",
            data_root="data/blender400", manifests="manifests", sparse_root="data/sparse", orbit=0, permute=False,
            force=False, dry_run=False, lpips=True, device="cpu", rays=4096, device_type=None, size=400, eval_chunk=None, amp=False):
    kind = "real" if scene.startswith("real") else "synthetic"
    n_int = resolve_n(manifests, scene, n)
    run_id = run_id or f"{scene}_{arm}_n{n}_s{seed}" + ("_perm" if permute else "")
    result = Path(results_dir) / f"{run_id}.json"
    if result.exists() and not force:
        print(f"[skip] {run_id}: result exists")
        return result
    spec = arms.arm_spec(arm, frozen, kind)
    log = None
    if not dry_run:
        logs = Path(results_dir).parent / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{run_id}.log"
        log.write_text(json.dumps({"run_id": run_id, "scene": scene, "arm": arm, "n": n_int, "seed": seed, "iters": iters,
                                   "frozen": frozen, "rays": rays, "amp": amp, **provenance()}) + "\n")
    print(f"=== {run_id}  SEED={seed}  ITERS={iters}  RAYS={rays}  AMP={amp}", flush=True)
    data_dir = datasets.make_sparse(data_root, manifests, scene, n, sparse_root, 0 if permute else None) if not dry_run \
        else Path(sparse_root) / f"{scene}_n{n_int}"
    cmd = arms.train_command(spec, data_dir=str(data_dir), out_dir=out_dir, run_id=run_id, iters=iters, seed=seed, scene_kind=kind,
                             rays=rays, device_type=device_type, eval_chunk=eval_chunk, amp=amp)
    t0 = time.time()
    sh(cmd, dry_run, log)
    train_s = time.time() - t0
    config = Path(out_dir) / run_id / spec.method / "main" / "config.yml"
    manifest = Path(manifests) / f"{scene}_n{n_int}.json"
    export = ["python", "-m", "freenerf_ns.export", "--config", config, "--scene", scene, "--manifest", manifest,
              "--data-root", data_root, "--out", Path(out_dir) / run_id / "export", "--seed", seed,
              "--width", size, "--height", size]
    if n_int > 16:
        export.append("--no-train")
    if orbit and kind == "synthetic":
        export += ["--orbit", orbit]
    sh(export, dry_run, log)
    arm_label = arm + ("_perm" if permute else "")        # a control run must never pollute the real arm's cell
    ev = ["python", "-m", "evalkit.run_eval", "--run-dir", Path(out_dir) / run_id, "--run-id", run_id, "--scene", scene,
          "--arm", arm_label, "--n", n_int, "--seed", seed, "--iters", iters, "--train-time-s", round(train_s, 1),
          "--manifest", manifest, "--data-root", data_root, "--out", results_dir, "--device", device]
    if not lpips:
        ev.append("--no-lpips")
    sh(ev, dry_run, log)
    if log is not None:                                   # keep committed logs small: header + last 300 lines
        lines = log.read_text().splitlines()
        if len(lines) > 400:
            log.write_text("\n".join(lines[:5] + ["... (training progress lines trimmed) ..."] + lines[-300:]) + "\n")
        with open(log, "a") as f:
            f.write(f"TRAIN_SECONDS={train_s:.1f}\n")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--arm", required=True, choices=arms.ARMS)
    ap.add_argument("--n", default="8", help="views, or 'dense'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--frozen", default="configs/frozen.json")
    ap.add_argument("--orbit", type=int, default=0)
    ap.add_argument("--permute-poses", action="store_true", help="negative control C1")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-lpips", action="store_true")
    ap.add_argument("--device", default="cpu", help="device for LPIPS in evalkit (cuda on a GPU machine)")
    ap.add_argument("--lite", action="store_true",
                    help="laptop profile (8 GB Mac): 200x200 data in data/blender200, 512 rays/batch, MPS, small render chunks")
    ap.add_argument("--device-type", default=None, help="override: cuda | mps | cpu")
    ap.add_argument("--rays", type=int, default=4096, help="rays per training batch")
    ap.add_argument("--amp", action="store_true", help="mixed-precision training (fp16 autocast)")
    a = ap.parse_args(argv)
    frozen = json.loads(Path(a.frozen).read_text()) if Path(a.frozen).exists() else None
    kw = lite_kwargs(a.device_type) if a.lite else {"device_type": a.device_type, "rays": a.rays}
    kw["amp"] = a.amp
    execute(a.scene, a.arm, a.n, a.seed, a.iters, frozen=frozen, orbit=a.orbit, permute=a.permute_poses, force=a.force,
            dry_run=a.dry_run, lpips=not a.no_lpips, device=a.device, **kw)


if __name__ == "__main__":
    main()
