#!/usr/bin/env python
"""E0: equal-budget tuning on the development scenes only (Drums, Ficus), 6 trials per tuned arm, same seed for every trial.

   python scripts/tune.py --arm A --iters 10000      # learning rate
   python scripts/tune.py --arm D --iters 10000      # curriculum end T and lambda_occ (uses A's frozen lr)
   python scripts/tune.py --arm E --iters 10000      # lambda_DS and patch size
Each trial = 2 short runs (one per dev scene); the objective is the mean test PSNR over the two scenes. The best parameters
are written into configs/frozen.json; then run scripts/freeze_config.py and tag `config-frozen` BEFORE touching Lego/Chair/real.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_arm import execute  # noqa: E402

DEV = ["drums", "ficus"]
SPACES = {
    "A": {"lr": [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3]},
    "D": {"freq_end_frac": [0.2, 0.4, 0.6, 0.7, 0.9], "occ_weight": [0.003, 0.01, 0.03]},
    "E": {"ds_weight": [0.01, 0.1, 1.0], "ds_patch_size": [4, 8]},
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=list(SPACES))
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--frozen", default="configs/frozen.json")
    ap.add_argument("--results", default="results/tuning")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    import optuna

    fpath = Path(a.frozen)
    frozen = json.loads(fpath.read_text()) if fpath.exists() else {}
    space = SPACES[a.arm]
    sampler = (optuna.samplers.GridSampler(space) if a.arm == "E" else optuna.samplers.RandomSampler(seed=0))
    Path(a.results).mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"tune_{a.arm}",
                                storage=f"sqlite:///{a.results}/tuning.db", load_if_exists=True)

    def objective(trial):
        params = {k: trial.suggest_categorical(k, v) for k, v in space.items()}
        psnr, lp = [], []
        for scene in DEV:
            rid = f"tune_{a.arm}_t{trial.number}_{scene}"
            execute(scene, a.arm, 8, 0, a.iters, frozen={**frozen, a.arm: {**frozen.get(a.arm, {}), **params}}, run_id=rid,
                    results_dir=a.results, out_dir="outputs/tuning", device=a.device)
            r = json.loads((Path(a.results) / f"{rid}.json").read_text())
            psnr.append(float(np.mean(r["test"]["psnr"])))
            lp.append(float(np.mean(r["test"].get("lpips_alex", [np.nan]))))
        row = {"arm": a.arm, "trial": trial.number, "params": params, "psnr": float(np.mean(psnr)),
               "lpips_alex": None if np.isnan(lp).any() else float(np.mean(lp)), "per_scene_psnr": dict(zip(DEV, psnr))}
        (Path(a.results) / f"{a.arm}_t{trial.number}.json").write_text(json.dumps(row, indent=1))
        return row["psnr"]

    done = len([t for t in study.trials if t.state.name == "COMPLETE"])
    study.optimize(objective, n_trials=max(a.trials - done, 0))
    best = study.best_trial.params
    frozen[a.arm] = {**frozen.get(a.arm, {}), **best}
    fpath.parent.mkdir(parents=True, exist_ok=True)
    fpath.write_text(json.dumps(frozen, indent=2))
    print(f"best for {a.arm}: {best}  (mean dev PSNR {study.best_value:.2f})  -> {fpath}")


if __name__ == "__main__":
    main()
