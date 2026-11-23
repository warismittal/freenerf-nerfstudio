#!/usr/bin/env python
"""Finalise configs/frozen.json: check the tuning budgets were equal, stamp it with a hash, print the tag commands.
   python scripts/freeze_config.py"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from freenerf_ns import arms  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frozen", default="configs/frozen.json")
ap.add_argument("--tuning", default="results/tuning")
a = ap.parse_args()

p = Path(a.frozen)
cfg = json.loads(p.read_text()) if p.exists() else {}
problems = []
counts = {arm: len(list(Path(a.tuning).glob(f"{arm}_t*.json"))) for arm in ("A", "D", "E")}
if len(set(counts.values())) != 1:
    problems.append(f"unequal tuning budgets: {counts}")
for arm in ("A", "D", "E"):
    if arm not in cfg:
        problems.append(f"arm {arm} has no tuned values (defaults would be used): {arms.DEFAULTS[arm]}")
body = {k: v for k, v in cfg.items() if k != "_meta"}
digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
cfg["_meta"] = {"frozen": not problems, "sha": digest, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "trials": counts}
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(cfg, indent=2))
print(json.dumps(cfg, indent=2))
if problems:
    print("\nNOT FROZEN:\n  " + "\n  ".join(problems))
    sys.exit(1)
print(f"\nFROZEN ({digest}). Now:\n  git add configs/frozen.json results/tuning && git commit -m 'Freeze configuration' && git tag config-frozen && git push --tags")
