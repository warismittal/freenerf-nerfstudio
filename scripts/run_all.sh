#!/usr/bin/env bash
# One command regenerates every number: run the experiments (finished runs are skipped), then aggregate.
#   bash scripts/run_all.sh --only E1 --iters 30000
set -uo pipefail
cd "$(dirname "$0")/.."
python scripts/run_all.py "$@"
status=$?
python -m evalkit.aggregate || echo "(aggregate skipped: not enough runs yet)"
exit $status
