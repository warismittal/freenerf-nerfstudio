#!/usr/bin/env bash
# ONE command that reproduces the main table (UCS532 brief, Section 2).
#
#   bash run_all.sh                 full reproduction on an NVIDIA GPU: data -> every run of the submission plan -> tables/figures
#   bash run_all.sh --tables-only   no GPU needed: rebuild every table, figure and hypothesis verdict from the committed
#                                   per-run results (results/runs/*.json, results/logs/*.log)
#
# Environment overrides: ITERS (default 30000), DEVICE (cuda), PLAN (submission | core), ONLY (comma-separated experiment ids)
set -euo pipefail
cd "$(dirname "$0")"
ITERS=${ITERS:-30000}
DEVICE=${DEVICE:-cuda}
PLAN=${PLAN:-submission}

if [ "${1:-}" != "--tables-only" ]; then
  [ -f data/blender400/lego/transforms_train.json ] || bash scripts/get_data.sh
  python scripts/make_manifests.py --check
  if [ -f data/blender400/real/transforms_train.json ] || ! grep -q '"real"' manifests/index.json; then :; fi
  python scripts/run_all.py --plan "$PLAN" --iters "$ITERS" --device "$DEVICE" ${ONLY:+--only "$ONLY"}
fi
python -m evalkit.aggregate --results results --manifests manifests
python scripts/run_controls.py --results results --manifests manifests || true
echo
echo "Main table: results/tables/main_table.md     Hypotheses + verdict: results/hypotheses.json"
cat results/tables/main_table.md
