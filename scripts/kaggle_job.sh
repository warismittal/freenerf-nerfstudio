#!/usr/bin/env bash
# Non-interactive entry point for a Kaggle (or Colab) GPU session. Everything lands in /kaggle/working/out.
#   bash scripts/kaggle_job.sh setup                      install + data (idempotent)
#   bash scripts/kaggle_job.sh smoke                      integration tests + 30-iteration runs of every feature
#   bash scripts/kaggle_job.sh pilot ITERS                one timed 8-view vanilla Lego run (tau)
#   bash scripts/kaggle_job.sh runs ITERS ONLY [SCENE]    a slice of the submission plan, e.g. runs 30000 S-main lego
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=${KAGGLE_OUT:-/kaggle/working/out}
mkdir -p "$OUT"
stage=${1:-setup}

setup() {
  python -c "import nerfstudio" 2>/dev/null || pip install -q nerfstudio
  python -c "import lpips" 2>/dev/null || pip install -q lpips
  python -c "import pytest" 2>/dev/null || pip install -q pytest
  pip install -q -e . --no-deps
  python - <<'PY'
import torch, platform
print("python", platform.python_version(), "torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
  [ -f data/blender400/lego/transforms_train.json ] || bash scripts/get_data.sh
  python scripts/make_manifests.py --check
}

save() {   # copy small, committable results + logs out of the session
  mkdir -p "$OUT"
  cp -r results "$OUT/" 2>/dev/null || true
  (cd "$OUT" && zip -qr results.zip results) || true
  ls -la "$OUT" | tail -5
}

case "$stage" in
  setup) setup ;;
  smoke) setup && bash scripts/smoke_test.sh 2>&1 | tee "$OUT/smoke.log" ;;
  pilot) setup && ( time python scripts/run_arm.py --scene lego --arm A --n 8 --seed 0 --iters "${2:-30000}" --device cuda ) \
           2>&1 | tee "$OUT/pilot.log"; save ;;
  runs)  setup && python scripts/run_all.py --plan submission --iters "${2:-30000}" --only "${3:?experiment ids}" \
           ${4:+--scene "$4"} --device cuda 2>&1 | tee -a "$OUT/runs.log"; save ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
