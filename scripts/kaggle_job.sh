#!/usr/bin/env bash
# Non-interactive entry point for a Kaggle (or Colab) GPU session. Everything lands in /kaggle/working/out.
#   bash scripts/kaggle_job.sh setup                      install + data (idempotent)
#   bash scripts/kaggle_job.sh smoke                      integration tests + 30-iteration runs of every feature
#   bash scripts/kaggle_job.sh pilot ITERS                one timed 8-view vanilla Lego run (tau)
#   bash scripts/kaggle_job.sh runs ITERS ONLY [SCENE]    a slice of the submission plan, e.g. runs 30000 S-main lego
#        ITERS=auto measures s/iter and applies the budget rule (env BUDGET_GPU_H, AMP=1, RAYS, DEADLINE hours,
#        PREV=/kaggle/input/<earlier output> to skip runs that an earlier job already finished,
#        CAPTURE=/kaggle/input/<photos> to pose the self-captured scene with COLMAP first)
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

media() {  # novel-view videos (R-videos) and qualitative grids (R5) from whatever seed-0 exports exist
  for s in lego chair drums real; do
    arms=""; for m in A D E B C; do [ -f "outputs/${s}_${m}_n8_s0/export/test_rgb.npy" ] && arms="$arms $m"; done
    [ -z "$arms" ] && continue
    vid=$(for m in $arms; do case $m in A|D|E) printf "%s " "$m";; esac; done)
    [ -n "$vid" ] && python scripts/make_videos.py --scene "$s" --seed 0 --arms $vid || true
    python scripts/qualitative_grid.py --scene "$s" --view 3 --arms $arms --out "results/figures/R5_qualitative_$s.png" || true
  done
}

real_setup() {  # CAPTURE=/kaggle/input/<photos>: COLMAP poses -> Blender layout + manifests for the 'real' scene
  [ -n "${CAPTURE:-}" ] || return 0
  [ -f data/blender400/real/transforms_train.json ] && return 0
  python -c "import pycolmap" 2>/dev/null || pip install -q pycolmap || { apt-get update -qq && apt-get install -y -qq colmap; }
  python scripts/colmap_scene.py --src "$CAPTURE" --out data/real/capture 2>&1 | tail -5
  python scripts/real_scene.py --ns-dir data/real/capture --out-root data/blender400 --manifests manifests
  mkdir -p results && cp data/real/capture/colmap_report.json results/real_colmap_report.json
}

import_prev() {  # PREV=/kaggle/input/<earlier version output>: reuse finished runs so they are skipped, not retrained
  [ -n "${PREV:-}" ] || return 0
  src=$(find "$PREV" -maxdepth 4 -type d -name runs -path "*results/runs" | head -1)
  [ -n "$src" ] || { echo "PREV set but no results/runs under $PREV"; return 0; }
  mkdir -p results/runs results/logs
  cp -n "$src"/*.json results/runs/ 2>/dev/null || true
  cp -n "$(dirname "$src")"/logs/*.log results/logs/ 2>/dev/null || true
  cp -n "$(dirname "$src")"/iters.json results/ 2>/dev/null || true
  echo "imported $(ls results/runs | wc -l) finished runs from $src"
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
  runs)  setup
         real_setup
         import_prev
         NG=$(python -c "import torch; print(max(torch.cuda.device_count(), 1))")
         ITERS=${2:-30000}
         if [ "$ITERS" = auto ]; then        # budget rule: measure s/iter here, pick I so the plan fits BUDGET_GPU_H
           NRUNS=$(python scripts/run_all.py --plan submission --only "${3:?experiment ids}" ${4:+--scene "$4"} --list | grep -c " cost ")
           eval "$(CUDA_VISIBLE_DEVICES=0 python scripts/choose_iters.py --budget-gpu-h "${BUDGET_GPU_H:-40}" --runs "$NRUNS" \
                   --rays "${RAYS:-4096}" ${AMP:+--amp} | tee "$OUT/choose_iters.log" | grep -E '^(ITERS|AMP)=')"
         fi
         echo "running on $NG GPU(s) in parallel, ITERS=$ITERS AMP=${AMP:-0} RAYS=${RAYS:-4096}"
         for k in $(seq 0 $((NG-1))); do
           CUDA_VISIBLE_DEVICES=$k python scripts/run_all.py --plan submission --iters "$ITERS" --only "${3:?experiment ids}" \
             ${4:+--scene "$4"} --device cuda --rays "${RAYS:-4096}" ${AMP:+--amp} ${DEADLINE:+--deadline-hours "$DEADLINE"} --shard "$k/$NG" > "$OUT/runs_gpu$k.log" 2>&1 &
         done
         wait
         tail -n 15 "$OUT"/runs_gpu*.log
         python -m evalkit.aggregate --results results --manifests manifests 2>&1 | tail -12 || true
         media
         python scripts/make_report_assets.py --results results --out results/report_assets || true
         save ;;
  *) echo "unknown stage $stage"; exit 2 ;;
esac
