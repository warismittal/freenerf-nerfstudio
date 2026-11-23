#!/usr/bin/env bash
# First thing to run on a GPU machine (or a Mac CPU, slowly): proves the nerfstudio integration end to end in minutes.
# Trains 30 iterations of every feature combination on tiny batches, then exports one low-resolution test view.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== 1. imports and method registration"
python -c "import freenerf_ns.config, freenerf_ns.model; print('imports ok')"
ns-train --help | grep -q freenerf && echo "freenerf is registered with nerfstudio"
echo "== 2. integration tests (layout, mask, losses, model forward/backward)"
python -m pytest tests/test_nerfstudio_integration.py -x -q
echo "== 3. training smoke runs"
python scripts/make_sparse_datasets.py --scene lego --n 8
i=0
while IFS= read -r FLAGS; do
  i=$((i+1))
  echo "-- run $i: ${FLAGS:-<vanilla>}"
  # shellcheck disable=SC2086
  ns-train freenerf --data data/sparse/lego_n8 --max-num-iterations 30 --optimizers.fields.scheduler.max-steps 30 \
      --pipeline.datamanager.train-num-rays-per-batch 512 --output-dir outputs/smoke --experiment-name "smoke$i" \
      --timestamp main --vis tensorboard --steps-per-save 30 $FLAGS blender-data > "outputs_smoke_$i.log" 2>&1 \
      || { echo "FAILED run $i - see outputs_smoke_$i.log"; tail -25 "outputs_smoke_$i.log"; exit 1; }
done <<'FLAGS_LIST'

--pipeline.model.freq-mask ramp --pipeline.model.freq-end-frac 0.5
--pipeline.model.freq-mask step --pipeline.model.freq-end-frac 0.5
--pipeline.model.freq-mask reverse --pipeline.model.freq-end-frac 0.5
--pipeline.model.occ-weight 0.01
--pipeline.model.ds-weight 0.1 --pipeline.model.ds-num-patches 4 --pipeline.model.ds-patch-size 4
--pipeline.model.freq-mask ramp --pipeline.model.freq-end-frac 0.5 --pipeline.model.occ-weight 0.01
FLAGS_LIST
echo "== 4. export one low-resolution test view from the last run"
python -m freenerf_ns.export --config outputs/smoke/smoke$i/freenerf/main/config.yml --scene lego \
    --manifest manifests/lego_n8.json --data-root data/blender400 --out outputs/smoke/export \
    --width 100 --height 100 --limit 1 --no-train
python - <<'PY'
import numpy as np
r = np.load("outputs/smoke/export/test_rgb.npy"); d = np.load("outputs/smoke/export/test_depth.npy")
assert r.shape == (1, 100, 100, 3) and d.shape == (1, 100, 100) and np.isfinite(r).all() and np.isfinite(d).all()
print("export ok", r.shape, "depth range", float(d.min()), float(d.max()))
PY
echo "SMOKE TEST PASSED"
