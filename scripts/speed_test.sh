#!/usr/bin/env bash
# Measure seconds per training iteration of the FreeNeRF arm (mask + occlusion) on 8-view Lego, fp32 vs mixed precision,
# one configuration per GPU in parallel. Prints the last logged iteration time and loss of each.
#   bash scripts/speed_test.sh [ITERS] [RAYS]
set -uo pipefail
cd "$(dirname "$0")/.."
IT=${1:-400}; RAYS=${2:-4096}
python scripts/make_sparse_datasets.py --scene lego --n 8 >/dev/null
run() {  # $1 gpu, $2 tag, rest: extra flags
  local g=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$g ns-train freenerf --data data/sparse/lego_n8 --max-num-iterations "$IT" \
    --optimizers.fields.scheduler.max-steps "$IT" --pipeline.datamanager.train-num-rays-per-batch "$RAYS" \
    --pipeline.model.freq-mask ramp --pipeline.model.occ-weight 0.01 --output-dir outputs/speed --experiment-name "$tag" \
    --timestamp main --vis tensorboard --steps-per-save 100000 "$@" blender-data > "outputs_speed_$tag.log" 2>&1
}
t0=$(date +%s)
run 0 fp32 & run 1 amp --mixed-precision True & wait
echo "wall seconds for both: $(( $(date +%s) - t0 ))"
for tag in fp32 amp; do
  echo "== $tag"
  grep -E "^[[:space:]]*[0-9]+ \(" "outputs_speed_$tag.log" | tail -3
  grep -i -E "nan|error|traceback" "outputs_speed_$tag.log" | head -5
done
