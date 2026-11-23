#!/usr/bin/env bash
# Download NeRF-synthetic, downscale to 400x400 (DietNeRF protocol) and build the hashed manifests.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
SRC=$(find data -maxdepth 3 -type d -name lego 2>/dev/null | head -1 | xargs -r dirname || true)
if [ -z "${SRC}" ] && [ -d /kaggle/input ]; then          # dataset attached as a Kaggle input
  TJ=$(find /kaggle/input -maxdepth 6 -path "*lego/transforms_train.json" 2>/dev/null | head -1 || true)
  [ -n "${TJ}" ] && SRC=$(dirname "$(dirname "${TJ}")") && echo ">> using Kaggle input ${SRC}"
fi
if [ -z "${SRC}" ]; then
  echo ">> downloading NeRF-synthetic with nerfstudio"
  ns-download-data blender --save-dir data || true
  SRC=$(find data -maxdepth 3 -type d -name lego 2>/dev/null | head -1 | xargs -r dirname || true)
fi
if [ -z "${SRC}" ]; then
  cat <<'MSG'
!! Automatic download failed (the dataset lives on Google Drive and is sometimes quota-limited).
   Manual route: download nerf_synthetic.zip from the NeRF project page (link in github.com/bmild/nerf README),
   unzip so that  data/blender/lego/transforms_train.json  exists, then re-run this script.
   On Kaggle: add the dataset as an input and symlink it:  ln -s /kaggle/input/<dataset>/nerf_synthetic data/blender
MSG
  exit 1
fi
echo ">> source: ${SRC}"
SCENES=${SCENES:-"lego chair drums ficus"}   # laptop:  SCENES=lego SIZE=200 CLEAN=1 bash scripts/get_data.sh
SIZE=${SIZE:-400}
# shellcheck disable=SC2086
python scripts/downscale_blender.py --src "${SRC}" --dst "data/blender${SIZE}" --scenes ${SCENES} --size "${SIZE}"
# shellcheck disable=SC2086
python scripts/make_manifests.py --data-root "data/blender${SIZE}" --manifests manifests --scenes ${SCENES}
if [ "${CLEAN:-0}" = "1" ]; then
  echo ">> CLEAN=1: deleting the full-size originals and any downloaded archive (only data/blender${SIZE} is needed)"
  rm -rf "${SRC}"; find data -maxdepth 2 -name "*.zip" -delete
fi
du -sh data 2>/dev/null || true
echo ">> done. Next: python scripts/run_all.py --only E1 --dry-run"
