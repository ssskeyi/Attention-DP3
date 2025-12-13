#!/usr/bin/env bash
set -euo pipefail

# 参数：
#   $1 frames_root (默认 ../3D-Diffusion-Policy/export/adroit_door_frames)
#   $2 output_root (默认 ../3D-Diffusion-Policy/export_gs2/adroit_door)
#   $3 text prompt (默认 "door handle. door.")
#   $4 device (默认 cuda)

ROOT_DIR="$(cd "$(dirname "$0")/.."; pwd)"
GS2_DIR="${GS2_DIR:-${ROOT_DIR}/Grounded-SAM-2}"

# 处理相对路径，转换为绝对路径
if [[ "${1:-}" =~ ^/ ]]; then
  FRAMES_ROOT="${1:-${ROOT_DIR}/3D-Diffusion-Policy/export/adroit_door_frames}"
else
  FRAMES_ROOT="${ROOT_DIR}/${1:-3D-Diffusion-Policy/export/adroit_door_frames}"
fi

if [[ "${2:-}" =~ ^/ ]]; then
  OUTPUT_ROOT="${2:-${ROOT_DIR}/3D-Diffusion-Policy/export_gs2/adroit_door}"
else
  OUTPUT_ROOT="${ROOT_DIR}/${2:-3D-Diffusion-Policy/export_gs2/adroit_door}"
fi

TEXT_PROMPT="${3:-door handle. door.}"
DEVICE="${4:-cuda}"

echo "[gs2.sh] ROOT_DIR: ${ROOT_DIR}"
echo "[gs2.sh] GS2_DIR: ${GS2_DIR}"
echo "[gs2.sh] FRAMES_ROOT: ${FRAMES_ROOT}"
echo "[gs2.sh] OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "[gs2.sh] TEXT_PROMPT: ${TEXT_PROMPT}"
echo "[gs2.sh] DEVICE: ${DEVICE}"

if [ ! -d "${FRAMES_ROOT}" ]; then
  echo "[ERROR] FRAMES_ROOT does not exist: ${FRAMES_ROOT}"
  exit 1
fi

if [ ! -d "${GS2_DIR}" ]; then
  echo "[ERROR] GS2_DIR does not exist: ${GS2_DIR}"
  exit 1
fi

cd "${GS2_DIR}"
echo "[gs2.sh] Changed to: $(pwd)"
echo "[gs2.sh] Running batch_grounded_sam2.py..."

python batch_grounded_sam2.py \
  --frames_root "${FRAMES_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --text "${TEXT_PROMPT}" \
  --sam2_ckpt checkpoints/sam2.1_hiera_large.pt \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --gdino_cfg grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --gdino_ckpt gdino_checkpoints/groundingdino_swint_ogc.pth \
  --device "${DEVICE}"

echo "[gs2.sh] Finished!"