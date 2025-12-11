#!/usr/bin/env bash
set -euo pipefail

# 参数：
#   $1 frames_root (默认 ../3D-Diffusion-Policy/export/adroit_door_frames)
#   $2 output_root (默认 ../3D-Diffusion-Policy/export_gs2/adroit_door)
#   $3 text prompt (默认 "door handle. door.")
#   $4 device (默认 cuda)

ROOT_DIR="$(cd "$(dirname "$0")/.."; pwd)"
GS2_DIR="${GS2_DIR:-${ROOT_DIR}/Grounded-SAM-2}"

FRAMES_ROOT="${1:-${ROOT_DIR}/3D-Diffusion-Policy/export/adroit_door_frames}"
OUTPUT_ROOT="${2:-${ROOT_DIR}/3D-Diffusion-Policy/export_gs2/adroit_door}"
TEXT_PROMPT="${3:-door handle. door.}"
DEVICE="${4:-cuda}"

cd "${GS2_DIR}"

python batch_grounded_sam2.py \
  --frames_root "${FRAMES_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --text "${TEXT_PROMPT}" \
  --sam2_ckpt checkpoints/sam2.1_hiera_large.pt \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --gdino_cfg grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --gdino_ckpt gdino_checkpoints/groundingdino_swint_ogc.pth \
  --device "${DEVICE}"