#!/usr/bin/env bash
set -euo pipefail

# 一键生成 Adroit (pen/hammer/door) 的无 attn + 有 attn 数据。
# 依赖：
#   - third_party/VRL3/src/gen_demonstration_expert.py （生成专家演示）
#   - scripts/export_adroit_frames.py （导出帧）
#   - Grounded-SAM-2 及其权重、配置（gs2.sh）
#   - scripts/convert_zarr_with_attn3d.py （生成 attn_3d zarr）
# 环境变量：
#   GPU (默认 0)              : 用于生成演示
#   DEVICE (默认 cuda)        : gs2 推理设备
#   ROOT (默认当前仓库根)      : 仓库根目录
#   GS2_DIR (可选)           : Grounded-SAM-2 目录，默认 $ROOT/Grounded-SAM-2
#   MAX_EP (默认 10)          : 生成演示/处理的 episode 数（帧导出 & attn 也会用）
#   N_POINTS (默认 512)       : attn_3d 采样点数
#   TASKS (默认 "door hammer pen")
#   GS2_CONDA_ENV (默认 aedp3_vis): 指定运行 gs2.sh 时的 conda 环境

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
MAX_EP="${MAX_EP:-10}"
N_POINTS="${N_POINTS:-512}"
TASKS="${TASKS:-door hammer pen}"
GS2_DIR="${GS2_DIR:-${ROOT}/Grounded-SAM-2}"
# 修改处：设置默认 conda 环境为 aedp3_vis
GS2_CONDA_ENV="${GS2_CONDA_ENV:-aedp3_vis}"

log() { echo -e "[make_adroit] $*"; }

gen_demo() {
  local task="$1"
  log "生成演示: ${task}"
  pushd "${ROOT}/third_party/VRL3/src" >/dev/null
  CUDA_VISIBLE_DEVICES="${GPU}" python gen_demonstration_expert.py --env_name "${task}" \
    --num_episodes "${MAX_EP}" \
    --root_dir "../../../3D-Diffusion-Policy/data/" \
    --expert_ckpt_path "../ckpts/vrl3_${task}.pt" \
    --img_size 84 \
    --not_use_multi_view \
    --use_point_crop
  popd >/dev/null
}

export_frames() {
  local task="$1"
  local zarr="${ROOT}/3D-Diffusion-Policy/data/adroit_${task}_expert.zarr"
  local out_dir="${ROOT}/3D-Diffusion-Policy/export/adroit_${task}_frames"
  log "导出帧: ${task} -> ${out_dir}"
  python "${ROOT}/scripts/export_adroit_frames.py" \
    --zarr "${zarr}" \
    --out_dir "${out_dir}" \
    --max_episodes "${MAX_EP}"
}

gs2_for_task() {
  local task="$1"
  local frames_root="${ROOT}/3D-Diffusion-Policy/export/adroit_${task}_frames"
  local output_root="${ROOT}/3D-Diffusion-Policy/export_gs2/adroit_${task}"
  local text_prompt
  case "${task}" in
    door)   text_prompt="door handle. door." ;;
    hammer) text_prompt="hammer. nail" ;;
    pen)    text_prompt="blue pen" ;;
    *)      text_prompt="${task}" ;;
  esac
  log "运行 GS2: ${task} -> ${output_root}"
  local runner=()
  # 因为上面设置了默认值，只要不显式传空值，这里都会进入 conda run 逻辑
  if [[ -n "${GS2_CONDA_ENV:-}" ]]; then
    runner=(conda run -n "${GS2_CONDA_ENV}")
  fi
  GS2_DIR="${GS2_DIR}" "${runner[@]}" bash "${ROOT}/scripts/gs2.sh" \
    "${frames_root}" \
    "${output_root}" \
    "${text_prompt}" \
    "${DEVICE}"
}

convert_attn_zarr() {
  local task="$1"
  local input_zarr="${ROOT}/3D-Diffusion-Policy/data/adroit_${task}_expert.zarr"
  local json_root="${ROOT}/3D-Diffusion-Policy/export_gs2/adroit_${task}"
  local output_zarr="${ROOT}/3D-Diffusion-Policy/data/adroit_${task}_expert_attn3d.zarr"
  log "生成 attn_3d zarr: ${task} -> ${output_zarr}"
  bash "${ROOT}/scripts/convert_zarr_with_attn3d.sh" \
    "${input_zarr}" \
    "${json_root}" \
    "${output_zarr}" \
    "${MAX_EP}" \
    "${N_POINTS}"
}

main() {
  log "ROOT=${ROOT}"
  log "TASKS=${TASKS}"
  log "GPU=${GPU}, DEVICE=${DEVICE}, MAX_EP=${MAX_EP}, N_POINTS=${N_POINTS}"
  log "GS2_DIR=${GS2_DIR}"
  log "GS2_CONDA_ENV=${GS2_CONDA_ENV}"
  for task in ${TASKS}; do
    gen_demo "${task}"
    export_frames "${task}"
    gs2_for_task "${task}"
    convert_attn_zarr "${task}"
  done
  log "全部完成"
}

main "$@"