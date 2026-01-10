#!/usr/bin/env bash
set -euo pipefail

# 一键生成 DexArt (bucket/faucet/laptop/toilet) 的 GS2 attn 数据。
# 依赖：
#   - scripts/gen_demonstration_dexart.sh （生成专家演示）
#   - scripts/export_adroit_frames.py （导出帧，复用Adroit的脚本）
#   - Grounded-SAM-2 及其权重、配置（gs2.sh）
#   - scripts/convert_zarr_with_attn3d.sh （生成 attn_3d zarr）
# 环境变量：
#   DEVICE (默认 cuda)        : gs2 推理设备
#   ROOT (默认当前仓库根)
#   MAX_EP (默认 10)
#   N_POINTS (默认 1024)      : DexArt使用1024点云
#   TASKS (默认 "bucket faucet laptop toilet")
#   GS2_CONDA_ENV (默认 aedp3_vis)

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
DEVICE="${DEVICE:-cuda}"
MAX_EP="${MAX_EP:-10}"
N_POINTS="${N_POINTS:-1024}"
TASKS="${TASKS:-bucket faucet laptop toilet}"
GS2_DIR="${GS2_DIR:-${ROOT}/Grounded-SAM-2}"
GS2_CONDA_ENV="${GS2_CONDA_ENV:-aedp3_vis}"

log() { echo -e "[make_dexart] $*"; }

gen_demo() {
  local task="$1"
  log "生成演示: ${task}"
  pushd "${ROOT}" >/dev/null
  bash "${ROOT}/scripts/gen_demonstration_dexart.sh" "${task}"
  popd >/dev/null
}

export_frames() {
  local task="$1"
  local zarr="${ROOT}/3D-Diffusion-Policy/data/dexart_${task}_expert.zarr"
  local out_dir="${ROOT}/3D-Diffusion-Policy/export/dexart_${task}_frames"
  log "导出帧: ${task} -> ${out_dir}"
  python "${ROOT}/scripts/export_adroit_frames.py" \
    --zarr "${zarr}" \
    --out_dir "${out_dir}" \
    --max_episodes "${MAX_EP}"
}

gs2_for_task() {
  local task="$1"
  local frames_root="${ROOT}/3D-Diffusion-Policy/export/dexart_${task}_frames"
  local output_root="${ROOT}/3D-Diffusion-Policy/export_gs2/dexart_${task}"

  # Resolve task -> descriptive text prompt (do not simply use the task name)
  task_to_prompt() {
    local t="$1"
    case "${t}" in
      bucket) echo "bucket. water bucket." ;;
      faucet) echo "faucet. water faucet." ;;
      laptop) echo "laptop. laptop computer." ;;
      toilet) echo "toilet. toilet bowl." ;;
      *) echo "${t}" ;; # fallback: pass through
    esac
  }
  local text_prompt
  text_prompt="$(task_to_prompt "${task}")"
  log "运行 GS2: ${task} -> ${output_root}"
  local runner=()
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
  local input_zarr="${ROOT}/3D-Diffusion-Policy/data/dexart_${task}_expert.zarr"
  local json_root="${ROOT}/3D-Diffusion-Policy/export_gs2/dexart_${task}"
  local output_zarr="${ROOT}/3D-Diffusion-Policy/data/dexart_${task}_expert_attn3d.zarr"
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
  log "MAX_EP=${MAX_EP}, N_POINTS=${N_POINTS}"
  for task in ${TASKS}; do
    gen_demo "${task}"
    export_frames "${task}"
    gs2_for_task "${task}"
    convert_attn_zarr "${task}"
  done
  log "全部完成"
}

main "$@"
