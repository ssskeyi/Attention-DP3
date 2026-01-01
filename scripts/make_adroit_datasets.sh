#!/usr/bin/env bash
set -euo pipefail

# 一键生成 Adroit (pen/hammer/door) 的无 attn + 有 attn 数据，支持GS2和环境分割。
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
#   SEG_TYPES (默认 "gs2 env"): 分割类型，可选 "gs2" "env" 或两者

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
MAX_EP="${MAX_EP:-10}"
N_POINTS="${N_POINTS:-512}"
TASKS="${TASKS:-door hammer pen}"
GS2_DIR="${GS2_DIR:-${ROOT}/Grounded-SAM-2}"
# 修改处：设置默认 conda 环境为 aedp3_vis
GS2_CONDA_ENV="${GS2_CONDA_ENV:-aedp3_vis}"
SEG_TYPES="${SEG_TYPES:-env gs2}"
DATA_OUTPUT_ROOT="${DATA_OUTPUT_ROOT:-}"
DATA_ROOT="${DATA_OUTPUT_ROOT:-${ROOT}/3D-Diffusion_policy/data}"

log() { echo -e "[make_adroit] $*"; }

gen_demo() {
  local task="$1"
  local seg_type="$2"
  local output_dir="${DATA_ROOT}/"
  local save_name="adroit_${task}_expert_${seg_type}.zarr"

  log "生成演示: ${task} (seg_type=${seg_type}) -> ${save_name}"

  pushd "${ROOT}/third_party/VRL3/src" >/dev/null
  local use_env_seg_flag=""
  if [ "${seg_type}" = "env" ]; then
    use_env_seg_flag="--use_env_seg"
  fi

  # 修改保存路径
  CUDA_VISIBLE_DEVICES="${GPU}" python gen_demonstration_expert.py --env_name "${task}" \
    --num_episodes "${MAX_EP}" \
    --root_dir "${output_dir}" \
    --save_name "${save_name}" \
    --expert_ckpt_path "../ckpts/vrl3_${task}.pt" \
    --img_size 84 \
    --not_use_multi_view \
    --use_point_crop \
    ${use_env_seg_flag}
  popd >/dev/null
}

export_frames() {
  local task="$1"
  local seg_type="$2"
  local zarr="${DATA_ROOT}/adroit_${task}_expert_${seg_type}.zarr"
  local out_dir="${ROOT}/3D-Diffusion-Policy/export/adroit_${task}_${seg_type}_frames"
  log "导出帧: ${task} (${seg_type}) -> ${out_dir}"
  python "${ROOT}/scripts/export_adroit_frames.py" \
    --zarr "${zarr}" \
    --out_dir "${out_dir}" \
    --max_episodes "${MAX_EP}"
}

gs2_for_task() {
  local task="$1"
  local seg_type="$2"
  # 只有GS2分割类型才需要运行GS2
  if [ "${seg_type}" != "gs2" ]; then
    return
  fi

  local frames_root="${ROOT}/3D-Diffusion-Policy/export/adroit_${task}_${seg_type}_frames"
  local output_root="${ROOT}/3D-Diffusion-Policy/export_gs2/adroit_${task}_${seg_type}"
  local text_prompt
  case "${task}" in
    door)   text_prompt="door handle. door." ;;
    hammer) text_prompt="hammer. nail." ;;
    pen)    text_prompt="blue pen in hand." ;;
    *)      text_prompt="${task}" ;;
  esac
  log "运行 GS2: ${task} (${seg_type}) -> ${output_root}"
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
  local seg_type="$2"
  local input_zarr="${DATA_ROOT}/adroit_${task}_expert_${seg_type}.zarr"
  local json_root="${ROOT}/3D-Diffusion-Policy/export_gs2/adroit_${task}_${seg_type}"
  local output_zarr="${DATA_ROOT}/adroit_${task}_expert_${seg_type}_attn3d.zarr"
  log "生成 attn_3d zarr: ${task} (${seg_type}) -> ${output_zarr}"
  local use_env_seg_flag=""
  if [ "${seg_type}" = "env" ]; then
    use_env_seg_flag="--use_env_seg"
  fi
  python "${ROOT}/scripts/convert_zarr_with_attn3d.py" \
    --input_zarr "${input_zarr}" \
    --json_root "${json_root}" \
    --output_zarr "${output_zarr}" \
    --n_points "${N_POINTS}" \
    --max_episodes "${MAX_EP}" \
    ${use_env_seg_flag}
}

main() {
  log "ROOT=${ROOT}"
  log "TASKS=${TASKS}"
  log "SEG_TYPES=${SEG_TYPES}"
  log "GPU=${GPU}, DEVICE=${DEVICE}, MAX_EP=${MAX_EP}, N_POINTS=${N_POINTS}"
  log "GS2_DIR=${GS2_DIR}"
  log "GS2_CONDA_ENV=${GS2_CONDA_ENV}"

  for seg_type in ${SEG_TYPES}; do
    log "开始处理分割类型: ${seg_type}"
    for task in ${TASKS}; do
      gen_demo "${task}" "${seg_type}"
      if [ "${seg_type}" = "env" ]; then
        # 使用环境分割，直接转换zarr
        convert_attn_zarr "${task}" "${seg_type}"
      else
        # 使用GS2流程
        export_frames "${task}" "${seg_type}"
        gs2_for_task "${task}" "${seg_type}"
        convert_attn_zarr "${task}" "${seg_type}"
      fi
    done
    log "完成分割类型: ${seg_type}"
  done
  log "全部完成"
}

main "$@"