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
DATA_ROOT="${DATA_OUTPUT_ROOT:-${ROOT}/3D-Diffusion-Policy/data}"

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
  local zarr_file="${3:-${DATA_ROOT}/adroit_${task}_expert_${seg_type}.zarr}"
  local out_dir="${ROOT}/3D-Diffusion-Policy/export/adroit_${task}_${seg_type}_frames"
  log "导出帧: ${task} (${seg_type}) -> ${out_dir}"
  python "${ROOT}/scripts/export_adroit_frames.py" \
    --zarr "${zarr_file}" \
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
  local input_zarr_file="${3:-${DATA_ROOT}/adroit_${task}_expert_${seg_type}.zarr}"
  local json_root="${ROOT}/3D-Diffusion-Policy/export_gs2/adroit_${task}_${seg_type}"
  local output_zarr="${DATA_ROOT}/adroit_${task}_expert_${seg_type}_attn3d.zarr"
  log "生成 attn_3d zarr: ${task} (${seg_type}) -> ${output_zarr}"
  local use_env_seg_flag=""
  if [ "${seg_type}" = "env" ]; then
    use_env_seg_flag="--use_env_seg"
  fi
  python "${ROOT}/scripts/convert_zarr_with_attn3d.py" \
    --input_zarr "${input_zarr_file}" \
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

  # 阶段1：为每个task生成基础数据集（使用env分割作为基础）
  log "=== 阶段1：生成基础数据集 ==="
  for task in ${TASKS}; do
    log "生成基础数据集: ${task}"
    gen_demo "${task}" "env"
  done

  # 阶段2：基于基础数据集生成不同attention版本
  log "=== 阶段2：生成attention版本 ==="
  for task in ${TASKS}; do
    log "处理任务: ${task}"
    base_zarr="${DATA_ROOT}/adroit_${task}_expert_env.zarr"

    # 生成no_attn版本（直接使用基础zarr，改名）
    no_attn_zarr="${DATA_ROOT}/adroit_${task}_expert_no_attn.zarr"
    if [ -d "$base_zarr" ] && [ ! -d "$no_attn_zarr" ]; then
      cp -r "$base_zarr" "$no_attn_zarr"
      log "创建no_attn版本: $no_attn_zarr"
    fi

    # 生成env_attn版本（直接基于基础zarr转换）
    convert_attn_zarr "${task}" "env" "$base_zarr"

    # 生成gs2_attn版本（基于基础zarr进行GS2处理）
    export_frames "${task}" "gs2" "$base_zarr"  # 从基础zarr导出帧到gs2目录
    gs2_for_task "${task}" "gs2"   # 运行GS2
    convert_attn_zarr "${task}" "gs2" "$base_zarr"
  done

  log "全部完成"
}

main "$@"