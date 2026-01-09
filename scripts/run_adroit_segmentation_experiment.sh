#!/usr/bin/env bash
set -euo pipefail

# Adroit分割实验：比较不同attention策略的效果
# 实验设计：
# - no_attn: 无attention（基准）
# - gs2_attn: GS2分割 + attention
# - env_attn: 环境分割 + attention
#
# 环境变量：
#   GPU_ID=0                训练使用的GPU ID
#   DATA_GPU=0              数据生成使用的GPU ID
#   SEED=0                  训练种子
#   CONFIG_NAME=dp3         Hydra配置名
#   TASKS="door hammer pen" 任务列表
#   SEG_TYPES="gs2 env"     分割类型
#   MAX_EP=10               每个任务的episode数
#   SKIP_DATA_GEN=false     是否跳过数据生成阶段

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
GPU_ID="${GPU_ID:-0}"
DATA_GPU="${DATA_GPU:-0}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-dp3}"
TASKS="${TASKS:-pen door hammer}"
SEG_TYPES="${SEG_TYPES:-env gs2}"
MAX_EP="${MAX_EP:-10}"
SKIP_DATA_GEN="${SKIP_DATA_GEN:-false}"

log() { echo -e "[experiment] $*"; }

# 数据生成阶段
data_generation() {
    log "=== 开始数据生成阶段 ==="
    export GPU="${DATA_GPU}"
    export MAX_EP="${MAX_EP}"
    export TASKS="${TASKS}"
    export SEG_TYPES="${SEG_TYPES}"

    if [ "${SKIP_DATA_GEN}" = "true" ]; then
        log "跳过数据生成阶段"
        return
    fi

    # 数据直接保存到data文件夹
    export DATA_OUTPUT_ROOT="${DATA_OUTPUT_ROOT:-${ROOT}/3D-Diffusion-Policy/data}"
    log "数据将保存到: ${DATA_OUTPUT_ROOT}"

    bash "${ROOT}/scripts/make_adroit_datasets.sh"

    log "数据生成完成"
}

# 训练阶段
training_phase() {
    log "=== 开始训练阶段 ==="

    # no_attn 实验（使用no_attn数据集）
    log "运行 no_attn 实验"
    export DATASET_TYPE="no_attn"
    export RUN_NAME_PREFIX="no_attn"
    export GPU_ID="${GPU_ID}"
    export SEED="${SEED}"
    export CONFIG_NAME="${CONFIG_NAME}"
    export ATTN_MODE="no_attn"
    export EXTRA_ARGS=""  # no_attn 不需要设置 seg_type

    bash "${ROOT}/scripts/train_all_adroit.sh"

    # gs2_attn 实验（使用gs2_attn数据集）
    log "运行 gs2_attn 实验"
    export DATASET_TYPE="gs2_attn"
    export RUN_NAME_PREFIX="gs2_attn"
    export GPU_ID="${GPU_ID}"
    export SEED="${SEED}"
    export CONFIG_NAME="${CONFIG_NAME}"
    export ATTN_MODE="attn"
    export EXTRA_ARGS="+task.env_runner.seg_type=gs2"

    bash "${ROOT}/scripts/train_all_adroit.sh"

    # env_attn 实验（使用env_attn数据集）
    log "运行 env_attn 实验"
    export DATASET_TYPE="env_attn"
    export RUN_NAME_PREFIX="env_attn"
    export GPU_ID="${GPU_ID}"
    export SEED="${SEED}"
    export CONFIG_NAME="${CONFIG_NAME}"
    export ATTN_MODE="attn"
    export EXTRA_ARGS="+task.env_runner.seg_type=env"

    bash "${ROOT}/scripts/train_all_adroit.sh"
}

# 主函数
main() {
    log "Adroit分割实验开始"
    log "配置:"
    log "  ROOT=${ROOT}"
    log "  GPU_ID=${GPU_ID}"
    log "  DATA_GPU=${DATA_GPU}"
    log "  SEED=${SEED}"
    log "  CONFIG_NAME=${CONFIG_NAME}"
    log "  TASKS=${TASKS}"
    log "  SEG_TYPES=${SEG_TYPES}"
    log "  MAX_EP=${MAX_EP}"
    log "  SKIP_DATA_GEN=${SKIP_DATA_GEN}"

    total_start=$(date +%s)

    data_generation
    training_phase

    total_end=$(date +%s)
    log "实验完成！总耗时 $((total_end - total_start)) 秒"

    log "实验结果总结:"
    log "生成的实验配置:"
    for task_base in ${TASKS}; do
        log "  - no_attn_${task_base}"
        log "  - gs2_attn_${task_base}"
        log "  - env_attn_${task_base}"
    done
}

main "$@"
