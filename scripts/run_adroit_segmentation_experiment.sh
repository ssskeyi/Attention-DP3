#!/usr/bin/env bash
set -euo pipefail

# Adroit分割实验：比较GS2分割 vs 环境分割，每种都有attn和no_attn版本
# 实验设计：
# - gs2_no_attn: GS2分割 + 无attention
# - gs2_attn: GS2分割 + 有attention
# - env_no_attn: 环境分割 + 无attention
# - env_attn: 环境分割 + 有attention
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
TASKS="${TASKS:-door hammer pen}"
SEG_TYPES="${SEG_TYPES:-gs2 env}"
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

    bash "${ROOT}/scripts/make_adroit_datasets.sh"
    log "数据生成完成"
}

# 训练阶段
training_phase() {
    log "=== 开始训练阶段 ==="

    for seg_type in ${SEG_TYPES}; do
        log "处理分割类型: ${seg_type}"

        # no_attn 实验
        log "运行 ${seg_type} no_attn 实验"
        export DATASET_TYPE="${seg_type}"
        export RUN_NAME_PREFIX="${seg_type}_no_attn"
        export GPU_ID="${GPU_ID}"
        export SEED="${SEED}"
        export CONFIG_NAME="${CONFIG_NAME}"

        bash "${ROOT}/scripts/train_all_adroit.sh"

        # attn 实验
        log "运行 ${seg_type} attn 实验"
        export RUN_NAME_PREFIX="${seg_type}_attn"
        export GPU_ID="${GPU_ID}"
        export SEED="${SEED}"
        export CONFIG_NAME="${CONFIG_NAME}"

        bash "${ROOT}/scripts/train_all_adroit.sh"
    done
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
    for seg_type in ${SEG_TYPES}; do
        for task_base in ${TASKS}; do
            log "  - ${seg_type}_no_attn_${task_base}"
            log "  - ${seg_type}_attn_${task_base}"
        done
    done
}

main "$@"
