#!/usr/bin/env bash
set -euo pipefail

# Hammer clutter实验：比较AEDP3和DP3在有干扰钉子环境下的泛化能力
# 实验设计：
# - AEDP3: 使用attention机制（attn模式）
# - DP3: 不使用attention机制（no_attn模式）
# - 环境：3个干扰钉子（clutter模式）
#
# 环境变量：
#   GPU_ID=0                使用的GPU ID
#   SEED=0                  训练种子
#   CONFIG_NAME=dp3         Hydra配置名

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-dp3}"

log() { echo -e "[hammer_clutter] $*"; }

# 运行AEDP3 clutter实验
run_aedp3_clutter() {
    log "=== 运行 AEDP3 clutter实验 ==="
    task_name="adroit_hammer_clutter"
    addition_info="AEDP3_clutter"
    exp_name="${task_name}-${CONFIG_NAME}-${addition_info}"
    run_dir="data/outputs/${exp_name}_seed${SEED}"

    cd "${ROOT}/3D-Diffusion-Policy"
    export HYDRA_FULL_ERROR=1
    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    python train.py --config-name=${CONFIG_NAME}.yaml \
        task=${task_name} \
        hydra.run.dir=${run_dir} \
        training.debug=false \
        training.seed=${SEED} \
        training.device="cuda:0" \
        exp_name=${exp_name} \
        logging.mode=online \
        logging.project=aedp3_clutter_comparison_0114 \
        checkpoint.save_ckpt=true
}

# 运行DP3 clutter实验
run_dp3_clutter() {
    log "=== 运行 DP3 clutter实验 ==="
    task_name="adroit_hammer_no_attn_clutter"
    addition_info="DP3_clutter"
    exp_name="${task_name}-${CONFIG_NAME}-${addition_info}"
    run_dir="data/outputs/${exp_name}_seed${SEED}"

    cd "${ROOT}/3D-Diffusion-Policy"
    export HYDRA_FULL_ERROR=1
    export CUDA_VISIBLE_DEVICES=${GPU_ID}

    python train.py --config-name=${CONFIG_NAME}.yaml \
        task=${task_name} \
        hydra.run.dir=${run_dir} \
        training.debug=false \
        training.seed=${SEED} \
        training.device="cuda:0" \
        exp_name=${exp_name} \
        logging.mode=online \
        logging.project=aedp3_clutter_comparison_0114 \
        checkpoint.save_ckpt=true
}

# 主函数
main() {
    log "Hammer clutter实验开始"
    log "配置:"
    log "  ROOT=${ROOT}"
    log "  GPU_ID=${GPU_ID}"
    log "  SEED=${SEED}"
    log "  CONFIG_NAME=${CONFIG_NAME}"

    start_time=$(date +%s)

    # 运行两个实验
    run_dp3_clutter
    run_aedp3_clutter

    end_time=$(date +%s)
    log "实验完成！总耗时 $((end_time - start_time)) 秒"

    log "实验结果："
    log "  - AEDP3 clutter: data/outputs/adroit_hammer_clutter-${CONFIG_NAME}-AEDP3_clutter_seed${SEED}"
    log "  - DP3 clutter: data/outputs/adroit_hammer_no_attn_clutter-${CONFIG_NAME}-DP3_clutter_seed${SEED}"
    log "L5指标将在训练过程中自动计算并记录到wandb项目: aedp3_clutter_comparison_0114"
}

main "$@"
