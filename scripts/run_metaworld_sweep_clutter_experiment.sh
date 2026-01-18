#!/usr/bin/env bash
set -euo pipefail

# Metaworld Sweep clutter experiment: Compare AEDP3 and DP3 generalization ability
# with 8 distraction objects in sweep environment
# Experiment design:
# - AEDP3: with attention mechanism (attn mode)
# - DP3: without attention mechanism (no_attn mode)
# - Environment: 8 distraction blocks around target area
#
# Environment variables:
#   GPU_ID=0                Training GPU
#   SEED=0                  Training seed
#   CONFIG_NAME=dp3         Hydra config name

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-dp3}"

log() { echo -e "[sweep_clutter] $*"; }

# Run AEDP3 clutter experiment
run_aedp3_clutter() {
    log "=== Running AEDP3 clutter experiment ==="
    task_name="metaworld_sweep_clutter"
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
        logging.project=aedp3_metaworld_clutter_comparison_0118 \
        checkpoint.save_ckpt=true
}

# Run DP3 clutter experiment
run_dp3_clutter() {
    log "=== Running DP3 clutter experiment ==="
    task_name="metaworld_sweep_no_attn_clutter"
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
        logging.project=aedp3_metaworld_clutter_comparison_0118 \
        checkpoint.save_ckpt=true
}

# Main function
main() {
    log "Metaworld Sweep clutter experiment started"
    log "Configuration:"
    log "  ROOT=${ROOT}"
    log "  GPU_ID=${GPU_ID}"
    log "  SEED=${SEED}"
    log "  CONFIG_NAME=${CONFIG_NAME}"

    start_time=$(date +%s)

    # Run both experiments
    run_dp3_clutter
    run_aedp3_clutter

    end_time=$(date +%s)
    log "Experiment completed! Total time: $((end_time - start_time)) seconds"

    log "Experiment results:"
    log "  - AEDP3 clutter: data/outputs/metaworld_sweep_clutter-${CONFIG_NAME}-AEDP3_clutter_seed${SEED}"
    log "  - DP3 clutter: data/outputs/metaworld_sweep_no_attn_clutter-${CONFIG_NAME}-DP3_clutter_seed${SEED}"
    log "L5 metrics will be automatically calculated and recorded to wandb project: aedp3_metaworld_clutter_comparison_0118"
}

main "$@"
