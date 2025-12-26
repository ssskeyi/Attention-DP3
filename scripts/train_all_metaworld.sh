#!/usr/bin/env bash
set -euo pipefail

# 一次性串行跑 Metaworld 指定实验（attn / no_attn 对比）。
# 需在已激活的 aedp3 环境下运行。
# 环境变量可选：
#   GPU_ID=0
#   SEED=0
#   CONFIG_NAME=dp3
#   EXTRA_ARGS=""

DEBUG=False
save_ckpt=True
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-dp3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TASKS=(
  metaworld_pick-place_no_attn
  metaworld_sweep_no_attn
  metaworld_shelf-place_no_attn
  metaworld_soccer_no_attn
  metaworld_stick-pull_no_attn
  metaworld_pick-place
  metaworld_sweep
  metaworld_shelf-place
  metaworld_soccer
  metaworld_stick-pull
)

log() { echo -e "[run_all_metaworld] $*"; }

if [ $DEBUG = True ]; then
    wandb_mode=offline
else
    wandb_mode=online
fi

cd "${ROOT}/3D-Diffusion-Policy"

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${GPU_ID}

total_start=$(date +%s)

for task in "${TASKS[@]}"; do
  task_start=$(date +%s)
  if [[ "${task}" == *_no_attn ]]; then
    addition_info="1221mw"
  else
    addition_info="1221mwaedp3"
  fi
  exp_name="${task}-${CONFIG_NAME}-${addition_info}"
  run_dir="data/outputs/${exp_name}_seed${SEED}"
  if [[ -n "${RUN_NAME_PREFIX}" ]]; then
    run_name="${RUN_NAME_PREFIX}_${exp_name}"
  else
    run_name="${exp_name}"
  fi
  log "开始训练: ${task} (exp_name=${exp_name}, gpu_id=${GPU_ID}, seed=${SEED})"
  python train.py --config-name=${CONFIG_NAME}.yaml \
                            task=${task} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${SEED} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            logging.name=${run_name} \
                            logging.project=aedp3_wetaworld_cmp \
                            checkpoint.save_ckpt=${save_ckpt} \
                            ${EXTRA_ARGS}
  task_end=$(date +%s)
  log "完成训练: ${task} 用时 $((task_end - task_start)) 秒"
done

total_end=$(date +%s)
log "全部任务完成，总耗时 $((total_end - total_start)) 秒"


