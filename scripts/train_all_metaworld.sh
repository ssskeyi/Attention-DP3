#!/usr/bin/env bash
set -euo pipefail

# 一次性串行跑 Metaworld 指定实验（attn / no_attn 对比）。
# 需在已激活的 aedp3 环境下运行。
# 环境变量可选：
#   GPU_ID=0
#   SEED=0
#   CONFIG_NAME=dp3
#   TASKS="task1 task2 ..."  # 指定要运行的任务，不设置则运行全部
#   GS2_PORT=5000           # GS2服务端口（仅GS2任务需要，会设置GS2_API_URL环境变量）
#   EXTRA_ARGS=""

DEBUG=False
save_ckpt=True
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-dp3}"
GS2_PORT="${GS2_PORT:-5000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# 默认任务列表
DEFAULT_TASKS=(
  # 之前运行过的任务，已注释掉
  # metaworld_pick-place_no_attn
  # metaworld_sweep_no_attn
  # metaworld_shelf-place_no_attn
  # metaworld_soccer_no_attn
  # metaworld_stick-pull_no_attn
  # metaworld_box-close_no_attn
  # metaworld_bin-picking_no_attn
  # metaworld_disassemble_no_attn
  # metaworld_reach_no_attn
  # metaworld_pick-place
  # metaworld_sweep
  # metaworld_shelf-place
  # metaworld_soccer
  # metaworld_stick-pull
  # metaworld_box-close
  # metaworld_bin-picking
  # metaworld_disassemble
  # metaworld_reach

  # 新增任务
  metaworld_pick-place-wall_no_attn
  metaworld_push_no_attn
  metaworld_pick-out-of-hole_no_attn
  metaworld_hand-insert_no_attn
  metaworld_assembly_no_attn
  metaworld_push-wall_no_attn
  metaworld_peg-insert-side_no_attn
  metaworld_pick-place-wall
  metaworld_push
  metaworld_pick-out-of-hole
  metaworld_hand-insert
  metaworld_assembly
  metaworld_push-wall
  metaworld_peg-insert-side
)

# 如果设置了TASKS环境变量，使用它；否则使用默认任务
if [[ -n "${TASKS:-}" ]]; then
  # 将TASKS字符串转换为数组
  IFS=' ' read -r -a TASKS_ARRAY <<< "$TASKS"
else
  TASKS_ARRAY=("${DEFAULT_TASKS[@]}")
fi

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

for task in "${TASKS_ARRAY[@]}"; do
  task_start=$(date +%s)
  if [[ "${task}" == *_no_attn ]]; then
    addition_info="0112mw"
    # DP3任务不需要GS2 API URL
    task_extra_args="${EXTRA_ARGS}"
  else
    addition_info="0112mwaedp3"
    # GS2任务需要设置GS2 API URL环境变量
    export GS2_API_URL="http://127.0.0.1:${GS2_PORT}"
    task_extra_args="${EXTRA_ARGS}"
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
                            logging.project=aedp3_wetaworld_cmp_0112 \
                            checkpoint.save_ckpt=${save_ckpt} \
                            ${task_extra_args}
  task_end=$(date +%s)
  log "完成训练: ${task} 用时 $((task_end - task_start)) 秒"
done

total_end=$(date +%s)
log "全部任务完成，总耗时 $((total_end - total_start)) 秒"


