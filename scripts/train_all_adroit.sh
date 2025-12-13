#!/usr/bin/env bash
set -euo pipefail

# 一次性串行跑完 Adroit 六个实验（attn / no_attn 各三项）。
# 需在已激活的 aedp3 环境下运行。
# addition_info 自动设置：
#   - 带 attn 的任务（adroit_pen/hammer/door）: 1212aedp3
#   - 不带 attn 的任务（*_no_attn）: 1212dp3
# exp_name 格式：${task}-${CONFIG_NAME}-${addition_info}（与 train_policy.sh 完全一致）
# 环境变量可选：
#   GPU_ID=0                使用的 GPU ID（会设 CUDA_VISIBLE_DEVICES）
#   SEED=0                  训练种子
#   CONFIG_NAME=dp3         Hydra 配置名
#   EXTRA_ARGS=""           额外透传给 train.py（如 training.debug=true）

DEBUG=False
save_ckpt=True
# 运行名前缀，可用于区分 wandb run；默认用 exp_name
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-dp3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

TASKS=(
  adroit_pen_no_attn
  adroit_pen
  adroit_hammer_no_attn
  adroit_hammer
  adroit_door_no_attn
  adroit_door
)

log() { echo -e "[run_all_adroit] $*"; }

if [ $DEBUG = True ]; then
    wandb_mode=offline
else
    wandb_mode=online
fi

cd "${ROOT}/3D-Diffusion-Policy"

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${GPU_ID}

# 记录总耗时
total_start=$(date +%s)

for task in "${TASKS[@]}"; do
  task_start=$(date +%s)
  # 根据任务名自动设置 addition_info：带 attn 用 1212aedp3，不带 attn 用 1212dp3
  if [[ "${task}" == *_no_attn ]]; then
    addition_info="1212dp3"
  else
    addition_info="1212aedp3"
  fi
  
  # exp_name 格式与 train_policy.sh 保持一致：${task}-${alg_name}-${addition_info}
  exp_name="${task}-${CONFIG_NAME}-${addition_info}"
  run_dir="data/outputs/${exp_name}_seed${SEED}"
  # wandb run name：默认为 exp_name，若设置 RUN_NAME_PREFIX 则用 "${RUN_NAME_PREFIX}_${exp_name}"
  if [[ -n "${RUN_NAME_PREFIX}" ]]; then
    run_name="${RUN_NAME_PREFIX}_${exp_name}"
  else
    run_name="${exp_name}"
  fi
  
  log "开始训练: ${task} (exp_name=${exp_name}, gpu_id=${GPU_ID}, seed=${SEED}, addition_info=${addition_info})"
  python train.py --config-name=${CONFIG_NAME}.yaml \
                            task=${task} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${SEED} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            logging.name=${run_name} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            ${EXTRA_ARGS}
  task_end=$(date +%s)
  log "完成训练: ${task} 用时 $((task_end - task_start)) 秒"
done

total_end=$(date +%s)
log "全部任务完成，总耗时 $((total_end - total_start)) 秒"

