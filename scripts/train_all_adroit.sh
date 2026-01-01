#!/usr/bin/env bash
set -euo pipefail

# 一次性串行跑完 Adroit 六个实验（attn / no_attn 各三项）。
# 需在已激活的 aedp3 环境下运行。
# addition_info 自动设置：
#   - 带 attn 的任务（adroit_pen/hammer/door）: 1221aedp3
#   - 不带 attn 的任务（*_no_attn）: 1221dp3
# exp_name 格式：${task}-${CONFIG_NAME}-${addition_info}（与 train_policy.sh 完全一致）
# 环境变量可选：
#   GPU_ID=0                使用的 GPU ID（会设 CUDA_VISIBLE_DEVICES）
#   SEED=0                  训练种子
#   CONFIG_NAME=dp3         Hydra 配置名
#   EXTRA_ARGS=""           额外透传给 train.py（如 training.debug=true）
#   DATASET_TYPE=standard   数据集类型: standard(标准), gs2, env

DEBUG=False
save_ckpt=True
# 运行名前缀，可用于区分 wandb run；默认用 exp_name
RUN_NAME_PREFIX="${RUN_NAME_PREFIX:-}"
DATASET_TYPE="${DATASET_TYPE:-standard}"
ATTN_MODE="${ATTN_MODE:-all}"

ROOT="${ROOT:-$(cd "$(dirname "$0")/.."; pwd)}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
CONFIG_NAME="${CONFIG_NAME:-dp3}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# 根据 ATTN_MODE 决定运行哪些任务
if [ "${ATTN_MODE}" = "no_attn" ]; then
    TASKS=(
      adroit_pen_no_attn
      adroit_hammer_no_attn
      adroit_door_no_attn
    )
elif [ "${ATTN_MODE}" = "attn" ]; then
    TASKS=(
      adroit_pen
      adroit_hammer
      adroit_door
    )
else
    # 默认运行所有任务（向后兼容）
    TASKS=(
      adroit_pen_no_attn
      adroit_hammer_no_attn
      adroit_door_no_attn
      adroit_pen
      adroit_hammer
      adroit_door
    )
fi

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
  # 根据任务名自动设置 addition_info：带 attn 用 1221aedp3，不带 attn 用 1221dp3
  if [[ "${task}" == *_no_attn ]]; then
    addition_info="0101dp3"
  else
    addition_info="0101aedp3"
  fi
  
  # exp_name 格式与 train_policy.sh 保持一致：${task}-${alg_name}-${addition_info}
  exp_name="${task}-${CONFIG_NAME}-${addition_info}"
  # run_dir 也加上 RUN_NAME_PREFIX，避免不同前缀实验覆盖
  if [[ -n "${RUN_NAME_PREFIX}" ]]; then
    run_name="${RUN_NAME_PREFIX}_${exp_name}"
    run_dir="data/outputs/${RUN_NAME_PREFIX}_${exp_name}_seed${SEED}"
  else
    run_name="${exp_name}"
    run_dir="data/outputs/${exp_name}_seed${SEED}"
  fi
  
  # 设置数据集路径（根据 task 名计算 base 名称，并区分是否为 no_attn）
  dataset_args=""
  # strip leading 'adroit_' if present
  base="${task#adroit_}"
  is_no_attn=false
  if [[ "${base}" == *"_no_attn" ]]; then
    is_no_attn=true
    base="${base%_no_attn}"
  fi
  if [ "${DATASET_TYPE}" = "gs2" ]; then
    if [ "${is_no_attn}" = "true" ]; then
      dataset_path="data/adroit_${base}_expert_gs2.zarr"
    else
      dataset_path="data/adroit_${base}_expert_gs2_attn3d.zarr"
    fi
  elif [ "${DATASET_TYPE}" = "env" ]; then
    if [ "${is_no_attn}" = "true" ]; then
      dataset_path="data/adroit_${base}_expert_env.zarr"
    else
      dataset_path="data/adroit_${base}_expert_env_attn3d.zarr"
    fi
  fi
  if [[ -n "${dataset_path:-}" ]]; then
    # override existing key task.dataset.zarr_path in Hydra config
    dataset_args="task.dataset.zarr_path=${dataset_path}"
  fi

  log "开始训练: ${task} (exp_name=${exp_name}, gpu_id=${GPU_ID}, seed=${SEED}, addition_info=${addition_info}, dataset_type=${DATASET_TYPE})"
  python train.py --config-name=${CONFIG_NAME}.yaml \
                            task=${task} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${SEED} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            logging.name=${run_name} \
                            logging.project=aedp3_adroit_cmp_0101_gs2 \
                            checkpoint.save_ckpt=${save_ckpt} \
                            ${dataset_args} \
                            ${EXTRA_ARGS}
  task_end=$(date +%s)
  log "完成训练: ${task} 用时 $((task_end - task_start)) 秒"
done

total_end=$(date +%s)
log "全部任务完成，总耗时 $((total_end - total_start)) 秒"

