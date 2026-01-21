#!/usr/bin/env bash
set -euo pipefail

# AEDP3消融实验运行脚本
# 支持注意力通道消融和编码器融合策略消融
# 需在已激活的 aedp3 环境下运行。
# 环境变量可选：
#   GPU_ID=0                        # 使用的 GPU ID
#   SEED=0                          # 训练种子
#   CONFIG_NAME=dp3                 # Hydra 配置名
#   TASKS="adroit_pen adroit_door"  # 指定要运行的任务，不设置则运行全部
#   ABLATION_VARIANTS=""            # 指定要运行的消融变体，不设置则运行全部
#   GS2_PORT=5000                   # GS2服务端口（仅注意力相关任务需要）
#   EXTRA_ARGS=""                   # 额外透传给 train.py的参数

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
  adroit_pen
  adroit_door
)

# 消融实验变体定义
# 格式: ID|名称|通道组合|融合策略|预期验证
declare -a ABLATION_VARIANTS=(
  # 基准线变体
  "DP3|DP3-baseline|no_attn|late_fusion|原始基准性能"
  "AEDP3-Full|AEDP3-full|all_channels|late_fusion|完整系统性能"

  # 注意力通道消融 - 单通道
  "AEDP3-C0|AEDP3-single-C0|channel_0|late_fusion|二进制mask的独立作用"
  "AEDP3-C1|AEDP3-single-C1|channel_1|late_fusion|距离加权的独立作用"
  "AEDP3-C2|AEDP3-single-C2|channel_2|late_fusion|反向注意力的独立作用"

  # 注意力通道消融 - 双通道
  "AEDP3-C01|AEDP3-dual-C01|channels_01|late_fusion|目标识别+重要性排序"
  "AEDP3-C02|AEDP3-dual-C02|channels_02|late_fusion|目标vs背景区分"
  "AEDP3-C12|AEDP3-dual-C12|channels_12|late_fusion|重要性+避障"

  # 编码器融合消融
  "AEDP3-Late|AEDP3-late-fusion|all_channels|late_fusion|晚期融合基准"
  "AEDP3-Early|AEDP3-early-fusion|all_channels|early_fusion|早期融合对比"
)

# 如果设置了TASKS环境变量，使用它；否则使用默认任务
if [[ -n "${TASKS:-}" ]]; then
  # 将TASKS字符串转换为数组
  IFS=' ' read -r -a TASKS_ARRAY <<< "$TASKS"
else
  TASKS_ARRAY=("${DEFAULT_TASKS[@]}")
fi

# 如果设置了ABLATION_VARIANTS环境变量，使用它；否则使用全部变体
if [[ -n "${ABLATION_VARIANTS:-}" ]]; then
  # 将ABLATION_VARIANTS字符串转换为数组（按逗号分割）
  IFS=',' read -r -a SELECTED_VARIANTS <<< "$ABLATION_VARIANTS"
  # 过滤出匹配的变体
  FILTERED_VARIANTS=()
  for variant_info in "${ABLATION_VARIANTS[@]}"; do
    variant_id=$(echo "$variant_info" | cut -d'|' -f1)
    for selected in "${SELECTED_VARIANTS[@]}"; do
      if [[ "$variant_id" == "$selected" ]]; then
        FILTERED_VARIANTS+=("$variant_info")
        break
      fi
    done
  done
  ABLATION_VARIANTS=("${FILTERED_VARIANTS[@]}")
fi

log() { echo -e "[ablation_experiments] $*"; }

if [ $DEBUG = True ]; then
    wandb_mode=offline
else
    wandb_mode=online
fi

cd "${ROOT}/3D-Diffusion-Policy"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${GPU_ID}

total_start=$(date +%s)

log "开始消融实验，共 ${#ABLATION_VARIANTS[@]} 个变体，${#TASKS_ARRAY[@]} 个任务"

# 遍历所有消融变体
for variant_info in "${ABLATION_VARIANTS[@]}"; do
  # 解析变体信息
  variant_id=$(echo "$variant_info" | cut -d'|' -f1)
  variant_name=$(echo "$variant_info" | cut -d'|' -f2)
  channels=$(echo "$variant_info" | cut -d'|' -f3)
  fusion_strategy=$(echo "$variant_info" | cut -d'|' -f4)
  description=$(echo "$variant_info" | cut -d'|' -f5)

  log "运行消融变体: $variant_id ($variant_name) - $description"

  # 为每个变体设置特定的配置参数
  variant_extra_args="$EXTRA_ARGS"

  # 根据通道配置设置参数
  case $channels in
    "no_attn")
      # DP3基准线：不使用注意力
      variant_extra_args="$variant_extra_args dataset.use_attn_3d=false"
      ;;
    "all_channels")
      # 全通道
      variant_extra_args="$variant_extra_args dataset.use_attn_3d=true"
      ;;
    "channel_0")
      # 仅通道0
      variant_extra_args="$variant_extra_args dataset.use_attn_3d=true policy.attn_channels=[0]"
      ;;
    "channel_1")
      # 仅通道1
      variant_extra_args="$variant_extra_args dataset.use_attn_3d=true policy.attn_channels=[1]"
      ;;
    "channel_2")
      # 仅通道2
      variant_extra_args="$variant_extra_args dataset.use_attn_3d=true policy.attn_channels=[2]"
      ;;
    "channels_01")
      # 通道0+1
      variant_extra_args="$variant_extra_args dataset.use_attn_3d=true policy.attn_channels=[0,1]"
      ;;
    "channels_02")
      # 通道0+2
      variant_extra_args="$variant_extra_args dataset.use_attn_3d=true policy.attn_channels=[0,2]"
      ;;
    "channels_12")
      # 通道1+2
      variant_extra_args="$variant_extra_args dataset.use_attn_3d=true policy.attn_channels=[1,2]"
      ;;
  esac

  # 根据融合策略设置参数
  case $fusion_strategy in
    "early_fusion")
      variant_extra_args="$variant_extra_args policy.fusion_strategy=early"
      ;;
    "late_fusion")
      variant_extra_args="$variant_extra_args policy.fusion_strategy=late"
      ;;
  esac

  # 遍历每个任务
  for task in "${TASKS_ARRAY[@]}"; do
    task_start=$(date +%s)

    # 设置任务特定的参数
    case $task in
      "adroit_pen")
        base_task="adroit_pen"
        ;;
      "adroit_door")
        base_task="adroit_door"
        ;;
      *)
        log "警告: 未知任务 $task，跳过"
        continue
        ;;
    esac

    # 根据变体设置数据集路径和GS2配置
    if [[ "$channels" == "no_attn" ]]; then
      dataset_path="data/${base_task}_expert.zarr"
      # DP3任务不需要GS2 API URL
      gs2_args=""
    else
      dataset_path="data/${base_task}_expert_attn3d.zarr"
      # GS2任务需要设置GS2 API URL环境变量
      export GS2_API_URL="http://127.0.0.1:${GS2_PORT}"
      gs2_args="gs2_api_url=${GS2_API_URL}"
    fi

    # 构建实验名称和运行名称
    exp_name="${task}-${variant_id}-${CONFIG_NAME}-abl"
    run_name="${exp_name}"
    if [[ -n "${RUN_NAME_PREFIX}" ]]; then
      run_name="${RUN_NAME_PREFIX}_${run_name}"
    fi

    run_dir="data/outputs/${exp_name}_seed${SEED}"

    log "  训练任务: ${task} (变体=${variant_id}, exp_name=${exp_name}, gpu_id=${GPU_ID}, seed=${SEED})"

    # 运行训练
    python train.py --config-name=${CONFIG_NAME}.yaml \
                        task=${task} \
                        hydra.run.dir=${run_dir} \
                        training.debug=$DEBUG \
                        training.seed=${SEED} \
                        training.device="cuda:0" \
                        exp_name=${exp_name} \
                        logging.mode=${wandb_mode} \
                        logging.name=${run_name} \
                        logging.project=aedp3_ablation_experiments \
                        checkpoint.save_ckpt=${save_ckpt} \
                        task.dataset.zarr_path=${dataset_path} \
                        ${gs2_args} \
                        ${variant_extra_args}

    task_end=$(date +%s)
    log "  完成任务: ${task} 用时 $((task_end - task_start)) 秒"
  done

  log "完成消融变体: $variant_id"
done

total_end=$(date +%s)
log "全部消融实验完成，总耗时 $((total_end - total_start)) 秒"
