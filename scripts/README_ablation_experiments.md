# AEDP3消融实验脚本使用指南

## 概述

本脚本用于运行AEDP3的消融实验，支持注意力通道消融和编码器融合策略消融。

## 实验变体

### 基准线变体
- **DP3**: 原始基准线，使用PointNet++编码3D坐标
- **AEDP3-Full**: 完整AEDP3，使用所有注意力通道和晚期融合

### 注意力通道消融
- **AEDP3-C0**: 单通道-物体（通道0：二进制mask）
- **AEDP3-C1**: 单通道-中心（通道1：距离加权）
- **AEDP3-C2**: 单通道-背景（通道2：反向注意力）
- **AEDP3-C01**: 双通道-物体+中心（通道0+1）
- **AEDP3-C02**: 双通道-物体+背景（通道0+2）
- **AEDP3-C12**: 双通道-中心+背景（通道1+2）

### 编码器融合消融
- **AEDP3-Late**: 晚期融合（当前默认实现）
- **AEDP3-Early**: 早期融合（注意力场与点云数据先拼接后编码）

## 使用方法

### 运行所有消融实验
```bash
# 运行所有预定义的消融实验变体
bash scripts/run_ablation_experiments.sh
```

### 运行指定变体
```bash
# 只运行指定的变体（用逗号分隔）
export ABLATION_VARIANTS="DP3,AEDP3-Full,AEDP3-C0"
bash scripts/run_ablation_experiments.sh
```

### 运行指定任务
```bash
# 只在指定任务上运行
export TASKS="adroit_pen adroit_door"
bash scripts/run_ablation_experiments.sh
```

### 自定义GPU和种子
```bash
export GPU_ID=1
export SEED=42
bash scripts/run_ablation_experiments.sh
```

### 设置GS2服务端口
```bash
# 对于注意力相关的任务，需要设置GS2服务端口
export GS2_PORT=5000
bash scripts/run_ablation_experiments.sh
```

### 添加额外参数
```bash
export EXTRA_ARGS="training.debug=true training.max_train_steps=1000"
bash scripts/run_ablation_experiments.sh
```

## 实验输出

每个实验会在 `data/outputs/` 目录下创建独立的文件夹，命名格式为：
```
{task}-{variant_id}-{config_name}-abl_seed{seed}/
```

例如：
- `adroit_pen-DP3-dp3-abl_seed0/`
- `adroit_pen-AEDP3-C0-dp3-abl_seed0/`

## 技术实现

### 注意力通道选择
通过 `policy.attn_channels` 参数控制使用的注意力通道：
- `null`: 使用所有通道（默认）
- `[0]`: 仅使用通道0
- `[0,1]`: 使用通道0和1

### 融合策略
通过 `policy.fusion_strategy` 参数控制融合方式：
- `late`: 晚期融合（分别编码后拼接）
- `early`: 早期融合（拼接后统一编码）

## 数据集要求

- **无注意力任务**: 使用 `data/adroit_pen_expert.zarr`
- **注意力任务**: 使用 `data/adroit_pen_expert_attn3d.zarr`

确保数据已通过 `scripts/convert_zarr_with_attn3d.py` 预处理。

## 多终端并行运行

对于大规模消融实验，建议使用多终端并行运行来提高效率。参考 `command_ablation_experiments.txt` 获取完整的多终端命令配置。

### 基本设置：
1. **启动GS2服务器**：在不同的GPU上启动多个GS2服务器（端口5000, 5001, 5002, 5003等）
2. **分配训练任务**：每个终端负责不同的任务或变体组合
3. **设置正确的端口**：确保每个训练任务使用对应的GS2_PORT

### 示例配置：
```bash
# 终端1: GS2服务器 (GPU0, 端口5000)
cd Grounded-SAM-2
CUDA_VISIBLE_DEVICES=0 python gs2_api_server.py --device cuda:0 --port 5000

# 终端2: 训练任务 (GPU1, 使用端口5000的GS2服务)
export CUDA_VISIBLE_DEVICES=1
export GPU_ID=1
export GS2_PORT=5000
export TASKS="adroit_pen"
export ABLATION_VARIANTS="DP3,AEDP3-Full"
bash scripts/run_ablation_experiments.sh
```

## 注意事项

1. 确保在激活的aedp3环境中运行
2. 确保数据路径正确
3. 不同变体可能需要不同的计算资源
4. 早期融合变体会改变输入维度，特征维度可能与晚期融合不同
5. **GS2服务依赖**：注意力相关的变体需要运行GS2服务器，非注意力变体（DP3）不需要

## 测试配置

运行测试脚本来验证配置：

```bash
# 测试消融实验参数配置
bash scripts/test_ablation_config.sh

# 测试GS2服务配置
bash scripts/test_ablation_gs2_config.sh
```

这些测试脚本将显示每个变体的参数配置，帮助验证设置是否正确。
