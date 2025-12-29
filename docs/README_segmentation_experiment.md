# Adroit分割实验说明

## 实验目标

比较不同分割方法对Adroit任务性能的影响：

1. **GS2分割**: 使用Grounded-SAM-2进行视觉分割（可能有噪声）
2. **环境分割**: 直接从MuJoCo环境获取精确的物理分割
3. **有/无Attention**: 在每种分割方法上分别训练带attention和不带attention的模型

## 实验设计

每个任务会生成4种实验配置：

| 配置名称 | 分割方法 | Attention | 数据集文件名 |
|---------|---------|----------|------------|
| `gs2_no_attn_{task}` | GS2分割 | 无 | `adroit_{task}_expert_gs2.zarr` |
| `gs2_attn_{task}` | GS2分割 | 有 | `adroit_{task}_expert_gs2_attn3d.zarr` |
| `env_no_attn_{task}` | 环境分割 | 无 | `adroit_{task}_expert_env.zarr` |
| `env_attn_{task}` | 环境分割 | 有 | `adroit_{task}_expert_env_attn3d.zarr` |

## 使用方法

### 一键运行完整实验

```bash
# 基本设置（可选）
export GPU_ID=0              # 训练GPU
export DATA_GPU=0            # 数据生成GPU
export SEED=42               # 随机种子
export MAX_EP=50             # 每个任务的演示数量

# 运行完整实验
bash scripts/run_adroit_segmentation_experiment.sh
```

### 分阶段运行

#### 1. 只生成数据
```bash
export SEG_TYPES="gs2 env"  # 生成GS2和环境分割数据
export MAX_EP=50
bash scripts/make_adroit_datasets.sh
```

#### 2. 只运行训练
```bash
export SKIP_DATA_GEN=true   # 跳过数据生成

# 训练GS2分割的实验
export DATASET_TYPE=gs2
export RUN_NAME_PREFIX=gs2_no_attn
bash scripts/train_all_adroit.sh

export RUN_NAME_PREFIX=gs2_attn
bash scripts/train_all_adroit.sh

# 训练环境分割的实验
export DATASET_TYPE=env
export RUN_NAME_PREFIX=env_no_attn
bash scripts/train_all_adroit.sh

export RUN_NAME_PREFIX=env_attn
bash scripts/train_all_adroit.sh
```

## 输出文件结构

```
3D-Diffusion-Policy/data/
├── adroit_door_expert_gs2.zarr/          # GS2分割基础数据
├── adroit_door_expert_gs2_attn3d.zarr/   # GS2分割+attention数据
├── adroit_door_expert_env.zarr/          # 环境分割基础数据
├── adroit_door_expert_env_attn3d.zarr/   # 环境分割+attention数据
└── outputs/
    ├── gs2_no_attn_adroit_door-dp3-1226dp3_seed0/
    ├── gs2_attn_adroit_door-dp3-1226aedp3_seed0/
    ├── env_no_attn_adroit_door-dp3-1226dp3_seed0/
    └── env_attn_adroit_door-dp3-1226aedp3_seed0/
```

## 关键参数说明

### 数据生成参数
- `SEG_TYPES`: 分割类型，`"gs2 env"` 或单独 `"gs2"` / `"env"`
- `MAX_EP`: 每个任务生成的episode数量
- `GPU`: 数据生成使用的GPU ID

### 训练参数
- `DATASET_TYPE`: `"gs2"` 或 `"env"`，指定使用哪种分割的数据集
- `RUN_NAME_PREFIX`: wandb运行名称前缀，用于区分实验
- `GPU_ID`: 训练使用的GPU ID
- `SEED`: 随机种子

## 预期结果分析

通过比较以下指标来评估分割质量对性能的影响：

1. **收敛速度**: 不同配置的训练收敛速度对比
2. **最终性能**: 成功率、奖励等指标的最终表现
3. **分割准确性**: GS2 vs 环境分割的attention质量差异

理论预期：
- 环境分割应该提供更准确的attention，带来更好的性能
- GS2分割可能引入噪声，但仍比无attention表现更好
- 完全准确的分割（环境分割）应该展现attention机制的最大潜力

## 注意事项

1. **存储空间**: 每种分割方法会生成完整的数据集，需要足够的存储空间
2. **计算时间**: GS2分割需要额外的推理时间，环境分割相对更快
3. **GPU内存**: attention模型需要更多GPU内存
4. **随机种子**: 使用相同种子确保公平比较</contents>
</xai:function_call">Scripts directory already contains a file named README_segmentation_experiment.md. Do you want to overwrite it?
