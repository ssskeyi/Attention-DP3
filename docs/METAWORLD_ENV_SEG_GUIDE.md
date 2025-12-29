# 将环境提供的分割（env_seg）接入数据流水线的迁移指南（中文）

本文档总结了在 AEDP3（Adroit）中从“不支持环境分割”到“支持环境分割”期间的所有关键改动， 并提供将相同方案迁移到 Metaworld 的逐步操作、代码片段和注意事项，便于工程复用与测试。

## 目标（明确）
- 直接从仿真器（MuJoCo）获取逐帧分割（segmentation），作为一等数据输入到数据管线，替代或补充基于视觉模型（如 GS2）的分割输出。  
- 统一分割格式为：`dtype=int32`、`shape=(H, W, 2)`（channels-last）。  
  - channels 含义：`[objtype, objid]`。  
- 采用主视图（`cameras[0]`）为 segmentation 源，避免多摄像头导致的 shape 不一致。  
- 保证 `reset()` / `step()` 返回的 timestep 结构一致，新增字段 `observation_segmentation`。  
- 在数据采集端加入防御性校验与统计（`seg_counters`），确保写入 zarr 时 shape 一致。

## 我们做了哪些改动（按模块）

### 1) 环境层（Wrapper）
- 文件：`third_party/VRL3/src/rrl_local/rrl_multicam.py`  
- 要点：
  - 新增 `render_segmentation` 参数（构造函数与 `get_obs` 支持）。  
  - 当启用时，调用 `sim.render(..., segmentation=True)` 获取原始分割输出，并规范化为 `(H,W,2)`（int32，channels-last）。  
  - 若返回 channels-first、多摄像头堆叠或单通道，会做对应转换：优先取主摄像头 `cameras[0]` 的分割并 nearest resize 到目标分辨率；若无法规范化则返回 `None`。  
  - `get_obs()` 在开启时返回 `(pixels, sensor, segmentation)`，否则返回 `(pixels, sensor)`。

### 2) Env 适配与 NamedTuple
- 文件：`third_party/VRL3/src/adroit.py`  
- 要点：
  - 在 `ExtendedTimeStepAdroit` 中加入 `observation_segmentation` 字段（默认 None，且放在非默认字段之后以避免 typing 错误）。  
  - `reset()` 与 `step()` 都要构造并返回 `observation_segmentation`，保证 reset/step 返回结构一致。  
  - `step()` 支持处理从底层 wrapper 返回的多种 observation 类型（tuple/list 或 NamedTuple），可兼容我们为 step 返回 segmentation 的改动。

### 3) Wrapper / 点云整合
- 文件：`3D-Diffusion-Policy/diffusion_policy_3d/gym_util/mjpc_wrapper.py`、`mjpc_diffusion_wrapper.py`  
- 要点：
  - 在这些 wrapper 的 `ExtendedTimeStepAdroit` 中加入 `observation_segmentation` 字段。  
  - 在 wrapper 的 `reset()` / `step()` 时把 segmentation 一并传递（如存在）。

### 4) 数据采集与 zarr 写入
- 文件：`third_party/VRL3/src/gen_demonstration_expert.py`  
- 要点：
  - 新增 CLI 参数 `--use_env_seg`（控制是否使用环境分割数据）。  
  - 在每步采集前对 `time_step.observation_segmentation` 做规范化：仅接受 `(H,W,2)` int32；若为其他 shape 则尝试转置/resize，否则记为 `None` 并统计为 skipped。  
  - 保存时仅对非 None 条目进行 `np.stack`；如果没有有效 segmentation，则跳过写入 segmentation dataset，并打印统计信息 `seg_counters`（`total_frames`、`seg_present`、`seg_normalized`、`seg_skipped`）。

### 5) Attention（attn）构建
- 文件：`scripts/convert_zarr_with_attn3d.py`  
- 要点：
  - 新增 `--use_env_seg` 参数：当启用时直接读取 zarr 中 `segmentation` 数据并用 `build_attn_from_env_seg()` 构建 `attn_3d`。  
  - `build_attn_from_env_seg()` 根据 pointcloud 的 UV 将点映射到 segmentation 的 objid，并生成 attention 通道（与 GS2 方法的通道定义一致）。

### 6) 脚本与调试工具
- `scripts/make_adroit_datasets.sh`：增加 `SEG_TYPES` / `USE_ENV_SEG` 支持，自动生成 `adroit_{task}_expert_env.zarr` 等数据集。  
- `tools/inspect_env_seg.py`（新增）：用于快速可视化 reset 与若干 step 的 `objtype` / `objid` 图像，并打印 id 样本，便于调试和确认分割质量。

## 为什么这些改动合理（严谨说明）
- MuJoCo 原生支持 offscreen segmentation 渲染；在 env 层读取该输出能获得仿真物理层面的精确分割，避免视觉模型带来的误差与额外开销。  
- 在环境边界做数据规范化可以保证后续处理（np.stack、zarr 写入、attn 生成）面对一致的输入，根本上消除 shape 不一致导致的运行时错误。  
- 将 segmentation 作为 timestep 的一部分，保持 observation 数据结构稳定，便于 replay buffer、训练与评估流程统一访问。

## 在 Metaworld 中复用——逐步清单

1) 在 Metaworld 的 env wrapper 中增加 `render_segmentation` 参数（默认 False）。当 True 时，使 wrapper 在 observation 中返回 segmentation。  

2) 在像素/相机采集 wrapper（假设只使用主视图）实现 `normalize_seg(raw_seg, H, W)`，参考实现：  

```python
raw_seg = sim.render(width=H, height=W, camera_name=primary_cam, segmentation=True)
def normalize_seg(raw_seg, H, W):
    seg = np.array(raw_seg)
    # channels-first -> channels-last
    if seg.ndim == 3 and seg.shape[0] == 2 and seg.shape[2] != 2:
        seg = seg.transpose((1,2,0))
    # 多摄像头堆叠 -> 取主摄像头
    if seg.ndim == 4 and seg.shape[0] > 1 and seg.shape[3] == 2:
        seg = seg[0]
    # 单通道 -> 扩展为两通道 [objtype, objid]
    if seg.ndim == 2:
        seg = np.stack([np.zeros_like(seg), seg], axis=2)
    # resize 到 (H,W)，使用最近邻保持 id
    if (seg.shape[0], seg.shape[1]) != (H, W):
        ch0 = Image.fromarray(seg[...,0].astype(np.uint8)).resize((W, H), resample=Image.NEAREST)
        ch1 = Image.fromarray(seg[...,1].astype(np.uint8)).resize((W, H), resample=Image.NEAREST)
        seg = np.stack([np.array(ch0, dtype=np.int32), np.array(ch1, dtype=np.int32)], axis=2)
    return seg.astype(np.int32)
```

3) 在 timestep 的 NamedTuple 中加入 `observation_segmentation`（默认 None），并在 `reset()` / `step()` 中包含该字段。注意：新增带默认值字段应放在非默认字段之后。  

4) 在数据采集脚本中新增 `--use_env_seg`，并在 append 前调用 `normalize_seg()`，对失败案例 append `None` 并统计。写入 zarr 时只对有效条目做 `np.stack`。  

5) 在 attn 构建脚本中添加读取 zarr segmentation 的路径，并实现 `build_attn_from_env_seg()`（按 UV 映射构建 attn 通道）。  

6) 添加调试工具并执行 smoke test（1–5 episodes），检查 `seg_counters` 与生成的 `objid` 可视化图像，确保大部分帧能成功规范化。

## 关键注意点（必须关注）
- **objid 范围**：PIL 使用 uint8 时会截断 >255 的 id；若仿真可能返回 >255 的 objid，需先 remap 或使用支持更大 dtype 的缩放方法（例如 scikit-image）。  
- **NamedTuple 字段顺序**：带默认值的字段必须放在非默认字段之后，否则会导致 Python typing 报错。构造 NamedTuple 时推荐使用关键字参数。  
- **多摄像头**：当前实现默认使用主视图 `cameras[0]`；若需要 multi-view，请提前定义 deterministic 的合并规则并保证输出 shape 稳定。  
- **下游兼容性**：确保 replay buffer / dataset reader 能兼容有或无 segmentation 的 zarr。

## 调试与常用命令
- 可视化 env segmentation（示例）：
```bash
PYTHONPATH=third_party/VRL3/src:3D-Diffusion-Policy/diffusion_policy_3d \
  python tools/inspect_env_seg.py --env door --steps 6 --out debug_seg
```
- 生成小样本数据（使用 env seg）：
```bash
MAX_EP=5 USE_ENV_SEG=true CUDA_VISIBLE_DEVICES=0 bash scripts/make_adroit_datasets.sh
```

## 附录：向 Metaworld 应用的最小改动
- 在 Metaworld wrapper 构造函数中添加 `render_segmentation`；在 `get_obs` 中调用 `normalize_seg()` 并把 `observation_segmentation` 放入 timestep。  

## 后续（可选）
- 我可以为 Metaworld 生成 PR 风格的 patch 模板（按文件给出 diff），或为 Metaworld 提供单元测试验证 segmentation 的 shape/dtype。如需我继续，可告知偏好。


