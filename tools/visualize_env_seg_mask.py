#!/usr/bin/env python3
"""
可视化从环境中获取的环境分割mask数据

Usage:
  python tools/visualize_env_seg_mask.py --task hammer --zarr_path data/adroit_hammer_expert_env.zarr --output_dir debug_env_seg_hammer

这将创建可视化结果，显示每个episode的帧以及对应的环境分割mask。
"""

import argparse
import os
import numpy as np
from PIL import Image
import zarr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap


def create_objid_colormap(unique_ids):
    """为不同的objid创建颜色映射"""
    n_colors = len(unique_ids)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_colors, 20)))
    if n_colors > 20:
        # 如果objid太多，重复使用颜色
        colors = np.tile(colors, (n_colors // 20 + 1, 1))[:n_colors]

    # 创建从objid到颜色的映射
    colormap = {}
    for i, objid in enumerate(unique_ids):
        colormap[objid] = colors[i][:3]  # RGB

    return colormap


def visualize_env_seg_frame(img, seg_data, frame_idx, output_path, task_name):
    """
    可视化单帧的环境分割数据

    Args:
        img: 原始图像 (H, W, 3)
        seg_data: 分割数据 (H, W, 2) - [objtype, objid]
        frame_idx: 帧索引
        output_path: 输出路径
        task_name: 任务名 (用于标题)
    """
    H, W = seg_data.shape[:2]
    objtype = seg_data[:, :, 0]  # 对象类型
    objid = seg_data[:, :, 1]    # 对象ID

    # 获取唯一的objid
    unique_objids = np.unique(objid)
    unique_objids = unique_objids[unique_objids > 0]  # 排除背景(0)

    if len(unique_objids) == 0:
        print(f"[warn] Frame {frame_idx}: No objects found (only background)")
        return

    colormap = create_objid_colormap(unique_objids)

    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 原始图像
    axes[0].imshow(img)
    axes[0].set_title(f'Original Image\n{task_name} Frame {frame_idx}')
    axes[0].axis('off')

    # objid mask (用颜色编码不同对象)
    objid_colored = np.zeros((H, W, 3), dtype=np.uint8)
    for obj_id in unique_objids:
        mask = (objid == obj_id)
        color = (np.array(colormap[obj_id]) * 255).astype(np.uint8)
        objid_colored[mask] = color

    axes[1].imshow(objid_colored)
    axes[1].set_title('Object ID Mask\n(Color-coded by objid)')
    axes[1].axis('off')

    # 创建图例
    legend_elements = []
    for obj_id in unique_objids:
        color = colormap[obj_id]
        legend_elements.append(patches.Patch(facecolor=color,
                                          label=f'objid={obj_id}'))

    axes[1].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    # objtype mask (二值化显示)
    objtype_binary = (objtype > 0).astype(np.uint8) * 255
    axes[2].imshow(objtype_binary, cmap='gray')
    axes[2].set_title('Object Type Mask\n(Binary: obj vs background)')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[vis] Frame {frame_idx}: Found {len(unique_objids)} objects with objids: {unique_objids}")


def main():
    parser = argparse.ArgumentParser(description="可视化环境分割mask数据")
    parser.add_argument("--task", type=str, required=True,
                       choices=["door", "hammer", "pen"],
                       help="任务名称")
    parser.add_argument("--zarr_path", type=str, required=True,
                       help="zarr文件路径")
    parser.add_argument("--output_dir", type=str, default="debug_env_seg_vis",
                       help="输出目录")
    parser.add_argument("--max_episodes", type=int, default=3,
                       help="最多可视化多少个episodes")
    parser.add_argument("--frames_per_ep", type=int, default=5,
                       help="每个episode可视化多少帧")
    parser.add_argument("--start_frame", type=int, default=0,
                       help="起始帧索引")

    args = parser.parse_args()

    # 创建输出目录
    task_output_dir = os.path.join(args.output_dir, f"{args.task}_env_seg")
    os.makedirs(task_output_dir, exist_ok=True)

    # 读取zarr文件
    print(f"Loading zarr: {args.zarr_path}")
    store = zarr.DirectoryStore(args.zarr_path)
    zarr_root = zarr.group(store=store)

    # 检查是否包含segmentation数据
    if "segmentation" not in zarr_root["data"]:
        print(f"[error] zarr文件不包含segmentation数据: {args.zarr_path}")
        return

    # 获取数据
    imgs = zarr_root["data"]["img"][:]      # (N, H, W, 3)
    segs = zarr_root["data"]["segmentation"][:]  # (N, H, W, 2)
    episode_ends = zarr_root["meta"]["episode_ends"][:]

    print(f"Total frames: {len(imgs)}")
    print(f"Image shape: {imgs.shape}")
    print(f"Segmentation shape: {segs.shape}")

    # 计算episode边界
    ep_starts = [0] + episode_ends[:-1].tolist()
    ep_ends = episode_ends.tolist()

    # 可视化指定数量的episodes
    episodes_to_vis = min(args.max_episodes, len(ep_starts))

    for ep_idx in range(episodes_to_vis):
        ep_start = ep_starts[ep_idx]
        ep_end = ep_ends[ep_idx]
        ep_length = ep_end - ep_start

        print(f"\nProcessing episode {ep_idx} (frames {ep_start}:{ep_end})")

        # 为每个episode创建子目录
        ep_output_dir = os.path.join(task_output_dir, "03d")
        os.makedirs(ep_output_dir, exist_ok=True)

        # 选择要可视化的帧
        frames_to_vis = min(args.frames_per_ep, ep_length)
        frame_indices = np.linspace(args.start_frame,
                                   min(ep_length - 1, args.start_frame + args.frames_per_ep * 10),
                                   frames_to_vis, dtype=int)

        for local_frame_idx in frame_indices:
            global_frame_idx = ep_start + local_frame_idx

            if global_frame_idx >= len(imgs):
                break

            # 获取数据
            img = imgs[global_frame_idx]  # (H, W, 3)
            seg_data = segs[global_frame_idx]  # (H, W, 2)

            # 输出路径
            output_path = os.path.join(ep_output_dir, "06d")

            # 可视化
            visualize_env_seg_frame(img, seg_data, global_frame_idx, output_path, args.task)

    print(f"\n可视化完成！结果保存在: {task_output_dir}")
    print("\n每个图像包含三列："    print("1. 原始图像")
    print("2. 对象ID mask (用颜色编码不同对象)"    print("3. 对象类型mask (二值化显示)")


if __name__ == "__main__":
    main()
