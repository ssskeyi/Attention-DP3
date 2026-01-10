#!/usr/bin/env python3
"""
可视化原始帧数据

从帧目录中读取PNG文件，为每个episode生成可视化结果。
用于对比GS2分割结果的原始输入。
"""

import os
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from tqdm import tqdm


def create_episode_grid(frames_dir, episode_id, max_frames=10, grid_cols=5):
    """
    为单个episode创建帧网格可视化

    Args:
        frames_dir: episode帧目录路径
        episode_id: episode ID
        max_frames: 最大显示帧数
        grid_cols: 网格列数
    """
    # 查找所有帧文件
    frame_pattern = os.path.join(frames_dir, "frame_*.png")
    frame_files = sorted(glob.glob(frame_pattern))

    if not frame_files:
        print(f"Warning: No frames found in {frames_dir}")
        return None

    # 限制帧数
    frame_files = frame_files[:max_frames]
    n_frames = len(frame_files)

    if n_frames == 0:
        return None

    # 计算网格布局
    grid_rows = int(np.ceil(n_frames / grid_cols))

    # 读取第一帧获取尺寸
    first_frame = Image.open(frame_files[0])
    frame_width, frame_height = first_frame.size

    # 创建大图
    grid_width = frame_width * grid_cols
    grid_height = frame_height * grid_rows
    grid_img = Image.new('RGB', (grid_width, grid_height), color='white')

    # 将帧粘贴到网格中
    for i, frame_file in enumerate(frame_files):
        frame_img = Image.open(frame_file)
        row = i // grid_cols
        col = i % grid_cols
        x = col * frame_width
        y = row * frame_height
        grid_img.paste(frame_img, (x, y))

    # 添加标题
    plt.figure(figsize=(grid_cols * 4, grid_rows * 3))
    plt.imshow(grid_img)
    plt.title(f'Episode {episode_id:04d} - Raw Frames ({n_frames} frames)', fontsize=14, pad=20)
    plt.axis('off')

    return plt.gcf()


def create_episode_animation(frames_dir, episode_id, output_path):
    """
    为单个episode创建帧动画（可选）

    Args:
        frames_dir: episode帧目录路径
        episode_id: episode ID
        output_path: 输出GIF路径
    """
    frame_pattern = os.path.join(frames_dir, "frame_*.png")
    frame_files = sorted(glob.glob(frame_pattern))

    if len(frame_files) < 2:
        return  # 不够帧数创建动画

    frames = []
    for frame_file in frame_files[:20]:  # 限制帧数
        frames.append(imageio.imread(frame_file))

    # 创建GIF动画
    imageio.mimsave(output_path, frames, duration=0.2, loop=0)


def main():
    parser = argparse.ArgumentParser(description="可视化原始帧数据")
    parser.add_argument(
        "--frames_root",
        type=str,
        required=True,
        help="帧数据根目录路径"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="可视化输出根目录"
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=10,
        help="最大处理的episode数"
    )
    parser.add_argument(
        "--create_animation",
        action="store_true",
        help="是否为每个episode创建GIF动画"
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=5,
        help="网格列数"
    )
    parser.add_argument(
        "--max_frames_per_episode",
        type=int,
        default=10,
        help="每个episode最多显示的帧数"
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_root, exist_ok=True)

    # 查找所有episode目录
    episode_dirs = []
    for item in os.listdir(args.frames_root):
        ep_path = os.path.join(args.frames_root, item)
        if os.path.isdir(ep_path) and item.startswith("ep_"):
            episode_dirs.append((item, ep_path))

    # 按episode ID排序
    episode_dirs.sort(key=lambda x: int(x[0].split("_")[1]))

    # 限制episode数量
    episode_dirs = episode_dirs[:args.max_episodes]

    print(f"Found {len(episode_dirs)} episodes to process")

    # 处理每个episode
    for ep_name, ep_dir in tqdm(episode_dirs, desc="Processing episodes"):
        episode_id = int(ep_name.split("_")[1])

        # 创建网格可视化
        fig = create_episode_grid(
            ep_dir,
            episode_id,
            max_frames=args.max_frames_per_episode,
            grid_cols=args.grid_cols
        )

        if fig is not None:
            # 保存网格图像
            output_path = os.path.join(args.output_root, f"episode_{episode_id:04d}_grid.png")
            fig.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved grid visualization: {output_path}")

        # 可选：创建动画
        if args.create_animation:
            anim_path = os.path.join(args.output_root, f"episode_{episode_id:04d}_animation.gif")
            create_episode_animation(ep_dir, episode_id, anim_path)
            print(f"Saved animation: {anim_path}")

    # 创建汇总报告
    summary_path = os.path.join(args.output_root, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Raw Frames Visualization Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Input directory: {args.frames_root}\n")
        f.write(f"Output directory: {args.output_root}\n")
        f.write(f"Episodes processed: {len(episode_dirs)}\n")
        f.write(f"Grid columns: {args.grid_cols}\n")
        f.write(f"Max frames per episode: {args.max_frames_per_episode}\n")
        f.write(f"Animation created: {'Yes' if args.create_animation else 'No'}\n")
        f.write("\nGenerated files:\n")
        for ep_name, _ in episode_dirs:
            episode_id = int(ep_name.split("_")[1])
            f.write(f"  - episode_{episode_id:04d}_grid.png\n")
            if args.create_animation:
                f.write(f"  - episode_{episode_id:04d}_animation.gif\n")

    print(f"Visualization complete! Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
