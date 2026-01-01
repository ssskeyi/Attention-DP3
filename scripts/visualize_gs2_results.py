#!/usr/bin/env python3
# 可视化 scripts/gs2.sh 生成的 Grounded-SAM 2 结果

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pycocotools.mask as mask_util


def color_from_name(name: str) -> Tuple[int, int, int]:
    """根据类别名生成稳定的 RGB 颜色。"""
    h = int(hashlib.md5(name.encode("utf-8")).hexdigest()[:6], 16)
    return ((h >> 16) & 255, (h >> 8) & 255, h & 255)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    """用半透明方式叠加分割掩码。"""
    mask_bool = mask.astype(bool)
    image[mask_bool] = image[mask_bool] * (1.0 - alpha) + np.array(color) * alpha
    return image


def visualize_frame(json_path: str, frames_root: str, output_root: str, score_thr: float, alpha: float, overwrite: bool):
    ep_dir = Path(json_path).parent.name
    stem = Path(json_path).stem
    frame_path = os.path.join(frames_root, ep_dir, f"{stem}.png")
    out_dir = os.path.join(output_root, ep_dir)
    out_path = os.path.join(out_dir, f"{stem}.png")

    if (not overwrite) and os.path.exists(out_path):
        return

    if not os.path.exists(frame_path):
        print(f"[warn] 跳过 {json_path}，找不到对应帧：{frame_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[warn] 读取图像失败：{frame_path}")
        return

    anns = data.get("annotations", [])
    canvas = img.copy()

    for ann in anns:
        raw_score = ann.get("score", 0.0)
        if isinstance(raw_score, list):
            # 当保存的是列表时（例如多掩码/多分数），取最大值
            try:
                score_val = max(float(s) for s in raw_score)
            except Exception:
                score_val = 0.0
        else:
            try:
                score_val = float(raw_score)
            except Exception:
                score_val = 0.0

        if score_val < score_thr:
            continue

        color = color_from_name(ann.get("class_name", "obj"))
        bbox = ann.get("bbox", [0, 0, 0, 0])
        x1, y1, x2, y2 = [int(v) for v in bbox]

        seg = ann.get("segmentation")
        if seg:
            mask = mask_util.decode(seg)
            if mask.ndim == 3:
                mask = mask[..., 0]
            canvas = overlay_mask(canvas, mask, color, alpha)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 1)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{ann.get('class_name', 'obj')} {score_val:.2f}"
        cv2.putText(canvas, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(out_path, canvas)


def main():
    parser = argparse.ArgumentParser(description="可视化 gs2.sh 输出的 JSON 分割结果。")
    base = "/mnt/disk2/ycb/AEDP3/3D-Diffusion-Policy"
    parser.add_argument(
        "--frames_root",
        default=f"{base}/export/adroit_hammer_gs2_frames",
        help="gs2.sh 的 $1，对应原始帧目录。",
    )
    parser.add_argument(
        "--gs2_root",
        default=f"{base}/export_gs2/adroit_hammer_gs2",
        help="gs2.sh 的 $2，对应 JSON 结果目录。",
    )
    parser.add_argument(
        "--output_root",
        default=f"{base}/export_gs2_vis/adroit_hammer_gs2_vis",
        help="可视化结果输出目录。",
    )
    parser.add_argument("--score_thr", type=float, default=0.0, help="过滤低置信度结果的阈值。")
    parser.add_argument("--alpha", type=float, default=0.5, help="掩码叠加透明度。")
    parser.add_argument("--overwrite", action="store_true", help="若已存在同名可视化文件则重新生成。")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    ep_dirs = sorted(glob.glob(os.path.join(args.gs2_root, "ep_*")))
    for ep in ep_dirs:
        json_files = sorted(glob.glob(os.path.join(ep, "frame_*.json")))
        for jp in json_files:
            visualize_frame(jp, args.frames_root, args.output_root, args.score_thr, args.alpha, args.overwrite)
        print(f"[vis] {ep} -> {os.path.join(args.output_root, os.path.basename(ep))}")


if __name__ == "__main__":
    main()

