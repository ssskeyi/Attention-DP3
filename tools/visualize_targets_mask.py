#!/usr/bin/env python3
"""
Visualize segmentation mask filtered by target geom ids.

Usage:
  python tools/visualize_targets_mask.py --task door --zarr_path path/to/adroit_door_expert_env.zarr --output_dir debug_targets --targets targets/door_geom_ids.json

If --targets is not provided the script will try to load targets/<task>_geom_ids.json.
"""
import argparse
import os
import json
import numpy as np
import zarr
import matplotlib.pyplot as plt


def load_targets(task, targets_path=None):
    if targets_path:
        path = targets_path
    else:
        path = os.path.join("targets", f"{task}_geom_ids.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def overlay_binary_mask(img, mask, color=(255, 0, 0), alpha=0.5):
    """Overlay binary mask (H,W) onto RGB image (H,W,3) and return uint8 image."""
    canvas = img.copy().astype(np.float32)
    mask_bool = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    canvas[mask_bool] = canvas[mask_bool] * (1.0 - alpha) + color_arr * alpha
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    return canvas


def visualize_frame(img, seg_data, targets_set, out_path, frame_idx, task_name):
    H, W = img.shape[:2]
    objid = seg_data[:, :, 1]

    # binary mask of target geom ids
    if targets_set is None:
        mask_target = objid > 0
    else:
        mask_target = np.isin(objid, targets_set)

    # build colored objid map for reference
    unique_objids = np.unique(objid)
    unique_objids = unique_objids[unique_objids > 0]
    # simple color map
    cmap = plt.cm.get_cmap("tab20", max(1, len(unique_objids)))
    objid_colored = np.zeros((H, W, 3), dtype=np.uint8)
    for i, oid in enumerate(unique_objids):
        color = (np.array(cmap(i)[:3]) * 255).astype(np.uint8)
        objid_colored[objid == oid] = color

    # overlay target mask on original image
    overlay = overlay_binary_mask(img, mask_target, color=(255, 0, 0), alpha=0.5)

    # plot three panels: original, overlay, objid colored
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title(f"Original Image\n{task_name} Frame {frame_idx}")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Target Mask Overlay")
    axes[1].axis("off")

    axes[2].imshow(objid_colored)
    axes[2].set_title("Object ID Mask (color-coded)")
    axes[2].axis("off")

    # (no legend)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation filtered by targets")
    parser.add_argument("--task", required=True, choices=["door", "hammer", "pen"])
    parser.add_argument("--zarr_path", required=True)
    parser.add_argument("--targets", default=None, help="path to targets json (overrides default)")
    parser.add_argument("--output_dir", default="debug_targets", help="output directory")
    parser.add_argument("--max_episodes", type=int, default=3)
    parser.add_argument("--frames_per_ep", type=int, default=5)
    args = parser.parse_args()

    targets = load_targets(args.task, args.targets)
    if targets is None:
        print(f"[warn] No targets file found for task {args.task}. Falling back to all non-zero objids.")
    else:
        print(f"[info] Loaded {len(targets)} target geom ids.")
    task_output_dir = os.path.join(args.output_dir, f"{args.task}_targets")
    os.makedirs(task_output_dir, exist_ok=True)

    # load zarr
    store = zarr.DirectoryStore(args.zarr_path)
    z = zarr.group(store=store)
    if "data" not in z:
        print("[error] zarr missing 'data' group")
        return
    data = z["data"]
    if "segmentation" not in data or "img" not in data:
        print("[error] zarr missing required datasets ('segmentation' or 'img')")
        return

    imgs = data["img"][:]
    segs = data["segmentation"][:]
    ep_ends = z["meta"]["episode_ends"][:]
    ep_starts = [0] + ep_ends[:-1].tolist()
    n_eps = min(args.max_episodes, len(ep_starts))

    for ep_idx in range(n_eps):
        ep_start = ep_starts[ep_idx]
        ep_end = int(ep_ends[ep_idx])
        ep_len = ep_end - ep_start
        frames_to_vis = min(args.frames_per_ep, ep_len)
        if frames_to_vis <= 0:
            continue
        frame_indices = np.linspace(0, ep_len - 1, frames_to_vis, dtype=int)
        ep_dir = os.path.join(task_output_dir, f"ep_{ep_idx:03d}")
        os.makedirs(ep_dir, exist_ok=True)
        for local_idx in frame_indices:
            global_idx = ep_start + int(local_idx)
            img = imgs[global_idx]
            seg = segs[global_idx]
            out_path = os.path.join(ep_dir, f"frame_{global_idx:06d}_targets.png")
            visualize_frame(img, seg, targets, out_path, global_idx, args.task)
            print(f"[vis] Saved {out_path}")

    print(f"[done] Results in {task_output_dir}")


if __name__ == "__main__":
    main()


