import os
import argparse
import json
import numpy as np
import zarr
import numcodecs
import pycocotools.mask as mask_util
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer, get_optimal_chunks
import re
import torch
try:
    import pytorch3d.ops as torch3d_ops
    HAS_TORCH3D = True
except ImportError:
    HAS_TORCH3D = False
    print("Warning: pytorch3d not found, falling back to random sampling for attention generation")

def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def point_cloud_sampling(point_cloud: np.ndarray, num_points: int, method: str = 'fps'):
    """
    Point cloud sampling function consistent with training
    point_cloud: (N, D) where D can be 6 (xyz+rgb), 8 (xyz+rgb+uv), or 3 (xyz)
    """
    if num_points == 'all':  # use all points
        return point_cloud

    if point_cloud.shape[0] <= num_points:
        # pad with zeros
        point_cloud_dim = point_cloud.shape[-1]
        point_cloud = np.concatenate([point_cloud, np.zeros((num_points - point_cloud.shape[0], point_cloud_dim))], axis=0)
        return point_cloud

    if method == 'uniform':
        # uniform sampling
        sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[sampled_indices]
    elif method == 'fps' and HAS_TORCH3D:
        # fast point cloud sampling using torch3d (consistent with training)
        point_cloud_tensor = torch.from_numpy(point_cloud).unsqueeze(0).cuda()
        num_points_tensor = torch.tensor([num_points]).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=point_cloud_tensor[..., :3], K=num_points_tensor)
        point_cloud = point_cloud_tensor.squeeze(0).cpu().numpy()
        point_cloud = point_cloud[sampled_indices.squeeze(0).cpu().numpy()]
    else:
        # fallback to random sampling if torch3d not available
        sampled_indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        point_cloud = point_cloud[sampled_indices]

    return point_cloud

def build_attn_from_mask(point_cloud, mask_json, img_res=(84, 84), n_points=512, n_channels=3):
    """
    精确版本：使用 UV 坐标查询 mask，精确标注哪些点在 mask 内。
    point_cloud: (N_pc, 8) xyzrgbuv，其中 uv 是归一化的 [0, 1]
    mask_json: json dict from grounded_sam2
    img_res: (H, W) of original image
    """
    H, W = img_res
    
    # Sample or pad point cloud using FPS (consistent with training)
    pc = point_cloud_sampling(point_cloud, n_points, method='fps')
    
    xyz = pc[:, :3]
    
    # Extract UV coordinates (normalized [0, 1])
    if pc.shape[1] >= 8:
        u_norm = pc[:, 6]  # normalized u
        v_norm = pc[:, 7]  # normalized v
    else:
        # Fallback: if no UV, use simple method
        u_norm = np.zeros(pc.shape[0], dtype=np.float32)
        v_norm = np.zeros(pc.shape[0], dtype=np.float32)
    
    # Convert normalized UV to pixel coordinates
    u_pix = np.clip((u_norm * W).round().astype(int), 0, W - 1)
    v_pix = np.clip((v_norm * H).round().astype(int), 0, H - 1)
    
    # Initialize attention field
    attn = np.zeros((n_channels, n_points), dtype=np.float32)
    
    # Check if points are inside any mask
    mask_hit = np.zeros(n_points, dtype=bool)
    
    if mask_json and "annotations" in mask_json and len(mask_json["annotations"]) > 0:
        # Decode all masks and check which points are inside
        for ann in mask_json["annotations"]:
            rle = ann["segmentation"]
            if isinstance(rle, dict) and "counts" in rle:
                try:
                    # Decode RLE mask
                    mask = mask_util.decode(rle).astype(bool)  # (H, W)
                    # Check which points are inside this mask
                    mask_hit |= mask[v_pix, u_pix]
                except Exception as e:
                    print(f"[warn] Failed to decode mask: {e}")
                    continue
    
    # Channel 0: Binary mask hit (1 if inside any mask, 0 otherwise)
    attn[0] = mask_hit.astype(np.float32)
    
    # Channel 1: Distance-based attention (closer to center = higher weight)
    # Use normalized x coordinate as proxy for distance
    x_center = xyz[:, 0].mean()
    x_dist = np.abs(xyz[:, 0] - x_center)
    x_dist_norm = x_dist / (x_dist.max() + 1e-6)
    attn[1] = (1.0 - x_dist_norm) * mask_hit.astype(np.float32)  # Only for points in mask
    
    # Channel 2: Inverse distance (for obstacle/background attention)
    attn[2] = (1.0 - mask_hit.astype(np.float32))  # Points NOT in mask
    
    # NOTE: Channel 3 (normalized x coordinate) removed — only channels 0-2 are kept.
    
    return attn

def build_attn_from_env_seg(point_cloud, seg_data, n_points=512, n_channels=3):
    """
    从环境直接提供的分割数据构建attention。
    seg_data: (H, W, 2) - MuJoCo segmentation data, where each pixel contains [objtype, objid]
    point_cloud: (N_pc, 8) xyzrgbuv，其中 uv 是归一化的 [0, 1]
    """
    H, W, _ = seg_data.shape

    # Sample or pad point cloud using FPS (consistent with training)
    pc = point_cloud_sampling(point_cloud, n_points, method='fps')

    xyz = pc[:, :3]

    # Extract UV coordinates (normalized [0, 1])
    if pc.shape[1] >= 8:
        u_norm = pc[:, 6]  # normalized u
        v_norm = pc[:, 7]  # normalized v
    else:
        # Fallback: if no UV, use simple method
        u_norm = np.zeros(pc.shape[0], dtype=np.float32)
        v_norm = np.zeros(pc.shape[0], dtype=np.float32)

    # Convert normalized UV to pixel coordinates
    u_pix = np.clip((u_norm * W).round().astype(int), 0, W - 1)
    v_pix = np.clip((v_norm * H).round().astype(int), 0, H - 1)

    # Get segmentation IDs at point locations
    seg_ids = seg_data[v_pix, u_pix, 1]  # objid

    # For adroit tasks, we prefer to use a precomputed target geom id list (by body)
    # If available, only those geom ids will be treated as targets. Otherwise,
    # fall back to treating any non-zero objid as target.
    global TARGET_GEOM_IDS  # may be set in main()
    if 'TARGET_GEOM_IDS' in globals() and TARGET_GEOM_IDS:
        # seg_ids correspond to geom ids (objid) from MuJoCo; keep only those in target list
        try:
            mask_hit = np.isin(seg_ids, np.array(TARGET_GEOM_IDS, dtype=seg_ids.dtype))
        except Exception:
            mask_hit = seg_ids > 0
    else:
        mask_hit = seg_ids > 0

    # Initialize attention field
    attn = np.zeros((n_channels, n_points), dtype=np.float32)

    # Channel 0: Binary mask hit (1 if target object, 0 otherwise)
    attn[0] = mask_hit.astype(np.float32)

    # Channel 1: Distance-based attention (closer to center = higher weight)
    # Use normalized x coordinate as proxy for distance
    x_center = xyz[:, 0].mean()
    x_dist = np.abs(xyz[:, 0] - x_center)
    x_dist_norm = x_dist / (x_dist.max() + 1e-6)
    attn[1] = (1.0 - x_dist_norm) * mask_hit.astype(np.float32)  # Only for points in mask

    # Channel 2: Inverse distance (for obstacle/background attention)
    attn[2] = (1.0 - mask_hit.astype(np.float32))  # Points NOT in mask

    return attn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_zarr", required=True, help="data/adroit_door_expert.zarr")
    ap.add_argument("--json_root", required=True, help="export_gs2/adroit_door (only used when not using env seg)")
    ap.add_argument("--output_zarr", required=True, help="data/adroit_door_expert_attn3d.zarr")
    ap.add_argument("--n_points", type=int, default=512)
    ap.add_argument("--n_channels", type=int, default=3)
    ap.add_argument("--max_episodes", type=int, default=None, help="limit episodes for quick test")
    ap.add_argument("--use_env_seg", action="store_true", help="use environment segmentation instead of GS2")
    args = ap.parse_args()

    # Try to infer task name from input_zarr filename and load corresponding targets if present
    # Expected pattern: .../adroit_<task>_expert*.zarr
    global TARGET_GEOM_IDS
    TARGET_GEOM_IDS = None
    try:
        basename = os.path.basename(args.input_zarr)
        m = re.search(r"adroit_(?P<task>[a-zA-Z0-9_]+)_expert", basename)
        if m:
            task_name = m.group("task")
            targets_path = os.path.join("targets", f"{task_name}_geom_ids.json")
            if os.path.exists(targets_path):
                try:
                    with open(targets_path, "r") as f:
                        TARGET_GEOM_IDS = json.load(f)
                    print(f"[info] Loaded target geom ids for task '{task_name}' from {targets_path} ({len(TARGET_GEOM_IDS)} ids)")
                except Exception as e:
                    print(f"[warn] Failed to load targets {targets_path}: {e}")
    except Exception:
        pass

    # 读原始 zarr
    keys = ["state", "action", "point_cloud", "img"]
    if args.use_env_seg:
        keys.append("segmentation")
    rb = ReplayBuffer.copy_from_path(
        args.input_zarr,
        keys=keys,
    )
    n_eps = rb.n_episodes if args.max_episodes is None else min(args.max_episodes, rb.n_episodes)

    # 创建输出 zarr
    store = zarr.DirectoryStore(args.output_zarr)
    root = zarr.group(store=store, overwrite=True)
    data_g = root.create_group("data")
    meta_g = root.create_group("meta")
    
    # 拷贝 meta（episode_ends），使用和原始数据相同的配置
    # meta数据通常整个数组作为一个chunk，不使用compressor
    episode_ends_arr = rb.episode_ends
    if isinstance(episode_ends_arr, zarr.Array):
        episode_ends_data = episode_ends_arr[:]
        episode_ends_chunks = episode_ends_arr.shape
    else:
        episode_ends_data = np.array(episode_ends_arr)
        episode_ends_chunks = episode_ends_data.shape
    meta_g.create_dataset("episode_ends", data=episode_ends_data,
                         shape=episode_ends_chunks, chunks=episode_ends_chunks,
                         dtype=episode_ends_data.dtype, compressor=None, overwrite=True)

    # 获取原始数据的chunks和compressor配置
    def get_chunks_and_compressor(key):
        arr = rb[key]
        if isinstance(arr, zarr.Array):
            return arr.chunks, arr.compressor
        else:
            # 如果是numpy数组，使用get_optimal_chunks计算
            chunks = get_optimal_chunks(shape=arr.shape, dtype=arr.dtype)
            compressor = ReplayBuffer.resolve_compressor('default')
            return chunks, compressor

    # 预创建 datasets，使用和原始数据相同的chunks和compressor
    state_chunks, state_compressor = get_chunks_and_compressor("state")
    action_chunks, action_compressor = get_chunks_and_compressor("action")
    pc_chunks, pc_compressor = get_chunks_and_compressor("point_cloud")
    img_chunks, img_compressor = get_chunks_and_compressor("img")
    
    data_g.create_dataset("state", shape=rb["state"].shape, dtype=rb["state"].dtype,
                         chunks=state_chunks, compressor=state_compressor, overwrite=True)
    data_g.create_dataset("action", shape=rb["action"].shape, dtype=rb["action"].dtype,
                         chunks=action_chunks, compressor=action_compressor, overwrite=True)
    data_g.create_dataset("point_cloud", shape=rb["point_cloud"].shape, dtype=rb["point_cloud"].dtype,
                         chunks=pc_chunks, compressor=pc_compressor, overwrite=True)
    data_g.create_dataset("img", shape=rb["img"].shape, dtype=rb["img"].dtype,
                         chunks=img_chunks, compressor=img_compressor, overwrite=True)
    
    # 新增 attn_3d: shape (T_total, n_channels, n_points)
    # 使用get_optimal_chunks计算合适的chunk大小
    T_total = rb["state"].shape[0]
    attn_shape = (T_total, args.n_channels, args.n_points)
    attn_chunks = get_optimal_chunks(shape=attn_shape, dtype=np.float32)
    attn_compressor = ReplayBuffer.resolve_compressor('default')
    data_g.create_dataset("attn_3d", shape=attn_shape, dtype=np.float32,
                         chunks=attn_chunks, compressor=attn_compressor, overwrite=True)

    # 写数据
    step_cursor = 0
    for ep_idx in range(n_eps):
        ep = rb.get_episode(ep_idx)
        T = ep["state"].shape[0]
        pc = ep["point_cloud"]  # (T, Npc, 6) or (T, Npc, 8) if has uv
        imgs = ep["img"]
        
        # Check if point cloud has UV coordinates
        has_uv = pc.shape[-1] >= 8
        if not has_uv:
            print(f"[warn] Episode {ep_idx}: point_cloud shape is {pc.shape}, expected at least 8 channels (xyzrgbuv). "
                  f"Falling back to simple attention generation.")
        # 拷贝原有字段
        data_g["state"][step_cursor:step_cursor+T] = ep["state"]
        data_g["action"][step_cursor:step_cursor+T] = ep["action"]
        data_g["point_cloud"][step_cursor:step_cursor+T] = pc
        data_g["img"][step_cursor:step_cursor+T] = imgs

        # 为每帧生成 attn_3d
        for t in range(T):
            if args.use_env_seg:
                # 使用环境分割数据
                seg_data = ep["segmentation"][t]  # (H, W, 2)
                attn = build_attn_from_env_seg(pc[t], seg_data,
                                             n_points=args.n_points, n_channels=args.n_channels)
            else:
                # 使用GS2 JSON数据
                json_path = os.path.join(args.json_root, f"ep_{ep_idx:04d}", f"frame_{t:04d}.json")
                mask_json = None
                if os.path.exists(json_path):
                    try:
                        mask_json = load_json(json_path)
                    except Exception as e:
                        print(f"[warn] fail to load {json_path}: {e}")
                attn = build_attn_from_mask(pc[t], mask_json, img_res=imgs[t].shape[:2],
                                            n_points=args.n_points, n_channels=args.n_channels)
            data_g["attn_3d"][step_cursor + t] = attn
        step_cursor += T
        print(f"[done] ep {ep_idx}, attn_3d filled.")
    print(f"Saved to {args.output_zarr}")

if __name__ == "__main__":
    main()