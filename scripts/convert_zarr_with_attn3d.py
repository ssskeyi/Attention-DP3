import os
import argparse
import json
import numpy as np
import zarr
import numcodecs
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer, get_optimal_chunks

def load_json(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def build_attn_from_mask(point_cloud, mask_json, img_res=(84, 84), n_points=1600, n_channels=4):
    """
    简易版本：把落在 mask 内的点权重设为 1，否则 0。返回 shape (C, N)。
    point_cloud: (N_pc, 6) xyzrgb
    mask_json: json dict from grounded_sam2
    img_res: (H, W) of original image
    """
    if point_cloud.shape[0] >= n_points:
        idx = np.random.choice(point_cloud.shape[0], n_points, replace=False)
        pc = point_cloud[idx]
    else:
        pc = np.zeros((n_points, point_cloud.shape[1]), dtype=point_cloud.dtype)
        pc[: point_cloud.shape[0]] = point_cloud
    xyz = pc[:, :3]
    # 简化：用归一化 x 作为几何通道；mask 权重简单设为 1（如果有任何 mask），未实现精确投影
    attn = np.zeros((n_channels, n_points), dtype=np.float32)
    # 通道3：归一化 x
    attn[3] = (xyz[:, 0] - xyz[:, 0].mean()) / (xyz[:, 0].std() + 1e-6)
    # 如果有 mask，就把通道0/1/2 设为 1
    if mask_json and "annotations" in mask_json and len(mask_json["annotations"]) > 0:
        attn[0] = 1.0
        attn[1] = 1.0
        attn[2] = 1.0
    return attn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_zarr", required=True, help="data/adroit_door_expert.zarr")
    ap.add_argument("--json_root", required=True, help="export_gs2/adroit_door")
    ap.add_argument("--output_zarr", required=True, help="data/adroit_door_expert_attn3d.zarr")
    ap.add_argument("--n_points", type=int, default=1600)
    ap.add_argument("--n_channels", type=int, default=4)
    ap.add_argument("--max_episodes", type=int, default=None, help="limit episodes for quick test")
    args = ap.parse_args()

    # 读原始 zarr
    rb = ReplayBuffer.copy_from_path(
        args.input_zarr,
        keys=["state", "action", "point_cloud", "img"],
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
        pc = ep["point_cloud"]  # (T, Npc, 6)
        imgs = ep["img"]
        # 拷贝原有字段
        data_g["state"][step_cursor:step_cursor+T] = ep["state"]
        data_g["action"][step_cursor:step_cursor+T] = ep["action"]
        data_g["point_cloud"][step_cursor:step_cursor+T] = pc
        data_g["img"][step_cursor:step_cursor+T] = imgs

        # 为每帧生成 attn_3d
        for t in range(T):
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
PY