#!/usr/bin/env python3
"""
Inspect environment segmentation returned by Adroit env.

Usage:
  PYTHONPATH=third_party/VRL3/src:3D-Diffusion-Policy/diffusion_policy_3d python tools/inspect_env_seg.py

This script will create a `debug_seg/` directory and save per-frame
objtype and objid images for the reset frame and several steps.
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image

ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "third_party/VRL3/src"))
sys.path.insert(0, os.path.join(ROOT, "3D-Diffusion-Policy/diffusion_policy_3d"))

from adroit import AdroitEnv
from diffusion_policy_3d.gym_util.mjpc_wrapper import MujocoPointcloudWrapperAdroit


def save_seg(seg, out_path_prefix, t):
    if seg is None:
        print(f"[{t}] segmentation: None")
        return
    seg = np.array(seg)
    if seg.ndim != 3 or seg.shape[2] != 2:
        print(f"[{t}] unexpected seg shape {seg.shape}, skipping save")
        return
    ch0 = Image.fromarray(seg[:, :, 0].astype(np.uint8))
    ch1 = Image.fromarray(seg[:, :, 1].astype(np.uint8))
    ch0.save(f"{out_path_prefix}_t{t}_objtype.png")
    ch1.save(f"{out_path_prefix}_t{t}_objid.png")
    unique_ids = np.unique(seg[:, :, 1])
    print(f"[{t}] seg shape={seg.shape}, dtype={seg.dtype}, unique_objids(sample)={unique_ids[:10]}")


def make_env(env_short_name, use_point_crop=True, render_seg=True, device="cuda"):
    action_repeat = 2
    frame_stack = 1
    env = AdroitEnv(env_name=env_short_name + '-v0', test_image=False, num_repeats=action_repeat,
                    num_frames=frame_stack, env_feature_type='pixels',
                    device=device, reward_rescale=True, render_segmentation=render_seg)
    env = MujocoPointcloudWrapperAdroit(env=env, env_name='adroit_' + env_short_name, use_point_crop=use_point_crop)
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="door", help="env short name: door/hammer/pen")
    parser.add_argument("--out", type=str, default="debug_seg", help="output dir")
    parser.add_argument("--steps", type=int, default=5, help="number of steps to inspect")
    parser.add_argument("--device", type=str, default="cuda", help="device for env")
    args = parser.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    env = make_env(args.env, use_point_crop=True, render_seg=True, device=args.device)

    # helper to get action_space robustly from nested wrappers
    def get_action_space(environment):
        # try common attributes and nesting orders
        try:
            if hasattr(environment, "action_space"):
                return environment.action_space
        except Exception:
            pass
        try:
            env0 = getattr(environment, "env", None)
            if env0 is not None and hasattr(env0, "action_space"):
                return env0.action_space
        except Exception:
            pass
        try:
            env0 = getattr(environment, "_env", None)
            if env0 is not None and hasattr(env0, "action_space"):
                return env0.action_space
        except Exception:
            pass
        # deeper nesting
        try:
            env0 = getattr(getattr(environment, "env", None), "_env", None)
            if env0 is not None and hasattr(env0, "action_space"):
                return env0.action_space
        except Exception:
            pass
        return None

    # Reset
    reset_res = env.reset()
    # reset_res may be a NamedTuple with observation_segmentation or tuple
    seg = getattr(reset_res, "observation_segmentation", None)
    if seg is None and isinstance(reset_res, (list, tuple)) and len(reset_res) >= 3:
        seg = reset_res[2]
    save_seg(seg, os.path.join(out_dir, "reset"), 0)

    # Step some frames
    action_space = get_action_space(env)
    if action_space is None:
        print("Warning: cannot find action_space on env; steps will use zero actions if possible.")
    for t in range(1, args.steps + 1):
        if action_space is not None:
            try:
                act = action_space.sample()
            except Exception:
                # fallback zero action if shape available
                try:
                    act = np.zeros(action_space.shape, dtype=np.float32)
                except Exception:
                    act = None
        else:
            act = None
        if act is None:
            # try env.step with zero vector if possible
            try:
                # attempt to create zero action from nested action_space attributes
                aspace = get_action_space(env)
                if aspace is not None and hasattr(aspace, "shape"):
                    act = np.zeros(aspace.shape, dtype=np.float32)
                else:
                    raise RuntimeError("No usable action_space found to create zero action.")
            except Exception as e:
                raise RuntimeError(f"Cannot obtain action for env.step(): {e}")
        res = env.step(act)
        # prefer attribute access; do NOT index NamedTuple (NamedTuple may override __getitem__)
        seg = getattr(res, "observation_segmentation", None)
        save_seg(seg, os.path.join(out_dir, "step"), t)

    print("Saved segmentation images to", out_dir)


if __name__ == "__main__":
    main()


