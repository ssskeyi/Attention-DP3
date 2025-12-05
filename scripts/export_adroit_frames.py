import os
import argparse
import imageio
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zarr",
        type=str,
        required=True,
        help="input zarr path, e.g. data/adroit_door_expert.zarr",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="output root dir for frames",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=5,
        help="how many episodes to export",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load replay buffer (only need img)
    rb = ReplayBuffer.copy_from_path(args.zarr, keys=["img"])
    n_eps = min(args.max_episodes, rb.n_episodes)

    for ep_idx in range(n_eps):
        ep = rb.get_episode(ep_idx)
        imgs = ep["img"]  # (T, H, W, 3), uint8
        ep_dir = os.path.join(args.out_dir, f"ep_{ep_idx:04d}")
        os.makedirs(ep_dir, exist_ok=True)
        for t, img in enumerate(imgs):
            out_path = os.path.join(ep_dir, f"frame_{t:04d}.png")
            imageio.imwrite(out_path, img)
        print(f"[done] episode {ep_idx}, frames saved to {ep_dir}")


if __name__ == "__main__":
    main()

