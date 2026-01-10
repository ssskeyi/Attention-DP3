import wandb
import numpy as np
import torch
import collections
import tqdm
import os
import subprocess
import tempfile
import json
import imageio
import pycocotools.mask as mask_util
import base64
from io import BytesIO
import time
import re
from termcolor import cprint
from diffusion_policy_3d.env import DexArtEnv
from diffusion_policy_3d.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy_3d.gym_util.video_recording_wrapper import SimpleVideoRecordingWrapper

from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
import diffusion_policy_3d.common.logger_util as logger_util


class DexArtRunner(BaseRunner):
    def __init__(self,
                 output_dir,
                 n_train=10,
                 max_steps=250,
                 n_obs_steps=8,
                 n_action_steps=8,
                 fps=10,
                 crf=22,
                 tqdm_interval_sec=5.0,
                 task_name=None,
                 ):
        super().__init__(output_dir)
        self.task_name = task_name

        steps_per_render = max(10 // fps, 1)

        def env_fn(is_test=True):
            return MultiStepWrapper(
                SimpleVideoRecordingWrapper(DexArtEnv(
                    task_name=task_name,
                    use_test_set=is_test,
                )),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
                reward_agg_method='sum',
            )

        self.env_train = env_fn(is_test=False)
        self.episode_train = n_train

        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        self.logger_util_train = logger_util.LargestKRecorder(K=3)
        self.logger_util_train10 = logger_util.LargestKRecorder(K=5)

        # Grounded-SAM-2 config (for realtime attn generation)
        self.gs2_api_url = os.getenv("GS2_API_URL", "http://127.0.0.1:5000")
        self.gs2_verbose = os.getenv("GS2_VERBOSE", "0").lower() in ("1", "true", "yes")
        self.gs2_conda_env = "aedp3_vis"
        self.gs2_text_prompt = self._get_text_prompt_for_task(task_name)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.gs2_script_path = os.path.join(project_root, "Grounded-SAM-2", "infer_grounded_sam2_single.py")
        self.gs2_sam2_ckpt = "checkpoints/sam2.1_hiera_large.pt"
        self.gs2_sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        self.gs2_gdino_cfg = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        self.gs2_gdino_ckpt = "gdino_checkpoints/groundingdino_swint_ogc.pth"
        self.gs2_device = "cuda"
        self.gs2_box_thr = 0.35
        self.gs2_text_thr = 0.25
        temp_dir = os.path.join(project_root, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        self.temp_dir = temp_dir

    def _get_text_prompt_for_task(self, task_name):
        """Get text prompt for Grounded-SAM-2 based on task name."""
        prompts = {
            'bucket': 'bucket. water bucket.',
            'faucet': 'faucet. water faucet.',
            'laptop': 'laptop. laptop computer.',
            'toilet': 'toilet. toilet bowl.'
        }
        return prompts.get(task_name, task_name)

    def _build_attn_from_mask(self, point_cloud_with_uv, mask_json, img_res=(84, 84), n_points=1024, n_channels=3):
        """Build attention from segmentation mask (DexArt version with 1024 points)."""
        """Build attention from segmentation mask (DexArt version with 1024 points)."""
        H, W = img_res
        xyz = point_cloud_with_uv[:, :3]  # (N, 3)
        uv = point_cloud_with_uv[:, 3:5]  # (N, 2)

        # normalize uv coordinates to [0, 1]
        u_norm = (uv[:, 0] - uv[:, 0].min()) / (uv[:, 0].max() - uv[:, 0].min() + 1e-6)
        v_norm = (uv[:, 1] - uv[:, 1].min()) / (uv[:, 1].max() - uv[:, 1].min() + 1e-6)

        # convert to pixel coordinates
        u_pix = np.clip((u_norm * W).round().astype(int), 0, W - 1)
        v_pix = np.clip((v_norm * H).round().astype(int), 0, H - 1)

        attn = np.zeros((n_channels, n_points), dtype=np.float32)
        mask_hit = np.zeros(n_points, dtype=bool)

        if mask_json and "annotations" in mask_json and len(mask_json["annotations"]) > 0:
            for ann in mask_json["annotations"]:
                rle = ann["segmentation"]
                if isinstance(rle, dict) and "counts" in rle:
                    try:
                        mask = mask_util.decode(rle).astype(bool)
                        mask_hit |= mask[v_pix, u_pix]
                    except Exception:
                        continue

        attn[0] = mask_hit.astype(np.float32)
        x_center = xyz[:, 0].mean()
        x_dist = np.abs(xyz[:, 0] - x_center)
        x_dist_norm = x_dist / (x_dist.max() + 1e-6)
        attn[1] = (1.0 - x_dist_norm) * mask_hit.astype(np.float32)
        attn[2] = (1.0 - mask_hit.astype(np.float32))
        return attn

    def _generate_attn_3d_inference(self, rgb_img, point_cloud_with_uv, img_res=(84, 84)):
        """Generate attn_3d during inference by calling Grounded-SAM-2 (API preferred)."""
        try:
            import requests
            HAS_REQUESTS = True
        except Exception:
            HAS_REQUESTS = False

        if self.gs2_api_url and HAS_REQUESTS:
            try:
                from PIL import Image
                img_pil = Image.fromarray(rgb_img)
                img_bytes = BytesIO()
                img_pil.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                img_base64 = base64.b64encode(img_bytes.read()).decode("utf-8")

                api_url = f"{self.gs2_api_url}/infer"
                payload = {
                    "image_base64": img_base64,
                    "text": self.gs2_text_prompt,
                    "box_thr": self.gs2_box_thr,
                    "text_thr": self.gs2_text_thr,
                }
                if self.gs2_verbose:
                    cprint(f"[GM2-API] Calling Grounded-SAM-2 API...", "cyan")
                response = requests.post(api_url, json=payload, timeout=60)
                if response.status_code != 200:
                    return np.zeros((3, 1024), dtype=np.float32)
                mask_json = response.json()
                attn_3d = self._build_attn_from_mask(point_cloud_with_uv, mask_json, img_res=img_res, n_points=1024, n_channels=3)
                return attn_3d
            except Exception:
                return self._generate_attn_3d_via_subprocess(rgb_img, point_cloud_with_uv, img_res)
        else:
            return self._generate_attn_3d_via_subprocess(rgb_img, point_cloud_with_uv, img_res)

    def _generate_attn_3d_via_subprocess(self, rgb_img, point_cloud_with_uv, img_res=(84,84)):
        try:
            temp_img_path = os.path.join(self.temp_dir, f"temp_img_{os.getpid()}_{np.random.randint(0,1000000)}.png")
            temp_json_path = temp_img_path.replace(".png", ".json")
            imageio.imwrite(temp_img_path, rgb_img)
            script_path = os.path.abspath(self.gs2_script_path)
            gs2_root = os.path.dirname(script_path)
            if not os.path.exists(gs2_root):
                return np.zeros((3,1024), dtype=np.float32)
            cmd = [
                "conda", "run", "-n", self.gs2_conda_env,
                "python", script_path,
                "--img_path", temp_img_path,
                "--output_json", temp_json_path,
                "--text", self.gs2_text_prompt,
                "--sam2_ckpt", self.gs2_sam2_ckpt,
                "--sam2_cfg", self.gs2_sam2_cfg,
                "--gdino_cfg", self.gs2_gdino_cfg,
                "--gdino_ckpt", self.gs2_gdino_ckpt,
                "--device", self.gs2_device,
                "--box_thr", str(self.gs2_box_thr),
                "--text_thr", str(self.gs2_text_thr),
            ]
            process = subprocess.Popen(cmd, cwd=gs2_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            stdout_lines = []
            timeout_seconds = 60
            start_time = time.time()
            warning_patterns = [r"UserWarning:", r"Triggered internally", r"Falling back to", r"Memory efficient"]
            def should_print_line(line):
                if not line.strip():
                    return False
                for pattern in warning_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        return False
                return True
            try:
                for line in process.stdout:
                    if time.time() - start_time > timeout_seconds:
                        process.terminate()
                        break
                    if self.gs2_verbose and should_print_line(line):
                        print(f"[GS2] {line.strip()}")
                    stdout_lines.append(line)
                process.wait(timeout=10)
                if process.returncode == 0 and os.path.exists(temp_json_path):
                    with open(temp_json_path, 'r') as f:
                        mask_json = json.load(f)
                    attn_3d = self._build_attn_from_mask(point_cloud_with_uv, mask_json, img_res=img_res, n_points=1024, n_channels=3)
                else:
                    attn_3d = np.zeros((3,1024), dtype=np.float32)
            except subprocess.TimeoutExpired:
                process.kill()
                attn_3d = np.zeros((3,1024), dtype=np.float32)
            except Exception:
                attn_3d = np.zeros((3,1024), dtype=np.float32)
            finally:
                # cleanup temp files
                for f in [temp_img_path, temp_json_path]:
                    try:
                        if os.path.exists(f):
                            os.unlink(f)
                    except Exception:
                        pass
            return attn_3d

        
    def run(self, policy: BasePolicy):
        device = policy.device
        dtype = policy.dtype
        env_train = self.env_train

        all_returns_train = []
        all_success_rates_train = []


        ##############################
        # train env loop
        for episode_id in tqdm.tqdm(range(self.episode_train), desc=f"DexArt {self.task_name} Train Env",leave=False, mininterval=self.tqdm_interval_sec):
            # start rollout
            obs = env_train.reset()

            policy.reset()

            done = False
            reward_sum = 0.
            for step_id in range(self.max_steps):
                # create obs dict
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=device))

                # Check if policy expects attn_3d
                needs_attn_3d = False
                if hasattr(policy, 'shape_meta') and 'attn_3d' in policy.shape_meta.get('obs', {}):
                    needs_attn_3d = True
                elif hasattr(policy, 'obs_encoder') and hasattr(policy.obs_encoder, 'use_attn_3d') and policy.obs_encoder.use_attn_3d:
                    needs_attn_3d = True

                # run policy
                with torch.no_grad():
                    # add batch dim to match. (1,2,3,84,84)
                    # and multiply by 255, align with all envs
                    obs_dict_input = {}  # flush unused keys
                    obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                    obs_dict_input['imagin_robot'] = obs_dict['imagin_robot'].unsqueeze(0)
                    obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)

                    if needs_attn_3d:
                        # If environment already provides attn_3d in obs, use it
                        if 'attn_3d' in np_obs_dict:
                            attn_3d = np_obs_dict['attn_3d']  # (T, C, N) or (C, N)
                            if attn_3d.ndim == 2:
                                attn_3d = np.expand_dims(attn_3d, 0)  # (1, C, N)
                                attn_3d = np.expand_dims(attn_3d, 0)  # (1, 1, C, N)
                            elif attn_3d.ndim == 3:
                                attn_3d = np.expand_dims(attn_3d, 0)  # (1, T, C, N)
                            obs_dict_input['attn_3d'] = torch.from_numpy(attn_3d).to(device=device, dtype=dtype)
                        else:
                            # Generate attn_3d on the fly using Grounded-SAM-2 (API preferred)
                            rgb_img = None
                            try:
                                # Try to render rgb image from env
                                base_env = env_train.env
                                while hasattr(base_env, 'env'):
                                    base_env = base_env.env
                                rgb_img = base_env.render('rgb_array')
                            except Exception:
                                rgb_img = None

                            # Try to get point cloud with UV coordinates
                            point_cloud_full = None
                            try:
                                # Use DexArt's point cloud generator to get point cloud with UV coordinates
                                base_env = env_train.env
                                while hasattr(base_env, 'env'):
                                    base_env = base_env.env

                                if hasattr(base_env, 'pc_generator'):
                                    # Generate point cloud with UV coordinates using the generator
                                    pc_with_uv = base_env.pc_generator.generate_point_cloud_with_uv(rgb_img)
                                    point_cloud_full = np.stack([pc_with_uv] * self.n_obs_steps, axis=0)
                                else:
                                    raise RuntimeError("DexArt environment does not have point cloud generator initialized.")
                            except Exception as e:
                                raise RuntimeError(f"Failed to generate point cloud with UV coordinates for attn_3d generation: {e}")

                            # Generate attn_3d for timesteps if possible
                            if rgb_img is not None and point_cloud_full is not None:
                                T = point_cloud_full.shape[0]
                                attn_3d_list = []
                                for t in range(T):
                                    pc_t = point_cloud_full[t]
                                    if pc_t.shape[-1] < 8:
                                        attn_3d_t = np.zeros((3, 1024), dtype=np.float32)
                                    else:
                                        attn_3d_t = self._generate_attn_3d_inference(rgb_img, pc_t, img_res=rgb_img.shape[:2])
                                    attn_3d_list.append(attn_3d_t)
                                attn_3d = np.stack(attn_3d_list, axis=0)  # (T, C, N)
                            else:
                                raise RuntimeError(
                                    "Cannot generate attn_3d at runtime: RGB image or point cloud with UV coordinates not available. "
                                    "Please use pre-computed attn_3d datasets."
                                )
                            obs_dict_input['attn_3d'] = torch.from_numpy(attn_3d).to(device=device, dtype=dtype).unsqueeze(0)

                    action_dict = policy.predict_action(obs_dict_input)


                # device_transfer
                np_action_dict = dict_apply(action_dict,
                                            lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action'].squeeze(0)

                # step env
                obs, reward, done, info = env_train.step(action)
                reward_sum += reward
                done = np.all(done)

                if done:
                    break

            all_returns_train.append(reward_sum)
            all_success_rates_train.append(env_train.is_success())

       

        SR_mean_train = np.mean(all_success_rates_train)
        returns_mean_train = np.mean(all_returns_train)

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        log_data
        log_data['mean_success_rates_train'] = SR_mean_train
        log_data['mean_returns_train'] = returns_mean_train

        log_data['test_mean_score'] = SR_mean_train

        self.logger_util_train.record(SR_mean_train)
        self.logger_util_train10.record(SR_mean_train)

        log_data['SR_train_L3'] = self.logger_util_train.average_of_largest_K()
        log_data['SR_train_L5'] = self.logger_util_train10.average_of_largest_K()
        

        cprint( f"Mean SR train: {SR_mean_train:.3f}", 'green')

        # visualize sim
        videos_train = env_train.env.get_video()

        if len(videos_train.shape) == 5:
            videos_train = videos_train[:, 0]
        sim_video_train = wandb.Video(videos_train, fps=self.fps, format="mp4")
        log_data[f'sim_video_train'] = sim_video_train

        # clear out video buffer
        _ = env_train.reset()
        videos_train = None
        del env_train

        return log_data
