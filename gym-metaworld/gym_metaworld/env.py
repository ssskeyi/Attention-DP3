import random
import sys
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def _import_metaworld():
    try:
        import metaworld  # type: ignore
        return metaworld
    except Exception:
        aedp3_root = Path(__file__).resolve().parents[2]
        local_gym = aedp3_root / "third_party" / "gym-0.21.0"
        if str(local_gym) not in sys.path:
            sys.path.insert(0, str(local_gym))
        local_metaworld = aedp3_root / "third_party" / "Metaworld"
        if str(local_metaworld) not in sys.path:
            sys.path.insert(0, str(local_metaworld))
        import metaworld  # type: ignore
        return metaworld


class MetaWorldVitaEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        task_name: str = "dial-turn",
        max_episode_steps: int = 200,
        observation_width: int = 128,
        observation_height: int = 128,
        camera_name: str = "corner2",
        render_mode: str = "rgb_array",
        device_id: int = 0,
    ):
        super().__init__()

        self.metaworld = _import_metaworld()
        self.task_name = self._normalize_task_name(task_name)
        self._max_episode_steps = int(max_episode_steps)
        self.observation_width = int(observation_width)
        self.observation_height = int(observation_height)
        self.camera_name = camera_name
        self.render_mode = render_mode
        self.device_id = int(device_id)
        self._step_count = 0

        env_cls = self.metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[self.task_name]
        self.env = env_cls()
        self.env._freeze_rand_vec = False

        self.action_space = spaces.Box(
            low=self.env.action_space.low.astype(np.float32),
            high=self.env.action_space.high.astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.observation_height, self.observation_width, 3),
                    dtype=np.uint8,
                ),
                "agent_pos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(9,),
                    dtype=np.float32,
                ),
            }
        )

    @staticmethod
    def _normalize_task_name(task_name: str) -> str:
        if task_name.endswith("-goal-observable"):
            return task_name
        if task_name.endswith("-v2"):
            return f"{task_name}-goal-observable"
        return f"{task_name}-v2-goal-observable"

    def _get_pixels(self) -> np.ndarray:
        try:
            img = self.env.sim.render(
                width=self.observation_width,
                height=self.observation_height,
                camera_name=self.camera_name,
                device_id=self.device_id,
            )
        except TypeError:
            img = self.env.sim.render(
                width=self.observation_width,
                height=self.observation_height,
                camera_name=self.camera_name,
            )
        return np.asarray(img, dtype=np.uint8)

    def _get_agent_pos(self, raw_obs: Optional[np.ndarray] = None) -> np.ndarray:
        try:
            eef_pos = self.env.get_endeff_pos()
            finger_right = self.env._get_site_pos("rightEndEffector")
            finger_left = self.env._get_site_pos("leftEndEffector")
            return np.concatenate([eef_pos, finger_right, finger_left]).astype(np.float32)
        except Exception:
            if raw_obs is None:
                return np.zeros((9,), dtype=np.float32)
            raw_obs = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
            if raw_obs.shape[0] >= 9:
                return raw_obs[:9].astype(np.float32)
            out = np.zeros((9,), dtype=np.float32)
            out[: raw_obs.shape[0]] = raw_obs
            return out

    def _make_obs(self, raw_obs: Optional[np.ndarray] = None):
        return {
            "pixels": self._get_pixels(),
            "agent_pos": self._get_agent_pos(raw_obs),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            try:
                self.env.seed(seed)
            except Exception:
                pass

        self._step_count = 0
        self.env.reset()
        try:
            raw_obs = self.env.reset_model()
        except Exception:
            raw_obs = self.env.reset()

        return self._make_obs(raw_obs), {"is_success": False}

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self._step_count += 1

        success = bool(info.get("success", False))
        terminated = success or bool(done)
        truncated = self._step_count >= self._max_episode_steps

        return self._make_obs(raw_obs), float(reward), terminated, truncated, {"is_success": success}

    def render(self):
        return self._get_pixels()

    def close(self):
        return None
