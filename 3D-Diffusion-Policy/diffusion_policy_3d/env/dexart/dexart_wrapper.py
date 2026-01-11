import gym
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import math
import open3d as o3d
from PIL import Image as PIL_Image

from termcolor import cprint
from dexart.env.task_setting import TRAIN_CONFIG, RANDOM_CONFIG
from dexart.env.create_env import create_env
from gym import spaces


def downsample_with_fps(points: np.ndarray, num_points: int = 1024):
    # fast point cloud sampling using torch3d
    points = torch.from_numpy(points).unsqueeze(0).cuda()
    num_points = torch.tensor([num_points]).cuda()
    # remember to only use coord to sample
    _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=num_points)
    points = points.squeeze(0).cpu().numpy()
    points = points[sampled_indices.squeeze(0).cpu().numpy()]
    return points


class DexArtEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, use_test_set=False, num_points=1024):
        if use_test_set:
            indeces = TRAIN_CONFIG[task_name]['unseen']
            cprint(f"using unseen instances {indeces}", 'yellow')
        else:
            indeces = TRAIN_CONFIG[task_name]['seen']
            cprint(f"using seen instances {indeces}", 'yellow')

        rand_pos = RANDOM_CONFIG[task_name]['rand_pos']
        rand_degree = RANDOM_CONFIG[task_name]['rand_degree']

        self.env = create_env(task_name=task_name,
                              use_visual_obs=True,
                              use_gui=False,
                              is_eval=True,
                              pc_noise=True,
                              pc_seg=True,
                              index=indeces,
                              img_type='robot',
                              rand_pos=rand_pos,
                              rand_degree=rand_degree)

        robot_dof = self.env.robot.dof

        self.obs_sensor_dim = 32
        self.num_points = num_points

        # Initialize point cloud generator for GS2 attention generation
        self.pc_generator = DexArtPointCloudGenerator(self, img_size=84, num_points=self.num_points)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(robot_dof,),
            dtype=np.float32
        )
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3, 84, 84),
                dtype=np.float32
            ),
            
            'depth': spaces.Box(
                low=0,
                high=1,
                shape=(84, 84),
                dtype=np.float32
            ),
            
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'imagin_robot': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(96, 7),
                dtype=np.float32
            ),

        })

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_pixels = obs['instance_1-rgb']  # (84, 84, 3)
        obs_depth = obs['instance_1-depth']  # (84, 84)
        obs_sensor = obs['state']  # (32,)
        obs_pointcloud = obs['instance_1-point_cloud']  # (1024, 3)
        if obs_pointcloud.shape[0] > self.num_points:
            obs_pointcloud = downsample_with_fps(
                obs_pointcloud, self.num_points)
        obs_imagin_robot = obs['imagination_robot']  # (96, 7)

        if obs_pixels.shape[0] != 3:  # make channel first
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': obs_depth,
            'agent_pos': obs_sensor,
            'point_cloud': obs_pointcloud,
            'imagin_robot': obs_imagin_robot
        }
        return obs_dict, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs_pixels = obs['instance_1-rgb']  # (84, 84, 3)
        obs_sensor = obs['state']  # (32,)
        obs_pointcloud = obs['instance_1-point_cloud']  # (1024, 3)
        obs_depth = obs['instance_1-depth']  # (84, 84)
        if obs_pointcloud.shape[0] > self.num_points:
            obs_pointcloud = downsample_with_fps(
                obs_pointcloud, self.num_points)
        obs_imagin_robot = obs['imagination_robot']  # (96, 7)

        if obs_pixels.shape[0] != 3:  # make channel first
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': obs_depth,
            'agent_pos': obs_sensor,
            'point_cloud': obs_pointcloud,
            'imagin_robot': obs_imagin_robot
        }
        return obs_dict

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def get_visual_observation(self):
        return self.env.get_visual_observation()

    def render(self, mode='rgb_array'):
        visual_obs = self.get_visual_observation()
        img = visual_obs['instance_1-rgb']  # (84,84,3), [0,1]
        # to uint8
        img = (img*255).astype(np.uint8)
        return img

    def close(self):
        pass

    def horizon(self):
        return self.env.horizon()

    def is_success(self):
        return self.env.is_eval_done


class DexArtPointCloudGenerator:
    """
    Point cloud generator for DexArt environments that produces point clouds with UV coordinates.

    This is needed for Grounded-SAM-2 attention generation during inference.
    """

    def __init__(self, env, img_size=84, fov=75, num_points=1024):
        """
        Initialize the point cloud generator.

        Args:
            env: DexArt environment instance
            img_size: Size of the rendered image (assumed square)
            fov: Field of view in degrees
            num_points: Number of points to generate
        """
        self.env = env
        self.img_size = img_size
        self.fov = fov
        self.num_points = num_points

        # Calculate camera intrinsic matrix
        fovy_rad = math.radians(fov)
        f = self.img_size / (2 * math.tan(fovy_rad / 2))
        self.camera_matrix = np.array([
            [f, 0, self.img_size / 2],
            [0, f, self.img_size / 2],
            [0, 0, 1]
        ])

    def generate_point_cloud_with_uv(self, rgb_img=None, depth_img=None):
        """
        Generate point cloud with UV coordinates using DexArt's native 3D data.

        DexArt directly provides 3D point cloud data from GPU rendering,
        no need to reconstruct from RGB-D like MetaWorld.

        Args:
            rgb_img: RGB image (H, W, 3), if None will render from environment
            depth_img: Depth image (H, W), if None will try to get from environment

        Returns:
            point_cloud: (N, 8) array with [x, y, z, r, g, b, u, v]
        """
        # Get visual observations from DexArt environment
        visual_obs = self.env.get_visual_observation()

        # Debug: check what keys are available in visual observation
        # print(f"[DEBUG] Available visual obs keys: {list(visual_obs.keys())}")
        # for key, value in visual_obs.items():
        #     if isinstance(value, np.ndarray):
        #         print(f"[DEBUG] {key}: shape {value.shape}, dtype {value.dtype}, range [{value.min():.3f}, {value.max():.3f}]")
        #     else:
        #         print(f"[DEBUG] {key}: {type(value)}")

        # Get depth image for 3D reconstruction (similar to MetaWorld approach)
        depth_img = visual_obs['instance_1-depth']  # (84, 84) or (2, 84, 84)

        # Handle multi-channel depth
        if depth_img.ndim == 3 and depth_img.shape[0] == 2:
            depth_img = depth_img[0]  # Take first channel

        # print(f"[DEBUG] Using depth-based reconstruction instead of DexArt point cloud")
        # print(f"[DEBUG] Depth shape: {depth_img.shape}, range: [{depth_img.min():.3f}, {depth_img.max():.3f}]")

        # Reconstruct 3D points from depth image (in camera coordinates)
        # This follows the standard camera projection model
        height, width = depth_img.shape
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

        # Create pixel coordinate grids
        u_grid, v_grid = np.meshgrid(np.arange(width), np.arange(height))
        u_grid = u_grid.astype(np.float32)
        v_grid = v_grid.astype(np.float32)

        # Back-project to 3D camera coordinates
        # Standard pinhole camera model: X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
        Z = depth_img.astype(np.float32)
        X = (u_grid - cx) * Z / fx
        Y = (v_grid - cy) * Z / fy

        # Flatten to get all points
        points_3d = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

        # Filter out invalid points (zero or negative depth)
        valid_depth = Z.flatten() > 0
        points_3d = points_3d[valid_depth]
        u_coords_flat = u_grid.flatten()[valid_depth]
        v_coords_flat = v_grid.flatten()[valid_depth]

        # Downsample to target number of points (1024)
        if len(points_3d) > self.num_points:
            indices = np.random.choice(len(points_3d), self.num_points, replace=False)
            points_3d = points_3d[indices]
            u_coords_flat = u_coords_flat[indices]
            v_coords_flat = v_coords_flat[indices]

        # print(f"[DEBUG] Reconstructed {len(points_3d)} points from depth image")

        # Get RGB image for color information
        if rgb_img is None:
            rgb_img = visual_obs['instance_1-rgb']  # (84, 84, 3) or (3, 84, 84)
            if rgb_img.shape[0] == 3:  # CHW format
                rgb_img = rgb_img.transpose(1, 2, 0)  # Convert to HWC

        # Convert RGB to float and normalize to [0, 1]
        rgb_img_float = rgb_img.astype(np.float32) / 255.0
        # print(f"[DEBUG] RGB image shape: {rgb_img_float.shape}, dtype: {rgb_img_float.dtype}")

        # UV coordinates are directly available from depth reconstruction
        # Since we reconstructed from depth, UV coords are just normalized pixel coordinates
        u_norm = u_coords_flat / self.img_size
        v_norm = v_coords_flat / self.img_size

        # All reconstructed points should be valid (we filtered zero depth already)
        valid_mask = np.ones(len(points_3d), dtype=bool)

        # print(f"[DEBUG] UV coordinates from depth reconstruction:")
        # print(f"[DEBUG] u_norm range: [{u_norm.min():.3f}, {u_norm.max():.3f}]")
        # print(f"[DEBUG] v_norm range: [{v_norm.min():.3f}, {v_norm.max():.3f}]")
        # print(f"[DEBUG] All points valid: {valid_mask.sum()}/{len(valid_mask)}")

        # For points within image bounds, sample colors from RGB image
        colors = np.zeros((len(points_3d), 3), dtype=np.float32)

        # Convert normalized UV to pixel coordinates for sampling
        u_pixel = np.clip((u_norm * self.img_size).astype(int), 0, self.img_size - 1)
        v_pixel = np.clip((v_norm * self.img_size).astype(int), 0, self.img_size - 1)

        # Sample colors from RGB image
        colors = rgb_img_float[v_pixel, u_pixel]  # (N, 3)

        # For invalid projections, set default color
        colors[~valid_mask] = np.array([0.5, 0.5, 0.5])  # Gray for invalid points

        # print(f"[DEBUG] UV range: u[{u_norm.min():.3f}, {u_norm.max():.3f}], v[{v_norm.min():.3f}, {v_norm.max():.3f}]")
        # print(f"[DEBUG] Valid projections: {valid_mask.sum()}/{len(valid_mask)}")

        # Combine into final point cloud (N, 8): [x, y, z, r, g, b, u, v]
        point_cloud = np.column_stack([
            points_3d,      # x, y, z (3D coordinates)
            colors,         # r, g, b (colors)
            u_norm,         # u (normalized UV)
            v_norm          # v (normalized UV)
        ])

        # print(f"[DEBUG] Final point cloud shape: {point_cloud.shape}")

        return point_cloud
