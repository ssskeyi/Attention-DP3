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

        # Initialize point cloud generator for GS2 attention generation
        self.pc_generator = DexArtPointCloudGenerator(self, img_size=84)
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(robot_dof,),
            dtype=np.float32
        )
        self.obs_sensor_dim = 32
        self.num_points = num_points
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

    def __init__(self, env, img_size=84, fov=75):
        """
        Initialize the point cloud generator.

        Args:
            env: DexArt environment instance
            img_size: Size of the rendered image (assumed square)
            fov: Field of view in degrees
        """
        self.env = env
        self.img_size = img_size
        self.fov = fov

        # Calculate camera intrinsic matrix
        fovy_rad = math.radians(fov)
        f = self.img_size / (2 * math.tan(fovy_rad / 2))
        self.camera_matrix = np.array([
            [f, 0, self.img_size / 2],
            [0, f, self.img_size / 2],
            [0, 0, 1]
        ])

    def generate_point_cloud_with_uv(self, rgb_img=None):
        """
        Generate point cloud with UV coordinates from RGB-D data.

        Args:
            rgb_img: RGB image (H, W, 3), if None will render from environment

        Returns:
            point_cloud: (N, 8) array with [x, y, z, r, g, b, u, v]
        """
        if rgb_img is None:
            rgb_img = self.env.render('rgb_array')  # (H, W, 3), uint8, [0, 255]

        # Get depth from environment if available
        try:
            depth_obs = self.env.get_visual_observation()
            depth_img = depth_obs.get('instance_1-depth', None)
            if depth_img is None:
                # If no depth available, create synthetic depth for testing
                # This is a fallback - in practice, DexArt should provide depth
                cprint("Warning: No depth image available, using synthetic depth", "yellow")
                H, W = rgb_img.shape[:2]
                depth_img = np.ones((H, W), dtype=np.float32) * 0.5  # Placeholder depth
        except Exception as e:
            cprint(f"Warning: Failed to get depth image: {e}, using synthetic depth", "yellow")
            H, W = rgb_img.shape[:2]
            depth_img = np.ones((H, W), dtype=np.float32) * 0.5

        # Convert to float and normalize RGB to [0, 1]
        rgb_img_float = rgb_img.astype(np.float32) / 255.0

        # Create Open3D RGBD image
        rgb_o3d = o3d.geometry.Image(rgb_img_float)
        depth_o3d = o3d.geometry.Image(depth_img)

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            convert_rgb_to_intensity=False
        )

        # Create camera intrinsic
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=self.img_size,
            height=self.img_size,
            fx=self.camera_matrix[0, 0],
            fy=self.camera_matrix[1, 1],
            cx=self.camera_matrix[0, 2],
            cy=self.camera_matrix[1, 2]
        )

        # Generate point cloud from RGBD
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic
        )

        # Get points and colors
        points = np.asarray(pcd.points)  # (N, 3)
        colors = np.asarray(pcd.colors) * 255  # (N, 3), convert back to [0, 255]

        # Generate UV coordinates
        H, W = rgb_img.shape[:2]

        # Create pixel coordinates for all points
        # Open3D creates points in camera coordinate system
        # We need to project back to image plane to get UV
        points_homogeneous = np.column_stack([points, np.ones(len(points))])  # (N, 4)

        # Camera projection matrix (simplified pinhole model)
        proj_matrix = np.array([
            [self.camera_matrix[0, 0], 0, self.camera_matrix[0, 2], 0],
            [0, self.camera_matrix[1, 1], self.camera_matrix[1, 2], 0],
            [0, 0, 1, 0]
        ])

        # Project to image plane
        projected = proj_matrix @ points_homogeneous.T  # (3, N)
        projected = projected[:2] / projected[2]  # (2, N), normalize by z

        u_coords = projected[0]  # (N,)
        v_coords = projected[1]  # (N,)

        # Normalize UV to [0, 1]
        u_norm = np.clip(u_coords / W, 0, 1).astype(np.float32)
        v_norm = np.clip(v_coords / H, 0, 1).astype(np.float32)

        # Combine into final point cloud: [x, y, z, r, g, b, u, v]
        point_cloud = np.column_stack([
            points.astype(np.float32),      # xyz coordinates
            colors.astype(np.float32),      # rgb colors
            u_norm,                         # normalized u coordinates
            v_norm                          # normalized v coordinates
        ])

        return point_cloud
