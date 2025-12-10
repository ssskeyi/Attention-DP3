from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class AdroitDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            use_attn_3d=True,
            attn_3d_n_points=512,
            attn_3d_n_channels=4,
            ):
        super().__init__()
        self.task_name = task_name
        self.use_attn_3d = use_attn_3d
        self.attn_3d_n_points = attn_3d_n_points
        self.attn_3d_n_channels = attn_3d_n_channels
        
        # Load zarr, including attn_3d if available
        keys_to_load = ['state', 'action', 'point_cloud', 'img']
        self.has_attn_3d_in_zarr = False
        if use_attn_3d:
            # Check if attn_3d exists in zarr - if not, raise error
            try:
                test_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=['attn_3d'])
                self.has_attn_3d_in_zarr = True
                keys_to_load.append('attn_3d')
            except (KeyError, ValueError):
                # attn_3d not found - raise error instead of generating on-the-fly
                raise ValueError(
                    f"use_attn_3d=True but attn_3d not found in zarr: {zarr_path}\n"
                    f"Please pre-compute attn_3d using convert_zarr_with_attn3d.py before training."
                )
        
        # Load replay buffer
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=keys_to_load)
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # Extract point cloud - if it has 8 channels (with UV), use only first 6 channels (xyzrgb)
        # This ensures normalization is consistent with training data format
        point_cloud = self.replay_buffer['point_cloud']
        if point_cloud.shape[-1] >= 8:
            # Create a view or copy with only first 6 channels for normalization
            # We need to handle this carefully to avoid modifying the original data
            point_cloud_for_norm = point_cloud[..., :6]
        else:
            point_cloud_for_norm = point_cloud
        
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': point_cloud_for_norm,
        }
        if self.use_attn_3d:
            # Use pre-computed attn_3d from zarr for normalization
            # If we reach here, has_attn_3d_in_zarr must be True (checked in __init__)
            if not self.has_attn_3d_in_zarr:
                raise ValueError("use_attn_3d=True but attn_3d not found in zarr. This should have been caught in __init__.")
            data['attn_3d'] = self.replay_buffer['attn_3d']
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6) or (T, 1024, 8) if has uv
        
        # If point cloud has UV coordinates (8 channels), extract only xyzrgb (6 channels) for training
        # This is necessary because we modified point cloud generation to include UV, but training still uses 6 channels
        # When use_attn_3d=False, this ensures compatibility with both old (6-channel) and new (8-channel) data
        if point_cloud.shape[-1] >= 8:
            point_cloud = point_cloud[:, :, :6]  # Keep only xyzrgb

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        
        # Load pre-computed 3D attention field from zarr - only if use_attn_3d is True
        # When use_attn_3d=False, this block is skipped, making behavior identical to original code
        if self.use_attn_3d:
            if not self.has_attn_3d_in_zarr:
                raise ValueError("use_attn_3d=True but attn_3d not found in zarr. This should have been caught in __init__.")
            if 'attn_3d' not in sample:
                raise ValueError("use_attn_3d=True but attn_3d not found in sample. Check SequenceSampler keys.")
            # Use pre-computed attn_3d from zarr
            attn_3d = sample['attn_3d'].astype(np.float32)  # (T, C, N)
            data['obs']['attn_3d'] = attn_3d
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

