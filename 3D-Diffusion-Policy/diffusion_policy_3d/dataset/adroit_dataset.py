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
from diffusion_policy_3d.model.vision.attention_field_builder import SimpleAttentionFieldBuilder

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
            attn_3d_n_points=1600,
            attn_3d_n_channels=4,
            ):
        super().__init__()
        self.task_name = task_name
        self.use_attn_3d = use_attn_3d
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'point_cloud', 'img'])
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
        
        # Initialize attention field builder if needed
        if self.use_attn_3d:
            self.attn_builder = SimpleAttentionFieldBuilder(
                n_points=attn_3d_n_points,
                n_channels=attn_3d_n_channels,
            )

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
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:],
            'point_cloud': self.replay_buffer['point_cloud'],
        }
        if self.use_attn_3d:
            # Generate attention fields for normalization
            # Sample a subset of episodes for efficiency
            n_samples = min(100, self.replay_buffer.n_episodes)
            attn_3d_samples = []
            for ep_idx in range(0, self.replay_buffer.n_episodes, max(1, self.replay_buffer.n_episodes // n_samples)):
                ep_data = self.replay_buffer.get_episode(ep_idx)
                pc = ep_data['point_cloud']
                state = ep_data['state']
                # Generate attention field for first timestep as example
                if len(pc) > 0:
                    attn_field = self.attn_builder.build_attention_field(
                        pc[0], state[0] if len(state) > 0 else None
                    )
                    attn_3d_samples.append(attn_field)
            if len(attn_3d_samples) > 0:
                # Stack and reshape: (N_samples, C, N) -> (N_samples * N, C) for normalization
                attn_3d_array = np.stack(attn_3d_samples, axis=0)
                data['attn_3d'] = attn_3d_array.reshape(-1, attn_3d_array.shape[-2], attn_3d_array.shape[-1])
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        point_cloud = sample['point_cloud'][:,].astype(np.float32) # (T, 1024, 6)

        data = {
            'obs': {
                'point_cloud': point_cloud, # T, 1024, 6
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        
        # Generate 3D attention field if enabled
        if self.use_attn_3d:
            T = point_cloud.shape[0]
            attn_3d_list = []
            for t in range(T):
                attn_field = self.attn_builder.build_attention_field(
                    point_cloud[t], agent_pos[t] if len(agent_pos) > t else None
                )
                attn_3d_list.append(attn_field)
            attn_3d = np.stack(attn_3d_list, axis=0)  # (T, C, N)
            data['obs']['attn_3d'] = attn_3d.astype(np.float32)
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

