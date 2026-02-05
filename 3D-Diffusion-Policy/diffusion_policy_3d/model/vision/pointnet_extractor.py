import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from .multi_channel_attention_encoder import MultiChannelAttentionFieldEncoder


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    

class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 for xyz, or 3+C for xyz+attention)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    


class AttentionFieldEncoder(nn.Module):
    """
    Encoder for 3D attention field (spatial modality).
    Input: (B, C, N) where C is channels, N is number of points
    Output: (B, out_dim) feature vector
    """
    def __init__(self, in_channels, n_points, out_dim=64, hidden_dims=[128, 256], channel_selection=None):
        super().__init__()
        self.in_channels = in_channels
        self.n_points = n_points
        self.channel_selection = channel_selection  # List of channel indices to use, None means use all

        # Determine effective input channels after selection
        if self.channel_selection is not None:
            effective_in_channels = len(self.channel_selection)
        else:
            effective_in_channels = in_channels

        # Per-point MLP to process each point's features
        layers = []
        dim_in = effective_in_channels
        for dim_out in hidden_dims:
            layers.extend([
                nn.Linear(dim_in, dim_out),
                nn.LayerNorm(dim_out),
                nn.ReLU(),
            ])
            dim_in = dim_out

        self.point_mlp = nn.Sequential(*layers)

        # Global pooling + final projection
        self.final_proj = nn.Sequential(
            nn.Linear(dim_in, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, N) attention field
        Returns:
            feat: (B, out_dim) feature vector
        """
        # Select specific channels if specified
        if self.channel_selection is not None:
            x = x[:, self.channel_selection, :]  # (B, len(channel_selection), N)

        # Transpose to (B, N, C) for per-point processing
        x = x.transpose(1, 2)  # (B, N, C_selected)
        # Process each point
        x = self.point_mlp(x)  # (B, N, hidden_dim)
        # Global max pooling
        x = torch.max(x, dim=1)[0]  # (B, hidden_dim)
        # Final projection
        x = self.final_proj(x)  # (B, out_dim)
        return x


class DP3Encoder(nn.Module):
    def __init__(self,
                 observation_space: Dict,
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 attn_3d_encoder_dim=64,
                 attn_channels=None,  # List of attention channel indices to use, None means use all
                 fusion_strategy='late',  # 'late' or 'early' fusion
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.attn_3d_key = 'attn_3d'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        self.attn_channels = attn_channels  # Store channel selection
        self.fusion_strategy = fusion_strategy  # Store fusion strategy
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None

        # Check if attn_3d is in observation space
        self.use_attn_3d = self.attn_3d_key in observation_space.keys()
        if self.use_attn_3d:
            self.attn_3d_shape = observation_space[self.attn_3d_key]
            # attn_3d_shape should be [C, N]
            if len(self.attn_3d_shape) == 2:
                attn_3d_channels, attn_3d_n_points = self.attn_3d_shape
            else:
                raise ValueError(f"attn_3d shape should be [C, N], got {self.attn_3d_shape}")

            # For early fusion, we don't need separate attention encoder
            if self.fusion_strategy == 'early':
                self.attn_3d_encoder = None
                # Early fusion: concatenate attention channels to point cloud
                # Determine effective attention channels after selection
                effective_attn_channels = len(attn_channels) if attn_channels is not None else attn_3d_channels
            else:  # late fusion
                # Create attention field encoder with independent per-channel encoders
                # and cross-attention fusion. Preserve channel selection behavior.
                self.attn_3d_encoder = MultiChannelAttentionFieldEncoder(
                    in_channels=attn_3d_channels,
                    n_points=attn_3d_n_points,
                    per_channel_hidden_dims=(128, 256),
                    per_channel_out_dim=64,
                    d_model=attn_3d_encoder_dim,
                    num_heads=4,
                    num_fusion_tokens=1,
                    channel_selection=attn_channels,
                )
                self.n_output_channels += attn_3d_encoder_dim
                effective_attn_channels = 0

            cprint(f"[DP3Encoder] attn_3d shape: {self.attn_3d_shape}", "yellow")
            cprint(f"[DP3Encoder] fusion_strategy: {self.fusion_strategy}", "yellow")
            cprint(f"[DP3Encoder] attn_channels: {attn_channels}", "yellow")
            if self.fusion_strategy == 'early':
                cprint(f"[DP3Encoder] early fusion: effective_attn_channels={effective_attn_channels}", "yellow")
            else:
                cprint(f"[DP3Encoder] attn_3d encoder output dim: {attn_3d_encoder_dim}", "yellow")
        else:
            self.attn_3d_shape = None
            self.attn_3d_encoder = None
            effective_attn_channels = 0
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if pointcloud_encoder_cfg is None:
                pointcloud_encoder_cfg = {}
            # Create a copy to avoid modifying the original config
            encoder_cfg = copy.copy(pointcloud_encoder_cfg)
            # Determine base channels based on point cloud type
            if use_pc_color:
                base_channels = 6  # xyz + rgb
            else:
                base_channels = 3  # xyz only

            # For early fusion, add attention channels to input
            if self.fusion_strategy == 'early' and self.use_attn_3d:
                effective_attn_channels = len(attn_channels) if attn_channels is not None else attn_3d_channels
                total_channels = base_channels + effective_attn_channels
                cprint(f"[DP3Encoder] Early fusion: base_channels={base_channels}, attn_channels={effective_attn_channels}, total={total_channels}", "yellow")
                # Early fusion always needs RGB encoder (supports 6 channels)
                encoder_cfg['in_channels'] = total_channels
                self.extractor = PointNetEncoderXYZRGB(**encoder_cfg)
            else:
                total_channels = base_channels
                # Use appropriate encoder based on color usage
                if use_pc_color:
                    encoder_cfg['in_channels'] = total_channels
                    self.extractor = PointNetEncoderXYZRGB(**encoder_cfg)
                else:
                    encoder_cfg['in_channels'] = total_channels
                    self.extractor = PointNetEncoderXYZ(**encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)

        # Handle early fusion: concatenate attention field to point cloud
        if self.fusion_strategy == 'early' and self.use_attn_3d and self.attn_3d_key in observations:
            attn_3d = observations[self.attn_3d_key]  # (B, C, N)
            if len(attn_3d.shape) == 3:
                # Select channels if specified
                if self.attn_channels is not None:
                    attn_3d = attn_3d[:, self.attn_channels, :]  # (B, selected_C, N)

                # Reshape attention field to match point cloud format (B, N, C)
                attn_reshaped = attn_3d.transpose(1, 2)  # (B, N, C)

                # For each point in the point cloud, concatenate attention features
                B, N_pc, C_pc = points.shape
                _, N_attn, C_attn = attn_reshaped.shape

                # Ensure dimensions match (use min of N_pc and N_attn)
                N = min(N_pc, N_attn)
                points = points[:, :N, :]  # (B, N, C_pc)
                attn_reshaped = attn_reshaped[:, :N, :]  # (B, N, C_attn)

                # Concatenate along feature dimension
                points = torch.cat([points, attn_reshaped], dim=-1)  # (B, N, C_pc + C_attn)
            else:
                raise ValueError(f"attn_3d shape should be (B, C, N), got {attn_3d.shape}")

        # points: B * N * C_total (after potential concatenation with attention)
        pn_feat = self.extractor(points)    # B * out_channel

        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64

        # Process 3D attention field for late fusion
        feat_list = [pn_feat, state_feat]
        if self.fusion_strategy == 'late' and self.use_attn_3d and self.attn_3d_encoder is not None and self.attn_3d_key in observations:
            attn_3d = observations[self.attn_3d_key]
            # attn_3d should be (B, C, N)
            if len(attn_3d.shape) == 3:
                attn_fused, _ = self.attn_3d_encoder(attn_3d)  # (B, attn_3d_encoder_dim)
                feat_list.append(attn_fused)
            else:
                raise ValueError(f"attn_3d shape should be (B, C, N), got {attn_3d.shape}")

        final_feat = torch.cat(feat_list, dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels