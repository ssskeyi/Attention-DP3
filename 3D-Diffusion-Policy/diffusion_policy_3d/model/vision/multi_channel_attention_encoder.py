import torch
import torch.nn as nn
import torch.nn.functional as F


class PerChannelPointEncoder(nn.Module):
    """
    Per-channel point encoder: operates on a single-channel attention field
    of shape (B, 1, N) and returns a pooled feature (B, out_dim).
    """
    def __init__(self, hidden_dims=(128, 256), out_dim=64):
        super().__init__()
        layers = []
        dim_in = 1
        for dim_out in hidden_dims:
            layers.extend(
                [
                    nn.Linear(dim_in, dim_out),
                    nn.LayerNorm(dim_out),
                    nn.ReLU(),
                ]
            )
            dim_in = dim_out
        self.point_mlp = nn.Sequential(*layers)
        self.final_proj = nn.Sequential(
            nn.Linear(dim_in, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        """
        x: (B, 1, N)
        returns: (B, out_dim)
        """
        # transpose to (B, N, 1)
        x = x.transpose(1, 2)
        x = self.point_mlp(x)  # (B, N, hidden)
        x = torch.max(x, dim=1)[0]  # (B, hidden)
        x = self.final_proj(x)  # (B, out_dim)
        return x


class MultiChannelAttentionFieldEncoder(nn.Module):
    """
    Multi-channel attention field encoder with independent per-channel encoders
    and simple concat+MLP fusion.

    Input:
        x: (B, C, N) attention field
    Output:
        fused: (B, num_fusion_tokens, d_model)  (if num_fusion_tokens==1 returns (B, d_model))
        channel_feats: (B, C, per_channel_out_dim)
    """
    def __init__(
        self,
        in_channels: int = 3,
        n_points: int = 512,
        per_channel_hidden_dims=(128, 256),
        per_channel_out_dim: int = 64,
        d_model: int = 128,
        num_heads: int = 4,
        num_fusion_tokens: int = 1,
        channel_selection: list = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channel_selection = channel_selection
        self.n_points = n_points
        self.per_channel_out_dim = per_channel_out_dim
        self.d_model = d_model
        self.num_fusion_tokens = num_fusion_tokens
        # Determine effective number of channels (after selection if provided)
        if self.channel_selection is not None:
            self.effective_in_channels = len(self.channel_selection)
        else:
            self.effective_in_channels = in_channels

        # Create independent encoder for each channel (independent weights)
        self.channel_encoders = nn.ModuleList(
            [PerChannelPointEncoder(hidden_dims=per_channel_hidden_dims, out_dim=per_channel_out_dim)
             for _ in range(self.effective_in_channels)]
        )

        # Fusion: concatenate all channel features then apply MLP to reduce dimension to d_model
        fusion_in_dim = self.effective_in_channels * per_channel_out_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        x: (B, C, N)
        returns: fused (B, d_model) or (B, num_fusion_tokens, d_model), channel_feats (B, C, per_channel_out_dim)
        """
        B, C, N = x.shape
        # Select channels if requested
        if self.channel_selection is not None:
            x = x[:, self.channel_selection, :]  # (B, selected_C, N)
            C = x.shape[1]
        assert C == self.effective_in_channels, f"Expected {self.effective_in_channels} channels, got {C}"

        # Encode each channel separately
        channel_feats = []
        for i, enc in enumerate(self.channel_encoders):
            xi = x[:, i : i + 1, :]  # (B,1,N)
            feat = enc(xi)  # (B, per_channel_out_dim)
            channel_feats.append(feat)
        # (B, C, per_channel_out_dim)
        channel_feats = torch.stack(channel_feats, dim=1)

        # Concatenate channel features: (B, C * per_channel_out_dim)
        fused_input = channel_feats.view(B, -1)

        # Apply MLP to get fused representation (B, d_model)
        fused = self.fusion_mlp(fused_input)

        return fused, channel_feats


