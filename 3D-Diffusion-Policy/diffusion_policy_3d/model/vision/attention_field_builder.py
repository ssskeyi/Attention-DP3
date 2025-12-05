"""
Simple 3D Attention Field Builder (MVP version)
This module generates a simplified 3D attention field for testing the integration.
In the full version, this will be replaced with LanguageGrounder2D + Fusion3DAttention.
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple


class SimpleAttentionFieldBuilder:
    """
    MVP version: Generate a simple 3D attention field from point cloud.
    This is a placeholder that creates attention weights based on point cloud geometry.
    In the full implementation, this will use 2D grounding + 3D fusion.
    """
    
    def __init__(
        self,
        n_points: int = 1600,
        n_channels: int = 4,
        attention_radius: float = 0.1,
        use_gaussian: bool = True,
    ):
        """
        Args:
            n_points: Number of 3D points to sample (N)
            n_channels: Number of channels in attention field (C)
            attention_radius: Radius for attention weighting
            use_gaussian: Whether to use Gaussian weighting
        """
        self.n_points = n_points
        self.n_channels = n_channels
        self.attention_radius = attention_radius
        self.use_gaussian = use_gaussian
    
    def build_attention_field(
        self,
        point_cloud: np.ndarray,
        agent_pos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build a 3D attention field from point cloud.
        
        Args:
            point_cloud: Point cloud array of shape (N_pc, 3) or (N_pc, 6)
            agent_pos: Agent position array of shape (D_pos,), optional
            
        Returns:
            attention_field: Array of shape (C, N) where C=n_channels, N=n_points
        """
        # Extract XYZ coordinates (first 3 channels)
        if point_cloud.shape[-1] >= 3:
            pc_xyz = point_cloud[..., :3]
        else:
            raise ValueError(f"Point cloud must have at least 3 channels, got {point_cloud.shape[-1]}")
        
        # Sample N points from point cloud (or use all if N_pc <= n_points)
        n_pc = pc_xyz.shape[0]
        if n_pc >= self.n_points:
            # Randomly sample n_points
            indices = np.random.choice(n_pc, self.n_points, replace=False)
            sampled_points = pc_xyz[indices]
        else:
            # Use all points and pad with zeros or repeat
            sampled_points = np.zeros((self.n_points, 3), dtype=pc_xyz.dtype)
            sampled_points[:n_pc] = pc_xyz
            # Repeat last point to fill
            if n_pc > 0:
                sampled_points[n_pc:] = pc_xyz[-1]
        
        # Build attention field: [C, N]
        # Channel 0: Distance to nearest surface (simplified: distance to origin or agent)
        # Channel 1: Attention weight for "target object" (simplified: distance-based)
        # Channel 2: Attention weight for "obstacle" (simplified: inverse distance)
        # Channel 3: Normalized spatial coordinates (x, y, z normalized)
        
        attention_field = np.zeros((self.n_channels, self.n_points), dtype=np.float32)
        
        # Compute center point (use agent_pos if available, else use point cloud centroid)
        if agent_pos is not None and len(agent_pos) >= 3:
            center = agent_pos[:3]
        else:
            center = np.mean(sampled_points, axis=0)
        
        # Compute distances from center
        distances = np.linalg.norm(sampled_points - center, axis=1)
        max_dist = np.max(distances) + 1e-6
        normalized_distances = distances / max_dist
        
        # Channel 0: Distance to surface (normalized)
        attention_field[0] = normalized_distances
        
        # Channel 1: Target object attention (Gaussian around center)
        if self.use_gaussian:
            attention_field[1] = np.exp(-distances**2 / (2 * self.attention_radius**2))
        else:
            attention_field[1] = 1.0 / (1.0 + distances)
        
        # Channel 2: Obstacle attention (inverse, peaks at far points)
        attention_field[2] = 1.0 - attention_field[1]
        
        # Channel 3: Normalized spatial coordinate (use x coordinate as example)
        # In full version, this would be a learned feature
        attention_field[3] = (sampled_points[:, 0] - center[0]) / max_dist
        
        return attention_field
    
    def build_attention_field_batch(
        self,
        point_clouds: np.ndarray,
        agent_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Build attention fields for a batch of point clouds.
        
        Args:
            point_clouds: Array of shape (B, T, N_pc, 3) or (B, T, N_pc, 6)
            agent_positions: Array of shape (B, T, D_pos), optional
            
        Returns:
            attention_fields: Array of shape (B, T, C, N)
        """
        B, T = point_clouds.shape[:2]
        attention_fields = []
        
        for b in range(B):
            batch_fields = []
            for t in range(T):
                pc = point_clouds[b, t]
                agent_pos = agent_positions[b, t] if agent_positions is not None else None
                attn_field = self.build_attention_field(pc, agent_pos)
                batch_fields.append(attn_field)
            attention_fields.append(np.stack(batch_fields, axis=0))
        
        return np.stack(attention_fields, axis=0)  # (B, T, C, N)

