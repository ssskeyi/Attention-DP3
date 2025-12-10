python scripts/convert_zarr_with_attn3d.py \
  --input_zarr 3D-Diffusion-Policy/data/adroit_door_expert.zarr \
  --json_root 3D-Diffusion-Policy/export_gs2/adroit_door \
  --output_zarr 3D-Diffusion-Policy/data/adroit_door_expert_attn3d.zarr \
  --max_episodes 10 \
  --n_points 512