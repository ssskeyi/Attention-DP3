cd /mnt/disk2/ycb/AEDP3/Grounded-SAM-2

python batch_grounded_sam2.py \
  --frames_root ../3D-Diffusion-Policy/export/adroit_door_frames \
  --output_root ../3D-Diffusion-Policy/export_gs2/adroit_door \
  --text "door handle. door." \
  --sam2_ckpt checkpoints/sam2.1_hiera_large.pt \
  --sam2_cfg configs/sam2.1/sam2.1_hiera_l.yaml \
  --gdino_cfg grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --gdino_ckpt gdino_checkpoints/groundingdino_swint_ogc.pth \
  --device cuda