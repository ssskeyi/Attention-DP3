"""
Single image inference script for Grounded-SAM-2.
This script runs in aedp3_vis environment and processes a single image.
Used by AdroitRunner to generate attn_3d during inference.
"""
import os
import sys
import argparse
import json
import warnings
import numpy as np
import torch
import random
from torchvision.ops import box_convert
import pycocotools.mask as mask_util
import cv2

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def single_mask_to_rle(mask: np.ndarray):
    """Convert binary mask (H, W) to RLE for json serialization."""
    encoded = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))
    # mask_util.encode returns a list of RLE dicts, get the first one
    rle = encoded[0]  # type: ignore
    rle["counts"] = rle["counts"].decode("utf-8")  # type: ignore
    return rle


def run_on_image(
    img_path: str,
    text_prompt: str,
    sam2_checkpoint: str,
    sam2_config: str,
    gdino_config: str,
    gdino_checkpoint: str,
    device: str,
    box_thr: float = 0.35,
    text_thr: float = 0.25,
    multimask: bool = False,
) -> dict:
    """Run Grounded-SAM-2 on a single image."""
    # Note: run_on_image does not set seeds itself; caller (main) should set seed
    image_source, image = load_image(img_path)

    # GroundingDINO
    grounding_model = load_model(
        model_config_path=gdino_config,
        model_checkpoint_path=gdino_checkpoint,
        device=device,
    )
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_thr,
        text_threshold=text_thr,
        device=device,
    )

    h, w, _ = image_source.shape
    
    # Handle empty detections
    if len(boxes) == 0:
        # Return empty results if no detections
        return {
            "image_path": img_path,
            "annotations": [],
        }
    
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    # Ensure input_boxes is 2D with shape (n, 4)
    if input_boxes.ndim == 1:
        input_boxes = input_boxes.reshape(1, -1)

    # SAM2
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    sam2_predictor.set_image(image_source)
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=multimask,
    )
    if multimask:
        best = np.argmax(scores, axis=1)
        masks = masks[np.arange(masks.shape[0]), best]

    # shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    results = {
        "image_path": img_path,
        "annotations": [],
    }
    for cls, box, mask, score in zip(labels, input_boxes.tolist(), masks, scores.tolist()):
        results["annotations"].append(
            {
                "class_name": cls,
                "bbox": box,
                "segmentation": single_mask_to_rle(mask),
                "score": score,
                "box_format": "xyxy",
                "img_width": w,
                "img_height": h,
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", required=True, help="Path to input image")
    parser.add_argument("--output_json", required=True, help="Path to output JSON file")
    parser.add_argument("--text", required=True, help='e.g. "door handle. door." (lowercase + dot)')
    parser.add_argument("--sam2_ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument(
        "--gdino_cfg",
        default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument("--gdino_ckpt", default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box_thr", type=float, default=0.35)
    parser.add_argument("--text_thr", type=float, default=0.25)
    parser.add_argument("--multimask", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for deterministic inference (default: 0)")
    args = parser.parse_args()

    # Set deterministic seeds for reproducibility
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Make cudnn deterministic where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(int(args.seed))

    # Get Grounded-SAM-2 root (current directory)
    gs2_root = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve checkpoint paths (need absolute paths)
    if not os.path.isabs(args.sam2_ckpt):
        args.sam2_ckpt = os.path.join(gs2_root, args.sam2_ckpt)
    if not os.path.isabs(args.gdino_ckpt):
        args.gdino_ckpt = os.path.join(gs2_root, args.gdino_ckpt)
    
    # For config files:
    # - gdino_cfg: needs absolute path for load_model
    if not os.path.isabs(args.gdino_cfg):
        args.gdino_cfg = os.path.join(gs2_root, args.gdino_cfg)
    
    # sam2_cfg: Keep as relative path (same as batch_grounded_sam2.py)
    # batch_grounded_sam2.py passes "configs/sam2.1/sam2.1_hiera_l.yaml" directly to build_sam2
    # We do the same - just ensure it's a relative path from gs2_root
    if os.path.isabs(args.sam2_cfg):
        args.sam2_cfg = os.path.relpath(args.sam2_cfg, gs2_root)
    # Keep it as-is (should be "configs/sam2.1/sam2.1_hiera_l.yaml" format)

    # Run inference
    results = run_on_image(
        args.img_path,
        args.text,
        args.sam2_ckpt,
        args.sam2_cfg,
        args.gdino_cfg,
        args.gdino_ckpt,
        args.device,
        args.box_thr,
        args.text_thr,
        multimask=args.multimask,
    )

    # Save results
    with open(args.output_json, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

