"""
Grounded-SAM-2 API Server
Provides HTTP API interface for Grounded-SAM-2 inference to avoid reloading models.
Run this server once and it will keep models loaded in memory for fast inference.
"""
import os
import sys
import json
import base64
import warnings
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_convert
import pycocotools.mask as mask_util
from PIL import Image
import cv2
import random

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
except ImportError:
    print("Flask and flask-cors are required. Install with: pip install flask flask-cors")
    sys.exit(1)

from grounding_dino.groundingdino.util.inference import load_model, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global model variables (loaded once at startup)
grounding_model = None
sam2_model = None
sam2_predictor = None
device = None
server_seed = 0


def single_mask_to_rle(mask: np.ndarray):
    """Convert binary mask (H, W) to RLE for json serialization."""
    encoded = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))
    rle = encoded[0]  # type: ignore
    rle["counts"] = rle["counts"].decode("utf-8")  # type: ignore
    return rle


def load_models(
    sam2_checkpoint: str,
    sam2_config: str,
    gdino_config: str,
    gdino_checkpoint: str,
    device_str: str,
):
    """Load models once at startup."""
    global grounding_model, sam2_model, sam2_predictor, device
    
    device = device_str
    
    # Validate CUDA device if specified
    if device.startswith("cuda"):
        if ":" in device:
            # Format: cuda:0, cuda:1, etc.
            device_id = int(device.split(":")[1])
            if device_id >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {device_id} not available. Available devices: 0-{torch.cuda.device_count()-1}")
            print(f"[GS2-API] Using CUDA device {device_id}: {torch.cuda.get_device_name(device_id)}")
        else:
            # Format: cuda (uses default device)
            if torch.cuda.is_available():
                print(f"[GS2-API] Using default CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                raise RuntimeError("CUDA is not available. Use 'cpu' or check your CUDA installation.")
    elif device == "cpu":
        print(f"[GS2-API] Using CPU device")
    else:
        print(f"[GS2-API] Using device: {device}")
    
    print(f"[GS2-API] Loading models on device: {device}")
    
    # Load GroundingDINO
    print("[GS2-API] Loading GroundingDINO...")
    grounding_model = load_model(
        model_config_path=gdino_config,
        model_checkpoint_path=gdino_checkpoint,
        device=device,
    )
    print("[GS2-API] GroundingDINO loaded.")
    
    # Load SAM2
    print("[GS2-API] Loading SAM2...")
    sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("[GS2-API] SAM2 loaded.")
    
    print("[GS2-API] All models loaded successfully!")


def set_seed(seed_val: int):
    """Set random seeds for reproducible inference."""
    global server_seed
    server_seed = int(seed_val)
    random.seed(server_seed)
    np.random.seed(server_seed)
    torch.manual_seed(server_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(server_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_inference(
    image_source: np.ndarray,
    text_prompt: str,
    box_thr: float = 0.35,
    text_thr: float = 0.25,
    multimask: bool = False,
) -> dict:
    """Run inference on an image (models must be loaded)."""
    global grounding_model, sam2_predictor, device
    
    if grounding_model is None or sam2_predictor is None or device is None:
        raise RuntimeError("Models not loaded. Call load_models() first.")
    
    # Prepare image for GroundingDINO
    # load_image expects a file path, but we have numpy array
    # So we need to transform it manually using the same transforms as load_image
    import grounding_dino.groundingdino.datasets.transforms as T
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_pil = Image.fromarray(image_source)
    image_transformed, _ = transform(image_pil, None)
    
    # GroundingDINO prediction
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image_transformed,
        caption=text_prompt,
        box_threshold=box_thr,
        text_threshold=text_thr,
        device=device,
    )
    
    h, w, _ = image_source.shape
    
    # Handle empty detections
    if len(boxes) == 0:
        return {
            "annotations": [],
            "img_width": w,
            "img_height": h,
        }
    
    # Convert boxes to pixel coordinates
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    # Ensure input_boxes is 2D with shape (n, 4)
    if input_boxes.ndim == 1:
        input_boxes = input_boxes.reshape(1, -1)
    
    # SAM2 prediction
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
        scores = scores[np.arange(scores.shape[0]), best]
    else:
        # When multimask=False, scores might be (n, 1) instead of (n,)
        # Flatten to ensure 1D array (don't use squeeze as it can return scalar when n=1)
        scores = scores.flatten()

    # shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    results = {
        "annotations": [],
        "img_width": w,
        "img_height": h,
    }
    
    for cls, box, mask, score in zip(labels, input_boxes.tolist(), masks, scores.tolist()):
        results["annotations"].append(
            {
                "class_name": cls,
                "bbox": box,
                "segmentation": single_mask_to_rle(mask),
                "score": float(score),
                "box_format": "xyxy",
            }
        )
    
    return results


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    models_loaded = grounding_model is not None and sam2_predictor is not None
    return jsonify({
        "status": "ok" if models_loaded else "models_not_loaded",
        "models_loaded": models_loaded,
    })


@app.route("/infer", methods=["POST"])
def infer():
    """Inference endpoint. Accepts image as base64 or file path."""
    try:
        data = request.get_json()
        
        # Get parameters
        text_prompt = data.get("text", "")
        box_thr = float(data.get("box_thr", 0.35))
        text_thr = float(data.get("text_thr", 0.25))
        multimask = bool(data.get("multimask", False))
        
        # Get image - support both base64 and file path
        image_source = None
        
        if "image_base64" in data:
            # Decode base64 image
            img_data = base64.b64decode(data["image_base64"])
            img_pil = Image.open(BytesIO(img_data))
            image_source = np.array(img_pil.convert("RGB"))
        elif "image_path" in data:
            # Load from file path
            img_path = data["image_path"]
            if not os.path.isabs(img_path):
                # If relative path, assume it's relative to current working directory
                img_path = os.path.abspath(img_path)
            image_source = cv2.imread(img_path)
            if image_source is None:
                return jsonify({"error": f"Failed to load image from {img_path}"}), 400
            image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
        else:
            return jsonify({"error": "Either 'image_base64' or 'image_path' must be provided"}), 400
        
        # Optionally set per-request seed (falls back to server default)
        req_seed = data.get("seed", None)
        if req_seed is not None:
            try:
                set_seed(int(req_seed))
            except Exception:
                # ignore invalid seed and continue with server default
                pass

        # Run inference
        results = run_inference(
            image_source=image_source,
            text_prompt=text_prompt,
            box_thr=box_thr,
            text_thr=text_thr,
            multimask=multimask,
        )
        
        return jsonify(results)
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({"error": error_msg}), 500


def main():
    parser = argparse.ArgumentParser(description="Grounded-SAM-2 API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to")
    parser.add_argument("--sam2_ckpt", default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument(
        "--gdino_cfg",
        default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument("--gdino_ckpt", default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--device", default="cuda", 
                        help="Device to use (e.g., 'cuda', 'cuda:0', 'cuda:1', 'cpu'). Default: 'cuda'")
    parser.add_argument("--seed", type=int, default=0, help="Server default random seed for deterministic inference (default: 0)")
    args = parser.parse_args()
    
    # Get Grounded-SAM-2 root (current directory)
    gs2_root = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve checkpoint paths
    if not os.path.isabs(args.sam2_ckpt):
        args.sam2_ckpt = os.path.join(gs2_root, args.sam2_ckpt)
    if not os.path.isabs(args.gdino_ckpt):
        args.gdino_ckpt = os.path.join(gs2_root, args.gdino_ckpt)
    if not os.path.isabs(args.gdino_cfg):
        args.gdino_cfg = os.path.join(gs2_root, args.gdino_cfg)
    
    # sam2_cfg: Keep as relative path
    if os.path.isabs(args.sam2_cfg):
        args.sam2_cfg = os.path.relpath(args.sam2_cfg, gs2_root)
    
    # Load models
    try:
        # Set server-level seed before loading models for deterministic initialization
        set_seed(int(args.seed))

        load_models(
            sam2_checkpoint=args.sam2_ckpt,
            sam2_config=args.sam2_cfg,
            gdino_config=args.gdino_cfg,
            gdino_checkpoint=args.gdino_ckpt,
            device_str=args.device,
        )
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Start server
    print(f"[GS2-API] Starting server on {args.host}:{args.port}")
    print(f"[GS2-API] API endpoint: http://{args.host}:{args.port}/infer")
    print(f"[GS2-API] Health check: http://{args.host}:{args.port}/health")
    
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()

