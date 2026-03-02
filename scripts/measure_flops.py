"""
FLOPs and Delay Measurement Script for AEDP3 vs DP3

This script measures:
- Total FLOPs (Floating Point Operations) for AEDP3 and DP3
- Delay (time from input observation to output action) for AEDP3 and DP3
- Grounded-SAM-2 FLOPs and Delay (via local model loading or API mode)

Requirements:
    pip install fvcore requests

Usage:
    # Using local models for FLOPs measurement (recommended):
    python scripts/measure_flops.py --gs2_root /path/to/Grounded-SAM-2 [--gs2_api_url URL]
    
    # Using API mode only (FLOPs not available):
    python scripts/measure_flops.py [--gs2_api_url URL]
    
    For delay measurement, the GS2 API server must be running.
    You can start it with: cd Grounded-SAM-2 && python gs2_api_server.py
    
    Optional GS2 model path arguments:
    --gs2_sam2_ckpt: Path to SAM2 checkpoint
    --gs2_sam2_cfg: Path to SAM2 config
    --gs2_gdino_cfg: Path to GroundingDINO config
    --gs2_gdino_ckpt: Path to GroundingDINO checkpoint
    --gs2_device: Device for GS2 models ('cpu' or 'cuda')
"""

import os
import sys
import pathlib
import time
import tempfile
from typing import Optional, Dict, Tuple

# Import requests for API calls
try:
    import requests
except ImportError:
    print("Error: requests library is required for API mode")
    print("Please install with: pip install requests")
    sys.exit(1)

# Import fvcore for FLOPs measurement
try:
    from fvcore.nn import flop_count
    try:
        from fvcore.nn import FlopCountMode
        HAS_FLOP_COUNT_MODE = True
    except ImportError:
        HAS_FLOP_COUNT_MODE = False
        print("Note: Using older fvcore API (FlopCountMode not available)")
except ImportError as e:
    print(f"Error: Could not import fvcore.nn")
    print(f"Please install fvcore with: pip install fvcore")
    sys.exit(1)

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
import hydra
from diffusion_policy_3d.model.common.normalizer import SingleFieldLinearNormalizer
import argparse


def setup_root():
    """Add project root to sys.path and chdir there."""
    this_file = pathlib.Path(__file__).resolve()
    root_dir = this_file.parent.parent
    local_dp3_root = str(root_dir / "3D-Diffusion-Policy")
    if local_dp3_root not in sys.path:
        sys.path.insert(0, local_dp3_root)
    os.chdir(root_dir)
    return root_dir


def compose_dp3_cfg(config_dir: pathlib.Path, task_name: str):
    """Use Hydra's compose API to get config with a specific task override."""
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    if not OmegaConf.has_resolver("now"):
        import datetime
        OmegaConf.register_new_resolver(
            "now",
            lambda fmt=None: datetime.datetime.now().strftime(fmt) if fmt else datetime.datetime.now().isoformat(),
            replace=True,
        )

    with initialize_config_dir(config_dir=str(config_dir), job_name=f"dp3_{task_name}"):
        cfg = compose(config_name="dp3", overrides=[f"task={task_name}"])
    return cfg


def create_dummy_obs(shape_meta: dict, batch_size: int = 1, n_obs_steps: int = 8, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Create dummy observation dict based on shape_meta."""
    obs_dict = {}
    obs_shape_meta = shape_meta['obs']
    
    for key, meta in obs_shape_meta.items():
        shape = meta['shape']
        obs_type = meta.get('type', 'low_dim')
        
        if obs_type == 'point_cloud':
            obs_dict[key] = torch.randn(batch_size, n_obs_steps, *shape, device=device, dtype=torch.float32)
        elif obs_type == 'spatial':
            obs_dict[key] = torch.randn(batch_size, n_obs_steps, *shape, device=device, dtype=torch.float32)
        else:
            obs_dict[key] = torch.randn(batch_size, n_obs_steps, *shape, device=device, dtype=torch.float32)
    
    return obs_dict


def measure_encoder_flops(obs_encoder: nn.Module, obs_dict: Dict[str, torch.Tensor], device: str = "cpu"):
    """Measure FLOPs for the observation encoder."""
    obs_encoder.eval()
    obs_encoder.to(device)
    
    B, T = next(iter(obs_dict.values())).shape[:2]
    obs_encoder_input = {}
    for k, v in obs_dict.items():
        obs_encoder_input[k] = v.reshape(B * T, *v.shape[2:]).to(device)
    
    try:
        class EncoderWrapper(nn.Module):
            def __init__(self, encoder):
                super().__init__()
                self.encoder = encoder
            def forward(self, obs_dict):
                return self.encoder(obs_dict)
        
        wrapper = EncoderWrapper(obs_encoder)
        if HAS_FLOP_COUNT_MODE:
            flops_dict, _ = flop_count(wrapper, (obs_encoder_input,), mode=FlopCountMode.OPERATION_COUNT)
        else:
            flops_dict, _ = flop_count(wrapper, (obs_encoder_input,))
        encoder_flops = sum(flops_dict.values())
        return encoder_flops, flops_dict
    except Exception as e:
        print(f"  Warning: Encoder FLOPs measurement failed: {e}")
        return None, None


def measure_diffusion_flops(diffusion_model: nn.Module, horizon: int, action_dim: int, 
                            obs_feature_dim: int, num_inference_steps: int, 
                            obs_as_global_cond: bool = True, n_obs_steps: int = 8,
                            condition_type: str = "film", device: str = "cpu"):
    """Measure FLOPs for one diffusion step, then multiply by num_inference_steps."""
    diffusion_model.eval()
    diffusion_model.to(device)
    
    if hasattr(diffusion_model, 'final_conv'):
        final_conv_last = diffusion_model.final_conv[-1]
        if isinstance(final_conv_last, nn.Conv1d):
            input_dim = final_conv_last.out_channels
        else:
            input_dim = action_dim if obs_as_global_cond else (action_dim + obs_feature_dim)
    else:
        input_dim = action_dim if obs_as_global_cond else (action_dim + obs_feature_dim)
    
    if obs_as_global_cond:
        if "cross_attention" in condition_type:
            global_cond_dim = obs_feature_dim
        else:
            global_cond_dim = obs_feature_dim * n_obs_steps
    else:
        global_cond_dim = None
    
    batch_size = 1
    dummy_trajectory = torch.randn(batch_size, horizon, input_dim, device=device)
    dummy_timestep = torch.tensor([500], device=device, dtype=torch.long)
    dummy_global_cond = torch.randn(batch_size, global_cond_dim, device=device) if global_cond_dim is not None else None
    
    try:
        class DiffusionWrapper(nn.Module):
            def __init__(self, model, has_global_cond):
                super().__init__()
                self.model = model
                self.has_global_cond = has_global_cond
            def forward(self, sample, timestep, global_cond=None):
                if self.has_global_cond:
                    return self.model(sample=sample, timestep=timestep, global_cond=global_cond)
                else:
                    return self.model(sample=sample, timestep=timestep, global_cond=None)
        
        wrapper = DiffusionWrapper(diffusion_model, has_global_cond=(global_cond_dim is not None))
        if dummy_global_cond is not None:
            args = (dummy_trajectory, dummy_timestep, dummy_global_cond)
        else:
            args = (dummy_trajectory, dummy_timestep)
        
        if HAS_FLOP_COUNT_MODE:
            flops_dict, _ = flop_count(wrapper, args, mode=FlopCountMode.OPERATION_COUNT)
        else:
            flops_dict, _ = flop_count(wrapper, args)
        step_flops = sum(flops_dict.values())
        total_diffusion_flops = step_flops * num_inference_steps
        return step_flops, total_diffusion_flops, flops_dict
    except Exception as e:
        print(f"  Warning: Diffusion FLOPs measurement failed: {e}")
        return None, None, None


def measure_model_flops(model: nn.Module, obs_dict: Dict[str, torch.Tensor], device: str = "cpu"):
    """Measure total FLOPs for a single forward pass through predict_action."""
    model.eval()
    model.to(device)
    
    obs_dict_device = {k: v.to(device) for k, v in obs_dict.items()}
    
    print("  Measuring encoder FLOPs...")
    encoder_flops, encoder_flops_dict = measure_encoder_flops(
        model.obs_encoder, obs_dict_device, device=device
    )
    
    print("  Measuring diffusion model FLOPs...")
    num_inference_steps = getattr(model, 'num_inference_steps', 100)
    step_flops, total_diffusion_flops, diffusion_flops_dict = measure_diffusion_flops(
        model.model,
        horizon=model.horizon,
        action_dim=model.action_dim,
        obs_feature_dim=model.obs_feature_dim,
        num_inference_steps=num_inference_steps,
        obs_as_global_cond=getattr(model, 'obs_as_global_cond', True),
        n_obs_steps=getattr(model, 'n_obs_steps', 8),
        condition_type=getattr(model, 'condition_type', 'film'),
        device=device
    )
    
    if encoder_flops is not None and total_diffusion_flops is not None:
        total_flops = encoder_flops + total_diffusion_flops
        return total_flops
    else:
        return None


def setup_identity_normalizer(model: nn.Module, obs_dict: Dict[str, torch.Tensor]):
    """Setup identity normalizer for the model."""
    from diffusion_policy_3d.model.common.normalizer import SingleFieldLinearNormalizer
    
    # Create identity normalizers for all observation keys and action
    for key in obs_dict.keys():
        model.normalizer[key] = SingleFieldLinearNormalizer.create_identity()
    
    # Also set identity normalizer for action
    action_shape = model.action_shape
    if len(action_shape) == 1:
        action_dim = action_shape[0]
    else:
        action_dim = action_shape[0] * action_shape[1]
    
    # Create identity normalizer for action (with correct dimension)
    action_normalizer = SingleFieldLinearNormalizer.create_identity()
    # Expand to match action dimension
    if action_normalizer.params_dict['scale'].shape[0] == 1 and action_dim > 1:
        scale = torch.ones(action_dim, dtype=action_normalizer.params_dict['scale'].dtype)
        offset = torch.zeros(action_dim, dtype=action_normalizer.params_dict['offset'].dtype)
        input_stats_dict = {
            'min': torch.full((action_dim,), -1.0, dtype=action_normalizer.params_dict['input_stats']['min'].dtype),
            'max': torch.full((action_dim,), 1.0, dtype=action_normalizer.params_dict['input_stats']['max'].dtype),
            'mean': torch.zeros(action_dim, dtype=action_normalizer.params_dict['input_stats']['mean'].dtype),
            'std': torch.ones(action_dim, dtype=action_normalizer.params_dict['input_stats']['std'].dtype)
        }
        action_normalizer = SingleFieldLinearNormalizer.create_manual(scale, offset, input_stats_dict)
    
    model.normalizer['action'] = action_normalizer


def measure_model_delay(model: nn.Module, obs_dict: Dict[str, torch.Tensor], 
                       num_warmup: int = 3, num_runs: int = 50, device: str = "cpu"):
    """Measure delay (time from input observation to output action)."""
    model.eval()
    model.to(device)
    
    # Setup identity normalizer if not already set
    if len(model.normalizer.params_dict) == 0:
        setup_identity_normalizer(model, obs_dict)
        # Move normalizer to device after setting it up
        model.normalizer.to(device)
    
    obs_dict_device = {k: v.to(device) for k, v in obs_dict.items()}
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.predict_action(obs_dict_device)
    
    # Synchronize if using CUDA
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Measure delay
    delays = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            result = model.predict_action(obs_dict_device)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            delay = (end_time - start_time) * 1000  # Convert to milliseconds
            delays.append(delay)
    
    mean_delay = np.mean(delays)
    std_delay = np.std(delays)
    
    return mean_delay, std_delay


def format_flops(flops: float) -> str:
    """Format FLOPs in human-readable format."""
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    else:
        return f"{flops:.2f} FLOPs"


def check_gs2_api_health(api_url: str) -> bool:
    """Check if GS2 API server is healthy."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("models_loaded", False)
        return False
    except Exception as e:
        print(f"  Warning: Failed to check API health: {e}")
        return False


def measure_gs2_flops(
    api_url: Optional[str] = None,
    image_size: tuple = (84, 84),
    text_prompt: str = "hammer.",
    gs2_root: Optional[pathlib.Path] = None,
    sam2_checkpoint: Optional[str] = None,
    sam2_config: Optional[str] = None,
    gdino_config: Optional[str] = None,
    gdino_checkpoint: Optional[str] = None,
    device: str = "cpu",
) -> Optional[float]:
    """Measure FLOPs for Grounded-SAM-2.
    
    If model paths are provided, loads models locally and measures FLOPs directly.
    Otherwise, returns None (FLOPs cannot be measured via API).
    
    Args:
        api_url: API URL (not used for local model loading, kept for compatibility)
        image_size: Image size (H, W)
        text_prompt: Text prompt for grounding
        gs2_root: Path to Grounded-SAM-2 root directory
        sam2_checkpoint: Path to SAM2 checkpoint
        sam2_config: Path to SAM2 config file
        gdino_config: Path to GroundingDINO config file
        gdino_checkpoint: Path to GroundingDINO checkpoint
        device: Device to use ('cpu' or 'cuda')
    
    Returns:
        Total FLOPs for GS2 inference, or None if models cannot be loaded
    """
    # If no model paths provided, cannot measure FLOPs
    if gs2_root is None:
        print("  Note: FLOPs measurement requires local model loading.")
        print("  Please provide model paths or use API mode (FLOPs not available via API).")
        return None
    
    try:
        # Add GS2 to path
        gs2_root = pathlib.Path(gs2_root).resolve()
        if str(gs2_root) not in sys.path:
            sys.path.insert(0, str(gs2_root))
        
        # Import GS2 modules
        from grounding_dino.groundingdino.util.inference import load_model as load_gdino_model, predict
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        import grounding_dino.groundingdino.datasets.transforms as T
        from PIL import Image
        
        # Set default paths if not provided
        if sam2_checkpoint is None:
            sam2_checkpoint = str(gs2_root / "checkpoints" / "sam2.1_hiera_large.pt")
        if sam2_config is None:
            sam2_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
        if gdino_config is None:
            gdino_config = str(gs2_root / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinB_cfg.py")
        if gdino_checkpoint is None:
            gdino_checkpoint = str(gs2_root / "gdino_checkpoints" / "groundingdino_swinb_cogcoor.pth")
        
        # Resolve paths
        if not os.path.isabs(sam2_checkpoint):
            sam2_checkpoint = str(gs2_root / sam2_checkpoint)
        if not os.path.isabs(gdino_checkpoint):
            gdino_checkpoint = str(gs2_root / gdino_checkpoint)
        if not os.path.isabs(gdino_config):
            gdino_config = str(gs2_root / gdino_config)
        
        print(f"  Loading models from local paths...")
        print(f"    SAM2 checkpoint: {sam2_checkpoint}")
        print(f"    SAM2 config: {sam2_config}")
        print(f"    GDINO config: {gdino_config}")
        print(f"    GDINO checkpoint: {gdino_checkpoint}")
        
        # Load models
        print("  Loading GroundingDINO...")
        grounding_model = load_gdino_model(
            model_config_path=gdino_config,
            model_checkpoint_path=gdino_checkpoint,
            device=device,
        )
        
        print("  Loading SAM2...")
        sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=device)
        sam2_predictor = SAM2ImagePredictor(sam2_model)
        
        print("  Models loaded successfully.")
        
        # Create dummy image
        H, W = image_size
        dummy_image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        image_pil = Image.fromarray(dummy_image)
        
        # Prepare transforms for GroundingDINO
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(image_pil, None)
        image_transformed = image_transformed.unsqueeze(0).to(device)  # Add batch dimension
        
        # Set models to eval mode
        grounding_model.eval()
        sam2_model.eval()
        
        # Create wrapper for FLOPs measurement
        # We need to measure both GroundingDINO and SAM2
        class GS2Wrapper(nn.Module):
            def __init__(self, gdino_model, sam2_predictor, image_source, text_prompt, device):
                super().__init__()
                self.gdino_model = gdino_model
                self.sam2_predictor = sam2_predictor
                self.image_source = image_source
                self.text_prompt = text_prompt
                self.device = device
                # Pre-set image for SAM2 (this is done once per image)
                self.sam2_predictor.set_image(image_source)
            
            def forward(self, image_tensor):
                # GroundingDINO forward
                boxes, confidences, labels = predict(
                    model=self.gdino_model,
                    image=image_tensor.squeeze(0),  # Remove batch dim for predict
                    caption=self.text_prompt,
                    box_threshold=0.35,
                    text_threshold=0.25,
                    device=self.device,
                )
                
                # Convert boxes for SAM2
                h, w, _ = self.image_source.shape
                if len(boxes) == 0:
                    # Return dummy output if no boxes
                    return torch.zeros(1, h, w, device=self.device)
                
                boxes = boxes * torch.Tensor([w, h, w, h]).to(self.device)
                from torchvision.ops import box_convert
                input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
                
                if input_boxes.ndim == 1:
                    input_boxes = input_boxes.reshape(1, -1)
                
                # SAM2 forward (using pre-set image)
                masks, scores, logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_boxes.cpu().numpy(),
                    multimask_output=False,
                )
                
                # Return first mask as tensor
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                mask_tensor = torch.from_numpy(masks[0]).to(self.device)
                return mask_tensor.unsqueeze(0)  # Add batch dimension
        
        print("  Measuring FLOPs...")
        wrapper = GS2Wrapper(grounding_model, sam2_predictor, dummy_image, text_prompt, device)
        wrapper.eval()
        wrapper.to(device)
        
        # Measure FLOPs
        try:
            if HAS_FLOP_COUNT_MODE:
                flops_dict, _ = flop_count(wrapper, (image_transformed,), mode=FlopCountMode.OPERATION_COUNT)
            else:
                flops_dict, _ = flop_count(wrapper, (image_transformed,))
            
            total_flops = sum(flops_dict.values())
            print(f"  Measured FLOPs: {format_flops(total_flops)}")
            return total_flops
            
        except Exception as e:
            print(f"  Warning: FLOPs measurement failed: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    except ImportError as e:
        print(f"  Error: Failed to import GS2 modules: {e}")
        print("  Make sure Grounded-SAM-2 is properly installed.")
        return None
    except Exception as e:
        print(f"  Error: Failed to load models or measure FLOPs: {e}")
        import traceback
        traceback.print_exc()
        return None


def measure_gs2_delay(
    api_url: str,
    image_size: tuple = (84, 84),
    text_prompt: str = "hammer.",
    num_warmup: int = 3,
    num_runs: int = 50,
) -> Tuple[Optional[float], Optional[float]]:
    """Measure delay for Grounded-SAM-2 inference via API."""
    try:
        from PIL import Image
        
        # Create dummy image
        H, W = image_size
        dummy_image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        
        # Save dummy image to temporary file for API
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            Image.fromarray(dummy_image).save(tmp_path)
        
        try:
            # Warmup
            print("  Warming up API calls...")
            for _ in range(num_warmup):
                payload = {
                    "image_path": tmp_path,
                    "text": text_prompt,
                    "box_thr": 0.35,
                    "text_thr": 0.25,
                    "multimask": False,
                    "seed": 0,
                }
                response = requests.post(f"{api_url}/infer", json=payload, timeout=60)
                if response.status_code != 200:
                    print(f"  Warning: API call failed during warmup: {response.status_code}")
            
            # Measure delay
            delays = []
            
            print(f"  Measuring delay for {num_runs} runs (1 image per run)...")
            for _ in range(num_runs):
                start_time = time.perf_counter()
                payload = {
                    "image_path": tmp_path,
                    "text": text_prompt,
                    "box_thr": 0.35,
                    "text_thr": 0.25,
                    "multimask": False,
                    "seed": 0,
                }
                response = requests.post(f"{api_url}/infer", json=payload, timeout=60)
                if response.status_code != 200:
                    print(f"  Warning: API call failed: {response.status_code}")
                    continue
                
                end_time = time.perf_counter()
                delay = (end_time - start_time) * 1000  # Convert to milliseconds
                delays.append(delay)
            
            if len(delays) == 0:
                return None, None
            
            mean_delay = np.mean(delays)
            std_delay = np.std(delays)
            
            return mean_delay, std_delay
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass
        
    except Exception as e:
        print(f"  Warning: Grounded-SAM-2 delay measurement failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Measure FLOPs and Delay for AEDP3 vs DP3")
    parser.add_argument(
        "--gs2_api_url",
        type=str,
        default="http://127.0.0.1:5000",
        help="Grounded-SAM-2 API server URL (default: http://127.0.0.1:5000). Used for delay measurement."
    )
    parser.add_argument(
        "--gs2_root",
        type=str,
        default=None,
        help="Path to Grounded-SAM-2 root directory. If provided, FLOPs will be measured using local models."
    )
    parser.add_argument(
        "--gs2_sam2_ckpt",
        type=str,
        default=None,
        help="Path to SAM2 checkpoint (default: checkpoints/sam2.1_hiera_large.pt relative to gs2_root)"
    )
    parser.add_argument(
        "--gs2_sam2_cfg",
        type=str,
        default=None,
        help="Path to SAM2 config (default: configs/sam2.1/sam2.1_hiera_l.yaml relative to gs2_root)"
    )
    parser.add_argument(
        "--gs2_gdino_cfg",
        type=str,
        default=None,
        help="Path to GroundingDINO config (default: grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py relative to gs2_root)"
    )
    parser.add_argument(
        "--gs2_gdino_ckpt",
        type=str,
        default=None,
        help="Path to GroundingDINO checkpoint (default: gdino_checkpoints/groundingdino_swinb_cogcoor.pth relative to gs2_root)"
    )
    parser.add_argument(
        "--gs2_device",
        type=str,
        default=None,
        help="Device for GS2 model loading ('cpu' or 'cuda', default: same as main device)"
    )
    args = parser.parse_args()
    
    root_dir = setup_root()

    config_dir = root_dir / "3D-Diffusion-Policy" / "diffusion_policy_3d" / "config"
    task_dir = config_dir / "task"
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gs2_device = args.gs2_device if args.gs2_device is not None else device
    
    # Resolve GS2 root if provided
    gs2_root = None
    if args.gs2_root:
        gs2_root = pathlib.Path(args.gs2_root).resolve()
        if not gs2_root.exists():
            print(f"Warning: GS2 root directory does not exist: {gs2_root}")
            gs2_root = None
    elif (root_dir / "Grounded-SAM-2").exists():
        # Try default location
        gs2_root = root_dir / "Grounded-SAM-2"
    
    # Check GS2 API health (for delay measurement)
    print("=" * 60)
    print("FLOPs and Delay Measurement for AEDP3 vs DP3")
    print("=" * 60)
    print(f"Using DP3 config dir: {config_dir}")
    print("  - AEDP3 (with attn):     dp3 + task=adroit_hammer")
    print("  - DP3 (without attn):    dp3 + task=adroit_hammer_no_attn")
    print(f"  - Grounded-SAM-2 API:    {args.gs2_api_url}")
    if gs2_root:
        print(f"  - Grounded-SAM-2 root:   {gs2_root} (local models for FLOPs)")
    else:
        print(f"  - Grounded-SAM-2:        Using API mode only (FLOPs not available)")
    print()
    
    print("Checking Grounded-SAM-2 API server...")
    if not check_gs2_api_health(args.gs2_api_url):
        print(f"Warning: Grounded-SAM-2 API server is not available at {args.gs2_api_url}")
        print("Delay measurement will fail, but FLOPs measurement can proceed if local models are available.")
        if not gs2_root:
            print("Please either:")
            print("  1. Start the API server: cd Grounded-SAM-2 && python gs2_api_server.py")
            print("  2. Provide --gs2_root for local model loading")
            sys.exit(1)
    else:
        print("  ✓ API server is healthy and models are loaded.\n")

    # Load configs
    aedp3_cfg = compose_dp3_cfg(config_dir, task_name="adroit_hammer")
    dp3_noattn_cfg = compose_dp3_cfg(config_dir, task_name="adroit_hammer_no_attn")

    # Load task shape_meta
    adroit_attn_task = OmegaConf.load(task_dir / "adroit_hammer.yaml")
    adroit_noattn_task = OmegaConf.load(task_dir / "adroit_hammer_no_attn.yaml")

    aedp3_cfg.shape_meta = adroit_attn_task.shape_meta
    aedp3_cfg.policy.shape_meta = adroit_attn_task.shape_meta

    dp3_noattn_cfg.shape_meta = adroit_noattn_task.shape_meta
    dp3_noattn_cfg.policy.shape_meta = adroit_noattn_task.shape_meta

    # Instantiate models
    print("Instantiating models...")
    aedp3_model = hydra.utils.instantiate(aedp3_cfg.policy)
    dp3_noattn_model = hydra.utils.instantiate(dp3_noattn_cfg.policy)
    print("Models instantiated.\n")

    # Get n_obs_steps from config
    n_obs_steps = aedp3_cfg.policy.get('n_obs_steps', 8)
    batch_size = 1
    
    # Create dummy observations
    print("Creating dummy observations...")
    aedp3_obs = create_dummy_obs(
        aedp3_cfg.shape_meta, 
        batch_size=batch_size, 
        n_obs_steps=n_obs_steps,
        device="cpu"
    )
    dp3_obs = create_dummy_obs(
        dp3_noattn_cfg.shape_meta,
        batch_size=batch_size,
        n_obs_steps=n_obs_steps,
        device="cpu"
    )
    print(f"  - Batch size: {batch_size}")
    print(f"  - Observation steps: {n_obs_steps}")
    print()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Measure FLOPs
    print("Measuring FLOPs...")
    print("-" * 60)
    
    print("Measuring AEDP3 FLOPs...")
    aedp3_flops = measure_model_flops(aedp3_model, aedp3_obs, device=device)
    
    print("Measuring DP3 FLOPs...")
    dp3_flops = measure_model_flops(dp3_noattn_model, dp3_obs, device=device)
    
    print("-" * 60)
    print()

    # Measure delay
    print("Measuring delay (inference time)...")
    print("-" * 60)
    
    print("Measuring AEDP3 delay...")
    aedp3_delay_mean, aedp3_delay_std = measure_model_delay(
        aedp3_model, aedp3_obs, num_warmup=3, num_runs=50, device=device
    )
    
    print("Measuring DP3 delay...")
    dp3_delay_mean, dp3_delay_std = measure_model_delay(
        dp3_noattn_model, dp3_obs, num_warmup=3, num_runs=50, device=device
    )
    
    print("-" * 60)
    print()

    # Measure Grounded-SAM-2
    print("Measuring Grounded-SAM-2 FLOPs and Delay...")
    print("-" * 60)
    
    print("Measuring Grounded-SAM-2 FLOPs...")
    gs2_flops = measure_gs2_flops(
        api_url=args.gs2_api_url,
        image_size=(84, 84),
        text_prompt="hammer.",
        gs2_root=gs2_root,
        sam2_checkpoint=args.gs2_sam2_ckpt,
        sam2_config=args.gs2_sam2_cfg,
        gdino_config=args.gs2_gdino_cfg,
        gdino_checkpoint=args.gs2_gdino_ckpt,
        device=gs2_device,
    )
    
    print("Measuring Grounded-SAM-2 delay...")
    gs2_delay_mean, gs2_delay_std = measure_gs2_delay(
        api_url=args.gs2_api_url,
        image_size=(84, 84),
        text_prompt="hammer.",
        num_warmup=3,
        num_runs=50,
    )
    
    print("-" * 60)
    print()

    # Print results
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()
    
    print("FLOPs:")
    print("-" * 60)
    if aedp3_flops is not None:
        print(f"AEDP3 (with attn):     {aedp3_flops:,.0f} ({format_flops(aedp3_flops)})")
    else:
        print("AEDP3 FLOPs:           Measurement failed")
    
    if dp3_flops is not None:
        print(f"DP3 (without attn):    {dp3_flops:,.0f} ({format_flops(dp3_flops)})")
    else:
        print("DP3 FLOPs:             Measurement failed")
    
    if gs2_flops is not None:
        print(f"Grounded-SAM-2:        {gs2_flops:,.0f} ({format_flops(gs2_flops)})")
    else:
        print("Grounded-SAM-2:        Not available (API mode)")
    
    if aedp3_flops is not None and dp3_flops is not None:
        diff = aedp3_flops - dp3_flops
        ratio = aedp3_flops / dp3_flops if dp3_flops > 0 else 0
        print(f"Difference (AEDP3-DP3): {diff:,.0f} ({format_flops(diff)})")
        print(f"Ratio (AEDP3/DP3):     {ratio:.3f}x")
    
    if gs2_flops is not None and aedp3_flops is not None:
        total_with_gs2 = aedp3_flops + gs2_flops
        print(f"AEDP3 + GS2:           {total_with_gs2:,.0f} ({format_flops(total_with_gs2)})")
        if dp3_flops is not None:
            ratio_with_gs2 = total_with_gs2 / dp3_flops if dp3_flops > 0 else 0
            print(f"Ratio ((AEDP3+GS2)/DP3): {ratio_with_gs2:.3f}x")
    
    print()
    print("Delay (inference time):")
    print("-" * 60)
    print(f"AEDP3 (with attn):     {aedp3_delay_mean:.2f} ± {aedp3_delay_std:.2f} ms")
    print(f"DP3 (without attn):    {dp3_delay_mean:.2f} ± {dp3_delay_std:.2f} ms")
    
    if gs2_delay_mean is not None:
        print(f"Grounded-SAM-2:        {gs2_delay_mean:.2f} ± {gs2_delay_std:.2f} ms")
    else:
        print("Grounded-SAM-2:        Measurement failed")
    
    delay_diff = aedp3_delay_mean - dp3_delay_mean
    delay_ratio = aedp3_delay_mean / dp3_delay_mean if dp3_delay_mean > 0 else 0
    print(f"Difference (AEDP3-DP3): {delay_diff:.2f} ms")
    print(f"Ratio (AEDP3/DP3):     {delay_ratio:.3f}x")
    
    if gs2_delay_mean is not None:
        total_delay_with_gs2 = aedp3_delay_mean + gs2_delay_mean
        print(f"AEDP3 + GS2:           {total_delay_with_gs2:.2f} ms")
        if dp3_delay_mean > 0:
            ratio_with_gs2 = total_delay_with_gs2 / dp3_delay_mean
            print(f"Ratio ((AEDP3+GS2)/DP3): {ratio_with_gs2:.3f}x")
    
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
