import os
import sys
import pathlib
from typing import Optional

import torch
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
import hydra


def setup_root():
    """
    Mimic the behavior in 3D-Diffusion-Policy/train.py:
    add project root to sys.path and chdir there,
    so Hydra instantiation can find the modules.
    """
    this_file = pathlib.Path(__file__).resolve()
    # project root is aedp3/
    root_dir = this_file.parent.parent
    # IMPORTANT: put local 3D-Diffusion-Policy at the FRONT of sys.path,
    # so it takes precedence over any pip-installed diffusion_policy_3d.
    local_dp3_root = str(root_dir / "3D-Diffusion-Policy")
    if local_dp3_root not in sys.path:
        sys.path.insert(0, local_dp3_root)
    os.chdir(root_dir)
    return root_dir


def load_cfg(cfg_path: str, task_override: Optional[str] = None):
    """
    Load a DP3 config YAML and manually resolve its `defaults: - task: ...`
    to attach the corresponding task config, similar to how Hydra would compose it.

    If `task_override` is given (e.g. "adroit_hammer_no_attn"), use that task
    instead of the one specified in `defaults`.
    """
    cfg_path = pathlib.Path(cfg_path).resolve()
    cfg_dir = cfg_path.parent

    # Register eval resolver used in task configs (e.g. ${eval:'${n_obs_steps}-1'})
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    # Register a dummy "now" resolver to support keys like ${now:%Y.%m.%d}
    # which are only used for logging/paths and don't affect model structure.
    if not OmegaConf.has_resolver("now"):
        import datetime

        OmegaConf.register_new_resolver(
            "now",
            lambda fmt=None: datetime.datetime.now().strftime(fmt) if fmt else datetime.datetime.now().isoformat(),
            replace=True,
        )

    cfg = OmegaConf.load(cfg_path)

    # Resolve task default: defaults: - task: adroit_hammer
    defaults = cfg.get("defaults", [])
    task_name = task_override
    if task_name is None:
        for item in defaults:
            if isinstance(item, dict) and "task" in item:
                task_name = item["task"]
                break
    if task_name is None:
        raise ValueError(f"No `task` found in defaults of {cfg_path} and no override provided")

    task_cfg_path = cfg_dir / "task" / f"{task_name}.yaml"
    if not task_cfg_path.is_file():
        raise FileNotFoundError(f"Task config not found: {task_cfg_path}")

    task_cfg = OmegaConf.load(task_cfg_path)
    cfg.task = task_cfg

    # We don't need Hydra runtime fields for counting params.
    # Drop them to avoid interpolation errors like ${hydra.job.num}.
    for key in ["hydra", "multi_run"]:
        if key in cfg:
            cfg.pop(key)

    # Some fields in the main cfg reference ${task.shape_meta}
    # Let OmegaConf resolve all remaining interpolations.
    OmegaConf.resolve(cfg)
    return cfg


def compose_dp3_cfg(config_dir: pathlib.Path, task_name: str):
    """
    Use Hydra's official compose API to get the same config as training,
    with a specific task override (e.g., adroit_hammer or adroit_hammer_no_attn).
    """
    # Register resolvers used in configs
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


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    root_dir = setup_root()

    config_dir = root_dir / "3D-Diffusion-Policy" / "diffusion_policy_3d" / "config"
    task_dir = config_dir / "task"

    print(f"Using DP3 config dir: {config_dir}")
    print("  - with attn (aedp3):       dp3 + task=adroit_hammer")
    print("  - without attn (dp3 base): dp3 + task=adroit_hammer_no_attn")

    # aedp3: DP3 + 3D attention (task has attn_3d obs)
    aedp3_cfg = compose_dp3_cfg(config_dir, task_name="adroit_hammer")
    # dp3 base: same DP3 architecture but without attn_3d obs
    dp3_noattn_cfg = compose_dp3_cfg(config_dir, task_name="adroit_hammer_no_attn")

    # To make sure encoder sees the correct obs (including attn_3d for aedp3),
    # explicitly load task shape_meta and assign.
    adroit_attn_task = OmegaConf.load(task_dir / "adroit_hammer.yaml")
    adroit_noattn_task = OmegaConf.load(task_dir / "adroit_hammer_no_attn.yaml")

    aedp3_cfg.shape_meta = adroit_attn_task.shape_meta
    aedp3_cfg.policy.shape_meta = adroit_attn_task.shape_meta

    dp3_noattn_cfg.shape_meta = adroit_noattn_task.shape_meta
    dp3_noattn_cfg.policy.shape_meta = adroit_noattn_task.shape_meta

    # Debug: print shape_meta to verify attn_3d is present only in aedp3
    print("---------- aedp3 shape_meta ----------")
    print(OmegaConf.to_yaml(aedp3_cfg.shape_meta))
    print("---------- aedp3 policy.shape_meta ----------")
    print(OmegaConf.to_yaml(aedp3_cfg.policy.shape_meta))
    print("---------- dp3_noattn shape_meta ----------")
    print(OmegaConf.to_yaml(dp3_noattn_cfg.shape_meta))
    print("---------- dp3_noattn policy.shape_meta ----------")
    print(OmegaConf.to_yaml(dp3_noattn_cfg.policy.shape_meta))

    # Instantiate models via Hydra configs (only policy is needed)
    aedp3_model = hydra.utils.instantiate(aedp3_cfg.policy)
    dp3_noattn_model = hydra.utils.instantiate(dp3_noattn_cfg.policy)

    # Extra debug: inspect obs_encoder flags
    print("===== DEBUG: aedp3 obs_encoder =====")
    enc = aedp3_model.obs_encoder
    print("use_attn_3d:", getattr(enc, "use_attn_3d", None))
    print("attn_3d_shape:", getattr(enc, "attn_3d_shape", None))
    print("fusion_strategy:", getattr(enc, "fusion_strategy", None))
    print("n_output_channels:", getattr(enc, "n_output_channels", None))
    print("has attn_3d_encoder:", hasattr(enc, "attn_3d_encoder") and enc.attn_3d_encoder is not None)

    print("===== DEBUG: dp3_noattn obs_encoder =====")
    enc2 = dp3_noattn_model.obs_encoder
    print("use_attn_3d:", getattr(enc2, "use_attn_3d", None))
    print("attn_3d_shape:", getattr(enc2, "attn_3d_shape", None))
    print("fusion_strategy:", getattr(enc2, "fusion_strategy", None))
    print("n_output_channels:", getattr(enc2, "n_output_channels", None))
    print("has attn_3d_encoder:", hasattr(enc2, "attn_3d_encoder") and enc2.attn_3d_encoder is not None)

    aedp3_params = count_params(aedp3_model)
    dp3_noattn_params = count_params(dp3_noattn_model)
    diff = aedp3_params - dp3_noattn_params

    print("======================================")
    print(f"AEDP3 (DP3 + attn) params:          {aedp3_params:,}")
    print(f"DP3 (without attn) params:          {dp3_noattn_params:,}")
    print(f"Difference (AEDP3 - DP3 no attn):   {diff:,}")
    print("======================================")


if __name__ == "__main__":
    main()

