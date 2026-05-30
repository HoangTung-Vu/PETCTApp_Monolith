"""Standalone inference VRAM estimator for nnU-Net models.

External helper — NOT part of the nnunetv2 library. It reuses the library's
loaders (PlansManager, get_network_from_plans, determine_num_input_channels)
to rebuild a model's architecture from its plans.json, loads the real
checkpoint weights, runs a single patch-sized forward pass on the GPU and
reads ``torch.cuda.max_memory_allocated()``. The measured peak (weights +
activations) plus an estimate of the predictor's full-volume output buffers is
returned in GB so the engine pool can decide how many workers fit in VRAM.
"""

import gc
import logging
from pathlib import Path

import torch

logger = logging.getLogger("engine.vram")


def _find_checkpoint(model_folder: Path) -> Path:
    """Return a usable checkpoint path under any ``fold_*`` directory."""
    for name in ("checkpoint_best.pth", "checkpoint_final.pth"):
        for fold in sorted(model_folder.glob("fold_*")):
            ckpt = fold / name
            if ckpt.exists():
                return ckpt
    raise FileNotFoundError(f"No checkpoint (checkpoint_best/final.pth) found under {model_folder}")


def estimate_vram_usage_inference_gb(model_folder: str, device: torch.device) -> dict:
    """Estimate per-inference VRAM (GB) for the model in ``model_folder``.

    Builds the network from ``plans.json``, loads the checkpoint weights, runs
    one ``patch_size`` forward pass under ``inference_mode`` + autocast on
    ``device`` and measures peak allocation. Adds an approximation of the
    predictor's full-volume logit/gaussian/count buffers (kept on GPU when
    ``perform_everything_on_gpu=True``). GPU-only; callers handle the CPU case.
    """
    # Library imports are local so importing this module never forces nnunetv2.
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    model_folder = Path(model_folder)
    plans = load_json(str(model_folder / "plans.json"))
    dataset_json = load_json(str(model_folder / "dataset.json"))
    plans_manager = PlansManager(plans)

    ckpt_path = _find_checkpoint(model_folder)
    logger.debug("Loading checkpoint for estimate: %s", ckpt_path)
    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Configuration name comes from the checkpoint; fall back to the folder suffix.
    config_name = (checkpoint.get("init_args", {}) or {}).get("configuration") \
        or model_folder.name.split("__")[-1]
    config = plans_manager.get_configuration(config_name)

    num_input_channels = determine_num_input_channels(plans_manager, config, dataset_json)
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_seg_heads = label_manager.num_segmentation_heads
    patch_size = tuple(int(x) for x in config.patch_size)
    logger.debug("config=%s patch=%s in_ch=%d seg_heads=%d",
                 config_name, patch_size, num_input_channels, num_seg_heads)

    # Full-volume buffers the predictor allocates on GPU (logits + gaussian +
    # n_predictions), approximated from the dataset median image size in fp16.
    median_shape = config.configuration.get("median_image_size_in_voxels") \
        or plans.get("original_median_shape_after_transp") or patch_size
    n_vox = 1
    for s in median_shape:
        n_vox *= int(round(float(s)))
    buffers_bytes = n_vox * num_seg_heads * 2 * 3  # 3 fp16 full-volume buffers

    net = get_network_from_plans(plans_manager, dataset_json, config,
                                 num_input_channels, deep_supervision=False)
    try:
        net.load_state_dict(checkpoint["network_weights"])
    except Exception:
        logger.warning("Could not load checkpoint weights for estimate; using "
                       "freshly-initialized weights (GPU memory footprint is identical).")
    net = net.to(device).eval()

    patch_peak = 0
    try:
        torch.cuda.reset_peak_memory_stats(device)
        dummy = torch.zeros((1, num_input_channels, *patch_size), device=device)
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            net(dummy)
        torch.cuda.synchronize(device)
        patch_peak = int(torch.cuda.max_memory_allocated(device))
        del dummy
    finally:
        del net
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    estimate_bytes = patch_peak + buffers_bytes
    result = {
        "estimate_gb": estimate_bytes / 1e9,
        "patch_peak_gb": patch_peak / 1e9,
        "buffers_gb": buffers_bytes / 1e9,
        "patch_size": patch_size,
        "num_input_channels": num_input_channels,
        "num_seg_heads": num_seg_heads,
        "config_name": config_name,
    }
    logger.info(
        "VRAM estimate for '%s' (%s): %.2f GB (patch_peak=%.2f, buffers=%.2f) "
        "| patch=%s in_ch=%d seg_heads=%d",
        model_folder.name, config_name, result["estimate_gb"], result["patch_peak_gb"],
        result["buffers_gb"], patch_size, num_input_channels, num_seg_heads,
    )
    return result
