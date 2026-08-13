"""Single-node MiniMax H3 optimization profile for NVIDIA V100 / SM70."""

import logging

import torch

from .flash_attention import V100FlashAttention
from .h3_mixed_precision import H3V100MixedPrecision
from .h3_prefetch_guard import H3RuntimePrefetchGuard
from .tokenwise_chunking import H3TokenwiseMLPChunking


LOGGER = logging.getLogger("H3_V100")


def _diffusion_model(model):
    try:
        return model.get_model_object("diffusion_model")
    except Exception:
        return None


def _is_supported_h3(diffusion_model):
    model_type = type(diffusion_model)
    if not (
        model_type.__module__ == "comfy.ldm.minimax.model"
        and model_type.__name__ == "MiniMaxH3Model"
        and getattr(diffusion_model, "hidden_size", None) == 5376
        and hasattr(diffusion_model, "token_refiner")
        and hasattr(diffusion_model, "rope")
        and hasattr(diffusion_model, "video_patch_proj")
        and hasattr(diffusion_model, "audio_patch_proj")
    ):
        return False
    blocks = getattr(diffusion_model, "blocks", None)
    if not blocks or len(blocks) != 50:
        return False
    return all(
        getattr(getattr(block, "attn", None), "head_dim", None) == 128
        and getattr(getattr(block, "attn", None), "heads", None) == 56
        and all(
            hasattr(block.attn, name)
            for name in ("qkv_proj", "q_norm", "k_norm", "out_proj")
        )
        for block in blocks
    )


class H3V100:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mixed_precision": ("BOOLEAN", {"default": True}),
                "flash_attention": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "H3_V100"
    DESCRIPTION = (
        "MiniMax H3 inference optimizations for NVIDIA V100: optional mixed "
        "precision and shape-benchmarked SM70 Flash Attention. Adaptive "
        "bounded-memory execution is always enabled."
    )

    def patch(
        self, model, mixed_precision=True, flash_attention=True,
    ):
        if not torch.cuda.is_available() or not any(
            torch.cuda.get_device_capability(index) == (7, 0)
            for index in range(torch.cuda.device_count())
        ):
            raise RuntimeError("H3_V100 requires an NVIDIA SM70/V100 CUDA device.")

        diffusion_model = _diffusion_model(model)
        if not _is_supported_h3(diffusion_model):
            raise RuntimeError(
                "H3_V100 rejected this MODEL: the exact supported MiniMax H3 "
                "50-block layout was not detected."
            )
        optimized = model
        if mixed_precision:
            optimized, = H3V100MixedPrecision().patch(
                optimized, enabled=True, v100_only=True
            )
        optimized, = H3TokenwiseMLPChunking().patch(
            optimized,
            adaptive=True,
            chunk_tokens=512,
            cache_trim=True,
            experimental_fp16=bool(mixed_precision),
        )
        if flash_attention:
            optimized, = V100FlashAttention().patch(
                optimized, enabled=True, min_tokens=1024, auto_benchmark=True
            )
        optimized = optimized.clone()
        transformer_options = optimized.model_options.setdefault(
            "transformer_options", {}
        )
        transformer_options["prefetch_dynamic_vbars"] = False
        optimized, = H3RuntimePrefetchGuard().patch(
            optimized,
            enabled=False,
            adaptive_memory=True,
            experimental_fp16=bool(mixed_precision),
        )
        LOGGER.info(
            "H3_V100 active: mixed_precision=%s, flash_attention=%s, "
            "adaptive_memory=True, fp16_linear=%s.",
            bool(mixed_precision), bool(flash_attention), bool(mixed_precision),
        )
        return (optimized,)
