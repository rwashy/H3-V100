"""Validated MiniMax H3 optimization profile for NVIDIA V100 / SM70."""

import logging

import torch

from .flash_attention import V100FlashAttention
from .h3_mixed_precision import H3V100MixedPrecision
from .h3_prefetch_guard import H3RuntimePrefetchGuard
from .sol_attention import MODE_FLASH, MODE_SOL
from .tokenwise_chunking import H3TokenwiseMLPChunking


LOGGER = logging.getLogger("H3V100Optimize")

FLASH_MIN_TOKENS = 1024
SOL_MIN_TOKENS = 4096
SOL_BLOCK_SIZE = 64
MLP_CHUNK_TOKENS = 640
CACHE_TRIM_THRESHOLD_MIB = 2048


def _diffusion_model(model):
    try:
        return model.get_model_object("diffusion_model")
    except Exception:
        return None


def _is_h3(diffusion_model):
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
    for block in blocks:
        attention = getattr(block, "attn", None)
        if not (
            attention is not None
            and getattr(attention, "head_dim", None) == 128
            and getattr(attention, "heads", None) == 56
            and all(
                hasattr(attention, name)
                for name in ("qkv_proj", "q_norm", "k_norm", "out_proj", "heads")
            )
        ):
            return False
    return True


class H3V100Optimize:
    """Apply the frozen H3 V100 precision, attention, and memory policy."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mixed_precision": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "attention_backend": (
                    (MODE_FLASH, MODE_SOL),
                    {"default": MODE_FLASH},
                ),
                "sol_tau": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05},
                ),
                "sol_start_percent": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "sol_end_percent": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/V100"
    DESCRIPTION = (
        "MiniMax H3 only. Choose whether to enable validated V100 mixed "
        "precision, then choose exact Flash Attention or the "
        "validated Flash-Sol-Flash long-sequence schedule."
    )

    def patch(
        self,
        model,
        mixed_precision=True,
        attention_backend=MODE_FLASH,
        sol_tau=1.0,
        sol_start_percent=0.2,
        sol_end_percent=0.8,
    ):
        if attention_backend not in (MODE_FLASH, MODE_SOL):
            raise ValueError(
                f"attention_backend must be {MODE_FLASH!r} or {MODE_SOL!r}; "
                f"received {attention_backend!r}."
            )
        fp16_mlp = bool(mixed_precision)
        if not 0.0 <= float(sol_start_percent) <= 1.0:
            raise ValueError("sol_start_percent must be between 0 and 1.")
        if not 0.0 <= float(sol_end_percent) <= 1.0:
            raise ValueError("sol_end_percent must be between 0 and 1.")
        if float(sol_start_percent) >= float(sol_end_percent):
            raise ValueError("sol_start_percent must be smaller than sol_end_percent.")
        if not 0.0 <= float(sol_tau) <= 4.0:
            raise ValueError("sol_tau must be between 0 and 4.")
        if not torch.cuda.is_available() or not any(
            torch.cuda.get_device_capability(index) == (7, 0)
            for index in range(torch.cuda.device_count())
        ):
            raise RuntimeError("H3 V100 Optimize requires an SM70 CUDA device.")

        diffusion_model = _diffusion_model(model)
        if not _is_h3(diffusion_model):
            model_type = type(diffusion_model)
            raise RuntimeError(
                "H3 V100 Optimize accepts only the validated MiniMax H3 model; "
                f"received {model_type.__module__}.{model_type.__name__}."
            )

        optimized, = H3V100MixedPrecision().patch(
            model, enabled=bool(mixed_precision), v100_only=True
        )
        optimized, = H3TokenwiseMLPChunking().patch(
            optimized,
            adaptive=True,
            chunk_tokens=MLP_CHUNK_TOKENS,
            cache_trim=True,
            cache_trim_threshold_mb=CACHE_TRIM_THRESHOLD_MIB,
            experimental_fp16=fp16_mlp,
        )
        optimized, = V100FlashAttention().patch(
            optimized,
            enabled=True,
            min_tokens=FLASH_MIN_TOKENS,
            auto_benchmark=False,
            attention_mode=attention_backend,
            sol_tau=float(sol_tau),
            sol_min_tokens=SOL_MIN_TOKENS,
            sol_block_size=SOL_BLOCK_SIZE,
            sol_probe=False,
            sol_start_percent=float(sol_start_percent),
            sol_end_percent=float(sol_end_percent),
            # Flash is a strict exact-attention path. Do not publish Sol
            # eligibility state when Sol was not selected.
            allow_sol=attention_backend == MODE_SOL,
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
            experimental_fp16=fp16_mlp,
        )
        LOGGER.info(
            "H3 V100 Optimize active: attention_backend=%s, mixed_precision=%s, "
            "adaptive_memory=True, sol_tau=%.3f, sol_min_tokens=%d, "
            "sol_block_size=%d, sol_start_percent=%.3f, sol_end_percent=%.3f.",
            attention_backend,
            bool(mixed_precision),
            float(sol_tau),
            SOL_MIN_TOKENS,
            SOL_BLOCK_SIZE,
            float(sol_start_percent),
            float(sol_end_percent),
        )
        return (optimized,)
