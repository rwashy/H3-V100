"""Workflow-scoped exact Flash/Sol dispatcher for the stable V100 profile."""

import logging

import torch

from . import backend
from . import sol_attention


LOGGER = logging.getLogger("V100FlashAttention")
ENABLED_KEY = "v100_flash_attention_enabled"
MIN_TOKENS_KEY = "v100_flash_attention_min_tokens"


def v100_attention_override(
    original,
    q,
    k,
    v,
    heads,
    mask=None,
    attn_precision=None,
    skip_reshape=False,
    skip_output_reshape=False,
    **kwargs,
):
    transformer_options = kwargs.get("transformer_options") or {}
    enabled = bool(transformer_options.get(ENABLED_KEY, True))
    min_tokens = int(transformer_options.get(MIN_TOKENS_KEY, 1024))
    attention_mode = transformer_options.get(
        sol_attention.MODE_KEY, sol_attention.MODE_FLASH
    )

    flash_reasons = list(
        backend.support_reasons(
            q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs
        )
    )
    if q.shape[-2] < min_tokens or k.shape[-2] < min_tokens:
        flash_reasons.append("below-min-tokens")
    if not enabled:
        flash_reasons.append("disabled")
    flash_eligible = not flash_reasons

    sol_allowed = bool(
        transformer_options.get("v100_sol_attention_h3_enabled", False)
    )
    if attention_mode == sol_attention.MODE_SOL and sol_allowed:
        sol_reasons = list(
            sol_attention.support_reasons(
                q, k, v, heads, mask, skip_reshape,
                skip_output_reshape, kwargs,
            )
        )
        sol_min_tokens = int(
            transformer_options.get(sol_attention.MIN_TOKENS_KEY, 16_384)
        )
        if int(q.shape[-2]) < sol_min_tokens:
            sol_reasons.append("below-sol-min-tokens")
        block_index = transformer_options.get(sol_attention.BLOCK_INDEX_KEY)
        block_count = transformer_options.get(sol_attention.BLOCK_COUNT_KEY)
        if block_index == 0 or (
            block_count is not None and block_index == int(block_count) - 1
        ):
            sol_reasons.append("dense-edge-block")
        sigmas = transformer_options.get("sigmas")
        if sigmas is not None and len(sigmas):
            sigma = float(sigmas[0])
            sigma_start = transformer_options.get(sol_attention.SIGMA_START_KEY)
            sigma_end = transformer_options.get(sol_attention.SIGMA_END_KEY)
            if sigma_start is not None and sigma > float(sigma_start):
                sol_reasons.append("before-sol-schedule")
            if sigma_end is not None and sigma < float(sigma_end):
                sol_reasons.append("after-sol-schedule")
        if not sol_reasons:
            return sol_attention.run_reference(
                q,
                k,
                v,
                transformer_options,
                exact_backend=lambda: backend.comfy_attention(
                    q, k, v, heads, scale=kwargs.get("scale")
                ),
                mode=sol_attention.MODE_SOL,
            )

    if flash_eligible:
        return backend.comfy_attention(q, k, v, heads, scale=kwargs.get("scale"))

    return original(
        q,
        k,
        v,
        heads,
        mask=mask,
        attn_precision=attn_precision,
        skip_reshape=skip_reshape,
        skip_output_reshape=skip_output_reshape,
        **kwargs,
    )


v100_attention_override._v100_flash_attention_override = True


class V100FlashAttention:
    def patch(
        self,
        model,
        enabled=True,
        min_tokens=1024,
        attention_mode=sol_attention.MODE_FLASH,
        sol_tau=1.0,
        sol_min_tokens=16_384,
        sol_block_size=64,
        allow_sol=False,
        sol_start_percent=0.2,
        sol_end_percent=0.8,
    ):
        if not enabled:
            return (model,)
        if not torch.cuda.is_available() or not any(
            torch.cuda.get_device_capability(index) == (7, 0)
            for index in range(torch.cuda.device_count())
        ):
            raise RuntimeError("V100 FlashAttention requires an SM70 CUDA device.")

        backend.load_extension()
        patched = model.clone()
        transformer_options = patched.model_options.setdefault(
            "transformer_options", {}
        )
        existing = transformer_options.get("optimized_attention_override")
        if existing is not None and not getattr(
            existing, "_v100_flash_attention_override", False
        ):
            raise RuntimeError(
                "Another optimized_attention_override is already active on this MODEL."
            )

        transformer_options["optimized_attention_override"] = v100_attention_override
        transformer_options[ENABLED_KEY] = True
        transformer_options[MIN_TOKENS_KEY] = int(min_tokens)
        transformer_options[sol_attention.MODE_KEY] = str(attention_mode)
        transformer_options[sol_attention.TAU_KEY] = float(sol_tau)
        transformer_options[sol_attention.MIN_TOKENS_KEY] = int(sol_min_tokens)
        transformer_options[sol_attention.BLOCK_SIZE_KEY] = int(sol_block_size)
        transformer_options[sol_attention.PROBE_KEY] = False
        if allow_sol:
            sampling = patched.get_model_object("model_sampling")
            transformer_options[sol_attention.SIGMA_START_KEY] = float(
                sampling.percent_to_sigma(float(sol_start_percent))
            )
            transformer_options[sol_attention.SIGMA_END_KEY] = float(
                sampling.percent_to_sigma(float(sol_end_percent))
            )
        transformer_options["v100_sol_attention_h3_enabled"] = bool(allow_sol)
        LOGGER.info(
            "V100 attention active: mode=%s, flash_min_tokens=%d, "
            "sol_allowed=%s, sol_min_tokens=%d, sol_tau=%.3f, "
            "sol_block_size=%d.",
            attention_mode,
            int(min_tokens),
            bool(allow_sol),
            int(sol_min_tokens),
            float(sol_tau),
            int(sol_block_size),
        )
        return (patched,)
