import logging

import torch

from . import backend


LOGGER = logging.getLogger("V100FlashAttention")
ENABLED_KEY = "v100_flash_attention_enabled"
MIN_TOKENS_KEY = "v100_flash_attention_min_tokens"
AUTO_BENCHMARK_KEY = "v100_flash_attention_auto_benchmark"
AUTO_BENCHMARK_MAX_TOKENS = 38_000
_decision_cache = {}


def _decision_key(q, k, v, heads):
    return (
        q.device.type,
        q.device.index,
        q.dtype,
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        int(heads),
    )


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
    enabled = transformer_options.get(ENABLED_KEY, True)
    min_tokens = transformer_options.get(MIN_TOKENS_KEY, 1024)
    reasons = list(backend.support_reasons(
        q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs
    ))
    if q.shape[-2] < min_tokens or k.shape[-2] < min_tokens:
        reasons.append("below-min-tokens")
    if not enabled:
        reasons.append("disabled")
    eligible = not reasons
    if eligible:
        if q.shape[-2] > AUTO_BENCHMARK_MAX_TOKENS:
            key = _decision_key(q, k, v, heads)
            first_forced = _decision_cache.get(key) is not True
            _decision_cache[key] = True
            if first_forced:
                LOGGER.info(
                    "V100 attention auto-benchmark bypassed for long sequence: "
                    "tokens=%d, threshold=%d; selected V100 to avoid duplicate "
                    "attention outputs.",
                    int(q.shape[-2]), AUTO_BENCHMARK_MAX_TOKENS,
                )
            return backend.comfy_attention(
                q, k, v, heads, scale=kwargs.get("scale")
            )
        if not transformer_options.get(AUTO_BENCHMARK_KEY, True):
            return backend.comfy_attention(q, k, v, heads, scale=kwargs.get("scale"))

        key = _decision_key(q, k, v, heads)
        decision = _decision_cache.get(key)
        if decision is None:
            candidate = backend.comfy_attention(q, k, v, heads, scale=kwargs.get("scale"))
            original_out = original(
                q, k, v, heads, mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            backend.comfy_attention(q, k, v, heads, scale=kwargs.get("scale"))
            end.record()
            end.synchronize()
            v100_ms = start.elapsed_time(end)
            start.record()
            original(
                q, k, v, heads, mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                **kwargs,
            )
            end.record()
            end.synchronize()
            original_ms = start.elapsed_time(end)
            decision = v100_ms < original_ms * 0.98
            _decision_cache[key] = decision
            LOGGER.info(
                "V100 attention auto-benchmark: shape=%s, heads=%d, V100=%.3f ms, "
                "ComfyUI=%.3f ms; selected %s.",
                tuple(q.shape), int(heads), v100_ms, original_ms,
                "V100" if decision else "ComfyUI",
            )
            return candidate if decision else original_out
        if decision:
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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "min_tokens": (
                    "INT",
                    {"default": 1024, "min": 128, "max": 65536, "step": 128},
                ),
                "auto_benchmark": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "V100"
    DESCRIPTION = (
        "Workflow-scoped SM70 attention for FP16, unmasked, non-causal, "
        "head_dim=128 inference. Unsupported calls fall back to ComfyUI's backend."
    )

    def patch(
        self, model, enabled=True, min_tokens=1024, auto_benchmark=True,
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
        transformer_options = patched.model_options.setdefault("transformer_options", {})
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
        transformer_options[AUTO_BENCHMARK_KEY] = bool(auto_benchmark)
        LOGGER.info(
            "Workflow-scoped V100 FlashAttention active (min_tokens=%d, auto_benchmark=%s).",
            int(min_tokens), bool(auto_benchmark),
        )
        return (patched,)
