import logging

import torch

from . import backend
from . import sol_attention
from .diagnostics import (
    DIAGNOSTICS_INTERVAL_KEY,
    DIAGNOSTICS_KEY,
    FLASH_BENCHMARK_OCCURRED_KEY,
)


LOGGER = logging.getLogger("V100FlashAttention")
ENABLED_KEY = "v100_flash_attention_enabled"
MIN_TOKENS_KEY = "v100_flash_attention_min_tokens"
AUTO_BENCHMARK_KEY = "v100_flash_attention_auto_benchmark"
DIAGNOSTICS_ONLY_KEY = "v100_flash_attention_diagnostics_only"
AUTO_BENCHMARK_MAX_TOKENS = 38_000
_decision_cache = {}
_diagnostic_seen = set()
_diagnostic_stats = {}


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


def _record_diagnostic(
    key, q, selected, eligible, reasons, decision_state, interval
):
    stats = _diagnostic_stats.setdefault(
        key, {"calls": 0, "v100": 0, "sol": 0, "comfyui": 0, "eligible": 0}
    )
    stats["calls"] += 1
    stats[selected.lower()] += 1
    stats["eligible"] += int(bool(eligible))
    if stats["calls"] == 1 or (interval > 0 and stats["calls"] % interval == 0):
        allocated = torch.cuda.memory_allocated(q.device) / (1024 ** 2) if q.is_cuda else 0
        reserved = torch.cuda.memory_reserved(q.device) / (1024 ** 2) if q.is_cuda else 0
        peak = torch.cuda.max_memory_allocated(q.device) / (1024 ** 2) if q.is_cuda else 0
        LOGGER.info(
            "V100 diagnostics attention summary: calls=%d, eligible=%d, V100=%d, "
            "Sol=%d, ComfyUI=%d, current=%s, decision=%s, reasons=%s, "
            "memory_allocated=%.1f MiB, memory_reserved=%.1f MiB, peak=%.1f MiB.",
            stats["calls"], stats["eligible"], stats["v100"], stats["sol"],
            stats["comfyui"],
            selected, decision_state, reasons or "none", allocated, reserved, peak,
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
    diagnostics = transformer_options.get(DIAGNOSTICS_KEY, False)
    diagnostics_only = transformer_options.get(DIAGNOSTICS_ONLY_KEY, False)
    diagnostics_interval = max(
        0, int(transformer_options.get(DIAGNOSTICS_INTERVAL_KEY, 50))
    )
    enabled = transformer_options.get(ENABLED_KEY, True)
    min_tokens = transformer_options.get(MIN_TOKENS_KEY, 1024)
    attention_mode = transformer_options.get(
        sol_attention.MODE_KEY, sol_attention.MODE_FLASH
    )
    sol_min_tokens = int(
        transformer_options.get(sol_attention.MIN_TOKENS_KEY, 16_384)
    )
    sol_allowed = bool(
        transformer_options.get("v100_sol_attention_h3_enabled", False)
    )
    reasons = list(backend.support_reasons(
        q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs
    ))
    if q.shape[-2] < min_tokens or k.shape[-2] < min_tokens:
        reasons.append("below-min-tokens")
    if not enabled:
        reasons.append("disabled")
    eligible = not reasons
    diagnostic_key = _decision_key(q, k, v, heads)
    if diagnostics and diagnostic_key not in _diagnostic_seen:
        _diagnostic_seen.add(diagnostic_key)
        LOGGER.info(
            "V100 diagnostics attention shape: q=%s, k=%s, v=%s, dtype=%s, "
            "heads=%d, mask=%s, min_tokens=%d, eligible=%s, layout=%s, "
            "attention_backend=%s, sol_min_tokens=%d, sol_allowed=%s.",
            tuple(q.shape), tuple(k.shape), tuple(v.shape), q.dtype, int(heads),
            "none" if mask is None else tuple(mask.shape), int(min_tokens),
            bool(eligible), "explicit-head" if skip_reshape else "standard",
            attention_mode, sol_min_tokens, sol_allowed,
        )
    if diagnostics_only:
        _record_diagnostic(
            diagnostic_key, q, "ComfyUI", eligible, tuple(reasons), "observe-only",
            diagnostics_interval,
        )
        return original(
            q, k, v, heads, mask=mask, attn_precision=attn_precision,
            skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
            **kwargs,
        )
    # Flash mode never dispatches to Sol. Skip Sol-only capability checks.
    if attention_mode != sol_attention.MODE_SOL or not sol_allowed:
        sol_reasons = ["sol-disabled"]
    else:
        sol_reasons = list(sol_attention.support_reasons(
            q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs
        ))
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
    sol_eligible = not sol_reasons
    if attention_mode == sol_attention.MODE_SOL and sol_eligible:
        if diagnostics:
            _record_diagnostic(
                diagnostic_key, q, "Sol", True, (),
                "reference-forced",
                diagnostics_interval,
            )
        return sol_attention.run_reference(
            q, k, v, transformer_options,
            exact_backend=lambda: backend.comfy_attention(
                q, k, v, heads, scale=kwargs.get("scale")
            ),
            mode=sol_attention.MODE_SOL,
        )
    if diagnostics and attention_mode == sol_attention.MODE_SOL:
        LOGGER.info(
            "V100 diagnostics Sol-Attn fallback: sequence=%d, block=%s, reasons=%s; "
            "using exact Flash Attention when eligible.",
            int(q.shape[-2]), block_index if block_index is not None else "unknown",
            tuple(sol_reasons),
        )
    if eligible:
        if q.shape[-2] > AUTO_BENCHMARK_MAX_TOKENS:
            key = _decision_key(q, k, v, heads)
            first_forced = _decision_cache.get(key) is not True
            _decision_cache[key] = True
            if first_forced:
                LOGGER.info(
                    "V100 exact Flash selected for long sequence: tokens=%d, "
                    "threshold=%d; duplicate attention outputs are disabled.",
                    int(q.shape[-2]), AUTO_BENCHMARK_MAX_TOKENS,
                )
            if diagnostics:
                _record_diagnostic(
                    diagnostic_key, q, "V100", True, (), "forced-long",
                    diagnostics_interval,
                )
            return backend.comfy_attention(
                q, k, v, heads, scale=kwargs.get("scale")
            )
        if not transformer_options.get(AUTO_BENCHMARK_KEY, True):
            if diagnostics:
                _record_diagnostic(
                    diagnostic_key, q, "V100", True, (), "forced",
                    diagnostics_interval,
                )
            return backend.comfy_attention(q, k, v, heads, scale=kwargs.get("scale"))

        key = _decision_key(q, k, v, heads)
        decision = _decision_cache.get(key)
        if decision is None:
            if diagnostics:
                transformer_options[FLASH_BENCHMARK_OCCURRED_KEY] = True
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
            if diagnostics:
                _record_diagnostic(
                    diagnostic_key, q, "V100" if decision else "ComfyUI", True,
                    (), "benchmarked", diagnostics_interval,
                )
            return candidate if decision else original_out
        if decision:
            if diagnostics:
                _record_diagnostic(
                    diagnostic_key, q, "V100", True, (), "cached",
                    diagnostics_interval,
                )
            return backend.comfy_attention(q, k, v, heads, scale=kwargs.get("scale"))
    if diagnostics:
        _record_diagnostic(
            diagnostic_key, q, "ComfyUI", eligible, tuple(reasons),
            "cached" if eligible else "ineligible", diagnostics_interval,
        )
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
        diagnostics_only=False, attention_mode=sol_attention.MODE_FLASH,
        sol_tau=1.0, sol_min_tokens=16_384, sol_block_size=64,
        sol_probe=False, allow_sol=False, sol_start_percent=0.2,
        sol_end_percent=0.8,
    ):
        if not enabled and not diagnostics_only:
            return (model,)
        if not torch.cuda.is_available() or not any(
            torch.cuda.get_device_capability(index) == (7, 0)
            for index in range(torch.cuda.device_count())
        ):
            raise RuntimeError("V100 FlashAttention requires an SM70 CUDA device.")

        if not diagnostics_only:
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
        effective_auto_benchmark = bool(
            auto_benchmark and attention_mode == sol_attention.MODE_AUTO
        )
        transformer_options[AUTO_BENCHMARK_KEY] = effective_auto_benchmark
        transformer_options[DIAGNOSTICS_ONLY_KEY] = bool(diagnostics_only)
        transformer_options[sol_attention.MODE_KEY] = str(attention_mode)
        transformer_options[sol_attention.TAU_KEY] = float(sol_tau)
        transformer_options[sol_attention.MIN_TOKENS_KEY] = int(sol_min_tokens)
        transformer_options[sol_attention.BLOCK_SIZE_KEY] = int(sol_block_size)
        transformer_options[sol_attention.PROBE_KEY] = bool(sol_probe)
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
            "Workflow-scoped V100 attention dispatcher active: mode=%s, "
            "flash_min_tokens=%d, auto_benchmark=%s, sol_allowed=%s, "
            "sol_min_tokens=%d, sol_tau=%.3f, sol_block_size=%d, sol_probe=%s.",
            attention_mode, int(min_tokens), effective_auto_benchmark, bool(allow_sol),
            int(sol_min_tokens), float(sol_tau), int(sol_block_size), bool(sol_probe),
        )
        return (patched,)
