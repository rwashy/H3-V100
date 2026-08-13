"""Bounded-memory execution for explicitly validated token-wise modules."""

import logging
import math
import types

import torch

LOGGER = logging.getLogger("V100TokenChunking")
PATCH_MARKER = "_v100_tokenwise_chunking"
ORIGINAL_FORWARD_ATTR = "_v100_tokenwise_original_forward"
OPTION_KEY = "v100_mlp_chunk_tokens"
ADAPTIVE_KEY = "v100_mlp_adaptive"
CACHE_TRIM_KEY = "v100_mlp_cache_trim"
CACHE_TRIM_THRESHOLD_KEY = "v100_mlp_cache_trim_threshold_mb"
QKV_CHUNKING_KEY = "v100_h3_qkv_chunking"
QKV_CHUNK_TOKENS_KEY = "v100_h3_qkv_chunk_tokens"
QKV_CHUNK_THRESHOLD_KEY = "v100_h3_qkv_chunk_threshold"
QKV_CACHE_TRIM_THRESHOLD_KEY = "v100_h3_qkv_cache_trim_threshold_mb"
EXPERIMENTAL_FP16_KEY = "v100_h3_experimental_fp16_linear"


def _unwrap_our_forward(value):
    """Return the clean callable beneath any previous copy of our wrapper."""
    current = value
    seen = set()
    while current is not None:
        function = getattr(current, "__func__", current)
        if not getattr(function, PATCH_MARKER, False):
            return current
        identity = id(function)
        if identity in seen:
            raise RuntimeError("H3 MLP chunking detected a cyclic V100 wrapper chain.")
        seen.add(identity)
        current = getattr(function, ORIGINAL_FORWARD_ATTR, None)
    raise RuntimeError("H3 MLP chunking could not recover its original forward.")


def _call_mlp(module, original_forward, x, transformer_options):
    if not transformer_options.get(EXPERIMENTAL_FP16_KEY, False):
        return original_forward(x)
    up = module.fc1(x.half())
    gate, value = up.chunk(2, dim=-1)
    swiglu = torch.nn.functional.silu(gate.float()).mul_(value.float())
    result = module.fc2((swiglu / 256.0).half()).float().mul_(256.0)
    return result


def _select_chunk_tokens(tokens, x, module, experimental_fp16=False):
    """Budget an H3 MLP call before the quantized fc1 allocation occurs."""
    device = x.device
    if device.type != "cuda":
        return tokens, "non_cuda_full", {}

    mib = 1024 ** 2
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    reusable_cache_bytes = max(0, reserved_bytes - allocated_bytes)
    # Allocator cache can be fragmented or otherwise unavailable to one large
    # allocation, so it must not be valued the same as driver-reported free VRAM.
    credited_cache_bytes = reusable_cache_bytes // 2
    effective_bytes = free_bytes + credited_cache_bytes

    # comfy_kitchen's SM70 eager INT8 fc1 materializes an INT32 accumulator.
    # Add the FP32 SwiGLU and FP32 result boundaries used by our validated path.
    fc1_width = int(getattr(module.fc1, "out_features", x.shape[-1] * 2))
    fc2_width = int(getattr(module.fc2, "out_features", x.shape[-1]))
    bytes_per_token = fc1_width * 4 + (fc1_width // 2) * 4 + fc2_width * 4
    safety_bytes = max(int(total_bytes * 0.10), 1536 * mib)
    usable_bytes = max(0, effective_bytes - safety_bytes)
    budget_tokens = max(1, int(usable_bytes // max(1, bytes_per_token)))

    # 32,917 succeeded while 36,176 OOMed in a full FP16 fc1 allocation on a
    # 16 GiB V100. Scale that measured guard by physical VRAM, while the live
    # budget below remains authoritative for other resident-memory layouts.
    reference_bytes = 16 * 1024 ** 3
    hard_full_limit = int(32_768 * total_bytes / reference_bytes)
    hard_full_limit = max(8_192, min(65_536, hard_full_limit))
    hard_full_limit = (hard_full_limit // 1024) * 1024
    if tokens > 96_000:
        long_sequence_cap = 4_096
        cap_reason = "very_long_sequence_guard"
    elif tokens > 64_000:
        long_sequence_cap = 8_192
        cap_reason = "long_sequence_guard"
    elif free_bytes < 1024 * mib:
        long_sequence_cap = 8_192
        cap_reason = "low_driver_free_guard"
    else:
        long_sequence_cap = 16_384
        cap_reason = "runtime_capacity"
    maximum = tokens if tokens <= hard_full_limit else min(tokens, long_sequence_cap)
    target = min(maximum, budget_tokens)
    candidates = (16_384, 8_192, 4_096, 2_048, 1_024, 512)
    if target >= tokens and tokens <= hard_full_limit:
        selected = tokens
        reason = "runtime_budget_full"
    else:
        selected = next((value for value in candidates if value <= target), 512)
        reason = cap_reason if tokens > hard_full_limit else "runtime_memory_budget"
    details = {
        "driver_free_mib": free_bytes / mib,
        "reusable_cache_mib": reusable_cache_bytes / mib,
        "credited_cache_mib": credited_cache_bytes / mib,
        "effective_mib": effective_bytes / mib,
        "safety_mib": safety_bytes / mib,
        "estimated_full_mib": tokens * bytes_per_token / mib,
        "budget_tokens": budget_tokens,
        "hard_full_limit": hard_full_limit,
        "selection_cap": long_sequence_cap,
    }
    return selected, reason, details


def _memory(device):
    if device.type != "cuda":
        return (0.0, 0.0, 0.0, 0.0)
    free, _ = torch.cuda.mem_get_info(device)
    mib = 1024 ** 2
    return (
        torch.cuda.memory_allocated(device) / mib,
        torch.cuda.memory_reserved(device) / mib,
        torch.cuda.max_memory_allocated(device) / mib,
        free / mib,
    )


def _trim_if_needed(device, transformer_options):
    if device.type != "cuda" or not transformer_options.get(CACHE_TRIM_KEY, False):
        return (False, None, None, 0.0)
    threshold = max(
        0.0, float(transformer_options.get(CACHE_TRIM_THRESHOLD_KEY, 4096))
    )
    before = _memory(device)
    cached_mib = max(0.0, before[1] - before[0])
    if before[3] >= threshold or cached_mib < 512.0:
        return (False, before, before, threshold)
    torch.cuda.empty_cache()
    return (True, before, _memory(device), threshold)


def _make_forward(original_forward, transformer_options):
    """Chunk the leading token dimension without retaining chunk outputs."""

    def chunked_forward(self, x):
        adaptive = bool(transformer_options.get(ADAPTIVE_KEY, False))
        if adaptive:
            selections = transformer_options.setdefault("v100_mlp_auto_selections", {})
            experimental_fp16 = bool(
                transformer_options.get(EXPERIMENTAL_FP16_KEY, False)
            )
            selection_key = (
                x.device.type, x.device.index, int(x.shape[0]), experimental_fp16
            )
            if selection_key not in selections:
                selected, reason, details = _select_chunk_tokens(
                    int(x.shape[0]), x, self, experimental_fp16
                )
                selections[selection_key] = selected
                LOGGER.info(
                    "V100 adaptive MLP memory: adapter=minimax_h3, tokens=%d, "
                    "device=%s, selected_chunk_tokens=%d, chunks=%d, "
                    "fp16_linear=%s, selection_reason=%s, "
                    "driver_free=%.1f MiB, reusable_cache=%.1f MiB, "
                    "credited_cache=%.1f MiB, "
                    "effective_budget=%.1f MiB, safety_reserve=%.1f MiB, "
                    "estimated_full_peak=%.1f MiB, budget_tokens=%d, "
                    "capacity_guard_tokens=%d, selection_cap=%d, "
                    "cache_trim_threshold=%d MiB.",
                    int(x.shape[0]), x.device, selections[selection_key],
                    math.ceil(int(x.shape[0]) / selections[selection_key]),
                    experimental_fp16, reason,
                    details["driver_free_mib"], details["reusable_cache_mib"],
                    details["credited_cache_mib"],
                    details["effective_mib"], details["safety_mib"],
                    details["estimated_full_mib"], details["budget_tokens"],
                    details["hard_full_limit"], details["selection_cap"],
                    int(transformer_options.get(CACHE_TRIM_THRESHOLD_KEY, 2048)),
                )
            chunk_tokens = selections[selection_key]
        else:
            chunk_tokens = max(0, int(transformer_options.get(OPTION_KEY, 0)))
        if chunk_tokens == 0 or x.ndim != 2 or x.shape[0] <= chunk_tokens:
            return _call_mlp(self, original_forward, x, transformer_options)

        tokens = int(x.shape[0])
        _trim_if_needed(x.device, transformer_options)

        # Gradients are not part of ComfyUI inference. Preserve normal autograd
        # semantics if this utility is ever called by a training workflow.
        if torch.is_grad_enabled() and x.requires_grad:
            result = torch.cat(
                [_call_mlp(
                    self, original_forward, part, transformer_options,
                )
                 for part in x.split(chunk_tokens, dim=0)],
                dim=0,
            )
        else:
            result = None
            for start in range(0, tokens, chunk_tokens):
                part = _call_mlp(
                    self, original_forward, x[start:start + chunk_tokens],
                    transformer_options,
                )
                if result is None:
                    result = torch.empty(
                        (tokens,) + tuple(part.shape[1:]),
                        dtype=part.dtype,
                        device=part.device,
                    )
                result[start:start + part.shape[0]].copy_(part)

        _trim_if_needed(x.device, transformer_options)
        return result

    setattr(chunked_forward, PATCH_MARKER, True)
    setattr(chunked_forward, ORIGINAL_FORWARD_ATTR, _unwrap_our_forward(original_forward))
    return chunked_forward


class H3TokenwiseMLPChunking:
    """Adapter for H3's validated [tokens, hidden] token-independent MLPs."""

    def patch(self, model, chunk_tokens=512, cache_trim=True, cache_trim_threshold_mb=2048,
              adaptive=False, experimental_fp16=False):
        chunk_tokens = max(0, int(chunk_tokens))
        if chunk_tokens == 0:
            return (model,)
        patched = model.clone()
        diffusion_model = patched.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "blocks", None)
        if not blocks:
            raise RuntimeError("H3 MLP chunking expected diffusion_model.blocks.")
        transformer_options = patched.model_options.setdefault("transformer_options", {})
        transformer_options[OPTION_KEY] = chunk_tokens
        transformer_options[ADAPTIVE_KEY] = bool(adaptive)
        transformer_options[CACHE_TRIM_KEY] = bool(cache_trim)
        transformer_options[CACHE_TRIM_THRESHOLD_KEY] = max(
            0, int(cache_trim_threshold_mb)
        )
        transformer_options[QKV_CHUNKING_KEY] = bool(adaptive)
        transformer_options[QKV_CHUNK_TOKENS_KEY] = 1024
        transformer_options[QKV_CHUNK_THRESHOLD_KEY] = 38_000
        transformer_options[QKV_CACHE_TRIM_THRESHOLD_KEY] = max(
            0, int(cache_trim_threshold_mb)
        )
        transformer_options[EXPERIMENTAL_FP16_KEY] = bool(experimental_fp16)
        count = 0
        refreshed = 0
        for index, block in enumerate(blocks):
            mlp = getattr(block, "mlp", None)
            if mlp is None or not all(hasattr(mlp, name) for name in ("fc1", "fc2")):
                raise RuntimeError(f"H3 MLP chunking rejected block {index}: incompatible MLP.")
            key = f"diffusion_model.blocks.{index}.mlp.forward"
            existing = patched.object_patches.get(key)
            if existing is not None:
                function = getattr(existing, "__func__", existing)
                if not getattr(function, PATCH_MARKER, False):
                    raise RuntimeError(f"H3 MLP chunking found another patch at {key}.")
                base_forward = _unwrap_our_forward(existing)
                refreshed += 1
            else:
                base_forward = _unwrap_our_forward(mlp.forward)
            patched.add_object_patch(
                key,
                types.MethodType(
                    _make_forward(base_forward, transformer_options), mlp
                ),
            )
            count += 1
        LOGGER.info(
            "H3 bounded-memory token-wise MLP active: blocks=%d, refreshed=%d, "
            "adaptive=%s, chunk_tokens=%d, "
            "cache_trim=%s, cache_trim_threshold=%d MiB. "
            "This reduces peak activation memory and may increase generation time.",
            count, refreshed, bool(adaptive), chunk_tokens, bool(cache_trim),
            max(0, int(cache_trim_threshold_mb)),
        )
        if experimental_fp16:
            LOGGER.warning(
                "H3 FP16 linear path active: MLP fc1/fc2 use FP16 with FP32 "
                "SwiGLU and fc2 scale=256."
            )
        if adaptive:
            LOGGER.info(
                "H3 adaptive QKV token chunking armed: threshold=%d tokens, "
                "chunk_tokens=%d, cache_trim_threshold=%d MiB.",
                38_000, 1024, max(0, int(cache_trim_threshold_mb)),
            )
        return (patched,)
