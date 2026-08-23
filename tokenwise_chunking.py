"""Bounded-memory execution for explicitly validated token-wise modules."""

import logging
import math
import time
import types

import torch

from .diagnostics import DIAGNOSTICS_INTERVAL_KEY, DIAGNOSTICS_KEY


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
_stats = {}
_fp16_reported = set()
_selection_reported = set()
_UPGRADE_STABLE_CALLS = 3
_UPGRADE_BUDGET_MARGIN_TOKENS = 1024
_TRIM_COOLDOWN_CHECKS = 12


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


def _update_fp16_diagnostic(state, x, up, swiglu, result):
    values = (x, up, swiglu, result)
    for name, value in zip(
        ("input_absmax", "fc1_absmax", "swiglu_absmax", "output_absmax"),
        values,
    ):
        maximum = value.detach().abs().amax().float()
        previous = state.get(name)
        state[name] = maximum if previous is None else torch.maximum(previous, maximum)
    finite = torch.isfinite(result.detach()).all()
    previous_finite = state.get("finite")
    state["finite"] = finite if previous_finite is None else torch.logical_and(
        previous_finite, finite
    )


def _report_fp16_diagnostic(state, block_index, tokens, chunks):
    packed = torch.stack(
        [
            state["input_absmax"], state["fc1_absmax"],
            state["swiglu_absmax"], state["output_absmax"],
            state["finite"].float(),
        ]
    ).cpu().tolist()
    LOGGER.info(
        "V100 diagnostics H3 FP16 MLP: block=%d, sequence=%d, "
        "chunks=%d, scale=256, input_absmax=%.6g, fc1_absmax=%.6g, "
        "swiglu_absmax=%.6g, output_absmax=%.6g, finite=%s.",
        block_index, tokens, chunks, packed[0], packed[1], packed[2], packed[3],
        bool(packed[4]),
    )


def _call_mlp(
    module, original_forward, x, transformer_options, block_index,
    diagnostic_state=None,
):
    if not transformer_options.get(EXPERIMENTAL_FP16_KEY, False):
        return original_forward(x)
    up = module.fc1(x.half())
    gate, value = up.chunk(2, dim=-1)
    swiglu = torch.nn.functional.silu(gate.float()).mul_(value.float())
    result = module.fc2((swiglu / 256.0).half()).float().mul_(256.0)
    if diagnostic_state is not None:
        _update_fp16_diagnostic(diagnostic_state, x, up, swiglu, result)
    return result


def _balanced_chunk_tokens(tokens, target, alignment=256, minimum=640):
    """Choose the largest balanced chunk that stays inside a live limit."""
    tokens = max(1, int(tokens))
    target = max(1, int(target))
    alignment = max(1, int(alignment))
    minimum = max(1, int(minimum))
    if tokens <= target:
        return tokens

    # Start with the minimum number of chunks required by the live limit, then
    # align the per-chunk GEMM rows. Alignment can push a candidate just over
    # the limit, so increase the chunk count until the aligned size is safe.
    chunks = max(2, math.ceil(tokens / target))
    while chunks <= tokens:
        rows = math.ceil(tokens / chunks)
        aligned = math.ceil(rows / alignment) * alignment
        if aligned <= target:
            return max(minimum, aligned)
        chunks += 1
    return minimum


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
    # MLP temporaries are PyTorch allocations and can reuse allocator cache.
    # AIMDO streamed weights cannot, so direct-allocation pressure is handled
    # separately by the guarded trim policy before selection. Crediting cache
    # here prevents the old oscillation where empty_cache made a large tier
    # appear affordable, then the next untrimmed block collapsed to 640 rows.
    cache_credit_allowed = reusable_cache_bytes > 0
    credited_cache_bytes = reusable_cache_bytes
    effective_bytes = free_bytes + credited_cache_bytes

    # comfy_kitchen's SM70 eager INT8 fc1 materializes an INT32 accumulator.
    # Add the FP32 SwiGLU and FP32 result boundaries used by our validated path.
    fc1_width = int(getattr(module.fc1, "out_features", x.shape[-1] * 2))
    fc2_width = int(getattr(module.fc2, "out_features", x.shape[-1]))
    bytes_per_token = fc1_width * 4 + (fc1_width // 2) * 4 + fc2_width * 4
    safety_bytes = max(int(total_bytes * 0.10), 1536 * mib)
    # Reserve space for the next streamed weight and its cast/dequant buffers.
    # The 4% floor is 655 MiB on a 16 GiB V100; the reproduced failure was a
    # 64 MiB direct AIMDO copy after the surrounding buffers exhausted VRAM.
    transfer_reserve_bytes = max(512 * mib, int(total_bytes * 0.04))
    reserve_bytes = safety_bytes + transfer_reserve_bytes
    usable_bytes = max(0, effective_bytes - reserve_bytes)
    budget_tokens = max(1, int(usable_bytes // max(1, bytes_per_token)))

    # 32,917 succeeded while 36,176 OOMed in a full FP16 fc1 allocation on a
    # 16 GiB V100. Scale that measured guard by physical VRAM, while the live
    # budget below remains authoritative for other resident-memory layouts.
    reference_bytes = 16 * 1024 ** 3
    hard_full_limit = int(32_768 * total_bytes / reference_bytes)
    hard_full_limit = max(8_192, min(65_536, hard_full_limit))
    hard_full_limit = (hard_full_limit // 1024) * 1024
    maximum = min(tokens, hard_full_limit)
    target = min(maximum, budget_tokens)
    # The validated FP16-linear path has a pronounced SM70 GEMM sweet spot at
    # 640 rows: it is about 6-8% faster than 512 rows at the real H3 widths and
    # adds only about 28 MiB peak allocation. Keep the old 512 floor for the
    # legacy eager-INT8 path, whose INT32 accumulator has a different budget.
    minimum = 640 if experimental_fp16 else 512
    if target >= tokens and tokens <= hard_full_limit:
        selected = tokens
        reason = "runtime_budget_full"
    elif experimental_fp16:
        selected = _balanced_chunk_tokens(
            tokens, target, alignment=256, minimum=minimum
        )
        reason = (
            "balanced_measured_full_limit"
            if tokens > hard_full_limit
            else "balanced_runtime_memory_budget"
        )
    else:
        # Preserve the established tiers for the legacy eager-INT8 path. Its
        # INT32 accumulator has different SM70 GEMM and memory characteristics.
        candidates = (16_384, 8_192, 4_096, 2_048, 1_024, 512)
        selected = next((value for value in candidates if value <= target), minimum)
        reason = "measured_full_limit" if tokens > hard_full_limit else "runtime_memory_budget"
    details = {
        "driver_free_mib": free_bytes / mib,
        "reusable_cache_mib": reusable_cache_bytes / mib,
        "credited_cache_mib": credited_cache_bytes / mib,
        "cache_credit_allowed": cache_credit_allowed,
        "effective_mib": effective_bytes / mib,
        "safety_mib": safety_bytes / mib,
        "transfer_reserve_mib": transfer_reserve_bytes / mib,
        "estimated_full_mib": tokens * bytes_per_token / mib,
        "budget_tokens": budget_tokens,
        "hard_full_limit": hard_full_limit,
        "selection_cap": hard_full_limit,
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
    trim_stats = transformer_options.setdefault(
        "v100_mlp_trim_stats",
        {
            "checks": 0, "performed": 0, "released_mib": 0.0,
            "wall_ms": 0.0, "cooldown_remaining": 0,
            "cooldown_skips": 0, "pressure_skips": 0,
            "hard_pressure_trims": 0,
        },
    )
    trim_stats["checks"] += 1
    configured_threshold = max(
        0.0, float(transformer_options.get(CACHE_TRIM_THRESHOLD_KEY, 4096))
    )
    total_mib = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    # Dynamic VBAR prefetch is disabled and casts are synchronous on this safe
    # path. The previous one-quarter-VRAM threshold emptied the same 4-6 GiB
    # cache before virtually every H3 MLP (about 400 trims per generation).
    # Keep a 2 GiB soft floor on a 16 GiB V100 and a smaller hard floor that can
    # bypass cooldown when a direct streamed allocation is genuinely at risk.
    # The observed 17K steady state sits at 616-648 MiB driver-free, while the
    # reproduced AIMDO direct copy was 64 MiB. The old 768 MiB hard floor
    # therefore classified the healthy 17K allocator state as an emergency and
    # bypassed cooldown more than 100 times. A 512 MiB floor retains an 8x
    # margin for that direct copy without producing the length inversion.
    threshold = max(configured_threshold, total_mib * 0.125)
    hard_threshold = max(512.0, total_mib * 0.03)
    before = _memory(device)
    cached_mib = max(0.0, before[1] - before[0])
    if before[3] >= threshold or cached_mib < 512.0:
        trim_stats["cooldown_remaining"] = max(
            0, int(trim_stats["cooldown_remaining"]) - 1
        )
        trim_stats["pressure_skips"] += 1
        return (False, before, before, threshold)
    if (
        before[3] >= hard_threshold
        and int(trim_stats["cooldown_remaining"]) > 0
    ):
        trim_stats["cooldown_remaining"] -= 1
        trim_stats["cooldown_skips"] += 1
        return (False, before, before, threshold)
    if before[3] < hard_threshold and int(trim_stats["cooldown_remaining"]) > 0:
        trim_stats["hard_pressure_trims"] += 1
    trim_started = time.perf_counter()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    after = _memory(device)
    trim_stats["performed"] += 1
    trim_stats["cooldown_remaining"] = _TRIM_COOLDOWN_CHECKS
    trim_stats["released_mib"] += max(0.0, before[1] - after[1])
    trim_stats["wall_ms"] += (time.perf_counter() - trim_started) * 1000.0
    return (True, before, after, threshold)


def _make_forward(original_forward, block_index, transformer_options):
    """Chunk the leading token dimension without retaining chunk outputs."""

    def chunked_forward(self, x):
        # Reclaim allocator cache before measuring driver-free memory. Measuring
        # first caused an unnecessary, irreversible drop to 512-token chunks.
        pre_trimmed, pre_trim_before, pre_trim_after, threshold = _trim_if_needed(
            x.device, transformer_options
        )
        adaptive = bool(transformer_options.get(ADAPTIVE_KEY, False))
        if adaptive:
            selections = transformer_options.setdefault("v100_mlp_auto_selections", {})
            experimental_fp16 = bool(
                transformer_options.get(EXPERIMENTAL_FP16_KEY, False)
            )
            selection_key = (
                x.device.type, x.device.index, int(x.shape[0]), experimental_fp16,
            )
            selected, reason, details = _select_chunk_tokens(
                int(x.shape[0]), x, self, experimental_fp16
            )
            previous_selected = selections.get(selection_key)
            upgrade_states = transformer_options.setdefault(
                "v100_mlp_upgrade_states", {}
            )
            upgrade_state = upgrade_states.setdefault(
                selection_key, {"candidate": None, "stable_calls": 0}
            )
            if previous_selected is None or selected < previous_selected:
                selections[selection_key] = selected
                upgrade_state["candidate"] = None
                upgrade_state["stable_calls"] = 0
            elif selected == previous_selected:
                upgrade_state["candidate"] = None
                upgrade_state["stable_calls"] = 0
            else:
                # A low-memory first block must not pin every later hot call to
                # the smaller tier. Upgrade only after the same larger tier is
                # independently affordable for several consecutive blocks and
                # the live budget retains another 1K-token margin on top of it.
                budget_has_margin = bool(
                    details
                    and details["budget_tokens"]
                    >= selected + _UPGRADE_BUDGET_MARGIN_TOKENS
                )
                if budget_has_margin:
                    if upgrade_state["candidate"] == selected:
                        upgrade_state["stable_calls"] += 1
                    else:
                        upgrade_state["candidate"] = selected
                        upgrade_state["stable_calls"] = 1
                    if upgrade_state["stable_calls"] >= _UPGRADE_STABLE_CALLS:
                        selections[selection_key] = selected
                        reason = "stable_runtime_budget_upgrade"
                        upgrade_state["candidate"] = None
                        upgrade_state["stable_calls"] = 0
                else:
                    upgrade_state["candidate"] = None
                    upgrade_state["stable_calls"] = 0
            applied = selections[selection_key]
            report_selection_key = (
                selection_key + (int(applied), bool(experimental_fp16))
            )
            if (
                details
                and (
                    previous_selected is None
                    or applied < previous_selected
                    or report_selection_key not in _selection_reported
                )
            ):
                _selection_reported.add(report_selection_key)
                LOGGER.info(
                    "V100 adaptive MLP memory: adapter=minimax_h3, tokens=%d, "
                    "device=%s, selected_chunk_tokens=%d, chunks=%d, "
                    "experimental_fp16=%s, selection_reason=%s, "
                    "driver_free=%.1f MiB, reusable_cache=%.1f MiB, "
                    "credited_cache=%.1f MiB, "
                    "effective_budget=%.1f MiB, safety_reserve=%.1f MiB, "
                    "transfer_reserve=%.1f MiB, "
                    "estimated_full_peak=%.1f MiB, budget_tokens=%d, "
                    "capacity_guard_tokens=%d, selection_cap=%d, "
                    "upgrade_candidate=%s, upgrade_stable_calls=%d/%d, "
                    "cache_trim_threshold=%d MiB.",
                    int(x.shape[0]), x.device, applied,
                    math.ceil(int(x.shape[0]) / applied),
                    experimental_fp16, reason,
                    details["driver_free_mib"], details["reusable_cache_mib"],
                    details["credited_cache_mib"],
                    details["effective_mib"], details["safety_mib"],
                    details["transfer_reserve_mib"],
                    details["estimated_full_mib"], details["budget_tokens"],
                    details["hard_full_limit"], details["selection_cap"],
                    upgrade_state["candidate"], upgrade_state["stable_calls"],
                    _UPGRADE_STABLE_CALLS,
                    int(transformer_options.get(CACHE_TRIM_THRESHOLD_KEY, 2048)),
                )
            chunk_tokens = selections[selection_key]
        else:
            chunk_tokens = max(0, int(transformer_options.get(OPTION_KEY, 0)))
        full_tokens = int(x.shape[0]) if x.ndim > 0 else 0
        report_key = (block_index, full_tokens)
        collect_fp16 = bool(
            transformer_options.get(EXPERIMENTAL_FP16_KEY, False)
            and transformer_options.get(DIAGNOSTICS_KEY, False)
            and block_index == 0
            and report_key not in _fp16_reported
        )
        diagnostic_state = {} if collect_fp16 else None
        if chunk_tokens == 0 or x.ndim != 2 or x.shape[0] <= chunk_tokens:
            diagnostics = bool(transformer_options.get(DIAGNOSTICS_KEY, False))
            report = diagnostics and block_index == 0 and x.ndim == 2
            before = _memory(x.device) if report else None
            started = time.perf_counter()
            cuda_start = cuda_end = None
            if report and x.is_cuda:
                cuda_start = torch.cuda.Event(enable_timing=True)
                cuda_end = torch.cuda.Event(enable_timing=True)
                cuda_start.record()
            result = _call_mlp(
                self, original_forward, x, transformer_options, block_index,
                diagnostic_state,
            )
            if cuda_end is not None:
                cuda_end.record()
            cpu_ms = (time.perf_counter() - started) * 1000.0
            if diagnostic_state:
                _report_fp16_diagnostic(diagnostic_state, block_index, full_tokens, 1)
                _fp16_reported.add(report_key)
            if report:
                cuda_ms = 0.0
                if cuda_end is not None:
                    cuda_end.synchronize()
                    cuda_ms = cuda_start.elapsed_time(cuda_end)
                after = _memory(x.device)
                trim_stats = transformer_options.get("v100_mlp_trim_stats", {})
                LOGGER.info(
                    "V100 diagnostics token-wise MLP: adapter=minimax_h3, "
                    "execution=full, block=%d, input_shape=%s, input_dtype=%s, "
                    "output_dtype=%s, device=%s, tokens=%d, chunk_tokens=%d, "
                    "chunks=1, CPU_submit=%.3f ms, CUDA=%.3f ms, "
                    "allocated_before/after=%.1f/%.1f MiB, "
                    "reserved_before/after=%.1f/%.1f MiB, global_peak=%.1f MiB, "
                    "cuda_free=%.1f MiB, trim_checks=%d, trim_performed=%d, "
                    "trim_released=%.1f MiB, trim_wall=%.3f ms, "
                    "trim_cooldown_skips=%d, trim_pressure_skips=%d, "
                    "trim_hard_pressure=%d.",
                    block_index, tuple(x.shape), x.dtype, result.dtype, x.device,
                    full_tokens, full_tokens, cpu_ms, cuda_ms,
                    before[0], after[0], before[1], after[1], after[2], after[3],
                    int(trim_stats.get("checks", 0)),
                    int(trim_stats.get("performed", 0)),
                    float(trim_stats.get("released_mib", 0.0)),
                    float(trim_stats.get("wall_ms", 0.0)),
                    int(trim_stats.get("cooldown_skips", 0)),
                    int(trim_stats.get("pressure_skips", 0)),
                    int(trim_stats.get("hard_pressure_trims", 0)),
                )
            return result

        tokens = int(x.shape[0])
        chunks = math.ceil(tokens / chunk_tokens)
        diagnostics = bool(transformer_options.get(DIAGNOSTICS_KEY, False))
        interval = max(0, int(transformer_options.get(DIAGNOSTICS_INTERVAL_KEY, 50)))
        key = (x.device.type, x.device.index, block_index, tokens, chunk_tokens)
        state = _stats.setdefault(key, {"calls": 0, "cpu_ms": 0.0})
        # One representative block is enough to expose the shared MLP cost.
        # Synchronizing all 50 blocks would perturb the generation being
        # measured and produce mostly redundant diagnostics.
        report = diagnostics and block_index == 0
        before = _memory(x.device) if report else None
        started = time.perf_counter()
        cuda_start = cuda_end = None
        if report and x.is_cuda:
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()

        # Gradients are not part of ComfyUI inference. Preserve normal autograd
        # semantics if this utility is ever called by a training workflow.
        if torch.is_grad_enabled() and x.requires_grad:
            result = torch.cat(
                [_call_mlp(
                    self, original_forward, part, transformer_options, block_index,
                    diagnostic_state,
                )
                 for part in x.split(chunk_tokens, dim=0)],
                dim=0,
            )
        else:
            result = None
            for start in range(0, tokens, chunk_tokens):
                part = _call_mlp(
                    self, original_forward, x[start:start + chunk_tokens],
                    transformer_options, block_index, diagnostic_state,
                )
                if result is None:
                    result = torch.empty(
                        (tokens,) + tuple(part.shape[1:]),
                        dtype=part.dtype,
                        device=part.device,
                    )
                result[start:start + part.shape[0]].copy_(part)
                del part

        if cuda_end is not None:
            cuda_end.record()
        cpu_ms = (time.perf_counter() - started) * 1000.0
        state["calls"] += 1
        state["cpu_ms"] += cpu_ms

        if diagnostic_state:
            _report_fp16_diagnostic(
                diagnostic_state, block_index, tokens, chunks
            )
            _fp16_reported.add(report_key)

        trim_performed, trim_before, trim_after, threshold = _trim_if_needed(
            x.device, transformer_options
        )

        if report:
            cuda_ms = 0.0
            if cuda_end is not None:
                cuda_end.synchronize()
                cuda_ms = cuda_start.elapsed_time(cuda_end)
            after = _memory(x.device)
            trim_stats = transformer_options.get("v100_mlp_trim_stats", {})
            LOGGER.info(
                "V100 diagnostics token-wise MLP: adapter=minimax_h3, block=%d, "
                "input_shape=%s, input_dtype=%s, output_dtype=%s, device=%s, "
                "tokens=%d, chunk_tokens=%d, chunks=%d, CPU=%.3f ms, CUDA=%.3f ms, "
                "allocated_before/after=%.1f/%.1f MiB, reserved_before/after=%.1f/%.1f MiB, "
                "global_peak=%.1f MiB, cuda_free=%.1f MiB, calls=%d, CPU_avg=%.3f ms, "
                "trim_checks=%d, trim_performed=%d, trim_released=%.1f MiB, "
                "trim_wall=%.3f ms, trim_cooldown_skips=%d, "
                "trim_pressure_skips=%d, trim_hard_pressure=%d.",
                block_index, tuple(x.shape), x.dtype, result.dtype, x.device,
                tokens, chunk_tokens, chunks, cpu_ms, cuda_ms,
                before[0], after[0], before[1], after[1], after[2], after[3],
                state["calls"], state["cpu_ms"] / state["calls"],
                int(trim_stats.get("checks", 0)),
                int(trim_stats.get("performed", 0)),
                float(trim_stats.get("released_mib", 0.0)),
                float(trim_stats.get("wall_ms", 0.0)),
                int(trim_stats.get("cooldown_skips", 0)),
                int(trim_stats.get("pressure_skips", 0)),
                int(trim_stats.get("hard_pressure_trims", 0)),
            )
            if pre_trimmed:
                LOGGER.info(
                    "V100 diagnostics MLP pre-load cache trim: block=%d, "
                    "threshold=%.1f MiB, allocated=%.1f MiB, "
                    "reserved_before/after=%.1f/%.1f MiB, "
                    "cuda_free_before/after=%.1f/%.1f MiB, released=%.1f MiB.",
                    block_index, threshold, pre_trim_before[0], pre_trim_before[1],
                    pre_trim_after[1], pre_trim_before[3], pre_trim_after[3],
                    max(0.0, pre_trim_before[1] - pre_trim_after[1]),
                )
            if trim_performed:
                LOGGER.info(
                    "V100 diagnostics MLP cache trim: block=%d, threshold=%.1f MiB, "
                    "allocated=%.1f MiB, reserved_before/after=%.1f/%.1f MiB, "
                    "cuda_free_before/after=%.1f/%.1f MiB, released=%.1f MiB.",
                    block_index, threshold, trim_before[0], trim_before[1],
                    trim_after[1], trim_before[3], trim_after[3],
                    max(0.0, trim_before[1] - trim_after[1]),
                )
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
                    _make_forward(base_forward, index, transformer_options), mlp
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
                "H3 validated FP16 linear path active: MLP fc1/fc2 use FP16 "
                "with FP32 SwiGLU and fc2 scale=256."
            )
        if adaptive:
            LOGGER.info(
                "H3 adaptive QKV token chunking armed: threshold=%d tokens, "
                "chunk_tokens=%d, cache_trim_threshold=%d MiB.",
                38_000, 1024, max(0, int(cache_trim_threshold_mb)),
            )
        return (patched,)
