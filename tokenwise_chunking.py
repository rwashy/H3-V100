"""Bounded-memory execution for explicitly validated token-wise modules."""

import logging
import math
import time
import types
from contextlib import ExitStack, contextmanager

import torch

from .native_dynamic_vbar import CONTROLLER_KEY


LOGGER = logging.getLogger("V100TokenChunking")
DIAGNOSTICS_KEY = "_h3_v100_removed_diagnostics"
DIAGNOSTICS_INTERVAL_KEY = "_h3_v100_removed_diagnostics_interval"
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
SCALED_FP16_SWIGLU_KEY = "v100_h3_scaled_fp16_swiglu"
_SWIGLU_BRANCH_SCALE = 16.0
_SWIGLU_FC2_SCALE = 8.0
_stats = {}
_fp16_reported = set()
_selection_reported = set()
_controller_bindings_reported = set()
CONTROLLER_ATTR = "_h3_v100_dynamic_vbar_policy"
_UPGRADE_STABLE_CALLS = 3
_UPGRADE_BUDGET_MARGIN_TOKENS = 1024
_TRIM_COOLDOWN_CHECKS = 12
_MLP_WEIGHT_PAIR_DRIVER_FLOOR_MIB = 1024
_weight_pair_fallback_reported = set()


class _WeightPairUnsupported(TypeError):
    pass


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


def _update_fp16_diagnostic(
    state, x, up, swiglu, result, *, arithmetic="fp32_swiglu_scale256",
    swiglu_restore_scale=1.0,
):
    state["arithmetic"] = arithmetic
    state["swiglu_restore_scale"] = float(swiglu_restore_scale)
    values = (x, up, swiglu, result)
    names = ("input", "fc1", "swiglu", "output")
    for name, value in zip(
        names,
        values,
    ):
        maximum = value.detach().abs().amax().float()
        maximum_key = f"{name}_absmax"
        previous = state.get(maximum_key)
        state[maximum_key] = (
            maximum if previous is None else torch.maximum(previous, maximum)
        )
        finite = torch.isfinite(value.detach()).all()
        finite_key = f"{name}_finite"
        previous_finite = state.get(finite_key)
        state[finite_key] = (
            finite if previous_finite is None
            else torch.logical_and(previous_finite, finite)
        )


def _report_fp16_diagnostic(state, block_index, tokens, chunks):
    packed = torch.stack(
        [
            state["input_absmax"], state["fc1_absmax"],
            state["swiglu_absmax"], state["output_absmax"],
            state["input_finite"].float(), state["fc1_finite"].float(),
            state["swiglu_finite"].float(), state["output_finite"].float(),
        ]
    ).cpu().tolist()
    restore_scale = float(state.get("swiglu_restore_scale", 1.0))
    LOGGER.info(
        "V100 diagnostics H3 FP16 MLP: block=%d, sequence=%d, "
        "chunks=%d, arithmetic=%s, input_absmax=%.6g, fc1_absmax=%.6g, "
        "stored_swiglu_absmax=%.6g, swiglu_restore_scale=%.1f, "
        "restored_swiglu_absmax=%.6g, output_absmax=%.6g, "
        "finite_input_fc1_swiglu_output=%s/%s/%s/%s.",
        block_index, tokens, chunks, state.get("arithmetic", "unknown"),
        packed[0], packed[1], packed[2], restore_scale,
        packed[2] * restore_scale, packed[3], bool(packed[4]), bool(packed[5]),
        bool(packed[6]), bool(packed[7]),
    )


def _call_mlp(
    module, original_forward, x, transformer_options, block_index,
    diagnostic_state=None, prepared_weights=None,
):
    if not transformer_options.get(EXPERIMENTAL_FP16_KEY, False):
        return original_forward(x)
    controller = transformer_options.get(CONTROLLER_KEY)
    if prepared_weights is None:
        if controller is not None:
            controller.reserve("fc1", x.device)
        up = module.fc1(x.half())
    else:
        fc1_weight, fc1_bias, fc2_weight, fc2_bias = prepared_weights
        from comfy.ops import run_every_op
        run_every_op()
        up = module.fc1._forward(x.half(), fc1_weight, fc1_bias)
    gate, value = up.chunk(2, dim=-1)
    if transformer_options.get(SCALED_FP16_SWIGLU_KEY, False):
        # Keep the large fc1/SwiGLU intermediates in FP16. Both divisors are
        # powers of two, so their scaling is exact in binary floating point;
        # only the final, narrow fc2 result is promoted before restoring the
        # combined scale. The activation bounds and final image/audio path have
        # been validated end to end on the stable V100 profile.
        # Scale the value branch before multiplication to prevent the gated
        # product itself overflowing. Reuse the activation allocation for the
        # product and fc2 scaling; this avoids another full [tokens, I] FP16
        # temporary at the real H3 width.
        scaled_value = value * (1.0 / _SWIGLU_BRANCH_SCALE)
        swiglu = torch.nn.functional.silu(gate).mul_(scaled_value)
        del scaled_value
        swiglu.mul_(1.0 / _SWIGLU_FC2_SCALE)
        if prepared_weights is None:
            if controller is not None:
                controller.reserve("fc2", x.device)
            projected = module.fc2(swiglu)
        else:
            run_every_op()
            projected = module.fc2._forward(swiglu, fc2_weight, fc2_bias)
        result = projected.float().mul_(
            _SWIGLU_BRANCH_SCALE * _SWIGLU_FC2_SCALE
        )
        if diagnostic_state is not None:
            _update_fp16_diagnostic(
                diagnostic_state, x, up, swiglu, result,
                arithmetic="fp16_swiglu_branch16_fc2scale8",
                swiglu_restore_scale=(
                    _SWIGLU_BRANCH_SCALE * _SWIGLU_FC2_SCALE
                ),
            )
        return result
    swiglu = torch.nn.functional.silu(gate.float()).mul_(value.float())
    fc2_input = (swiglu / 256.0).half()
    if prepared_weights is None:
        if controller is not None:
            controller.reserve("fc2", x.device)
        projected = module.fc2(fc2_input)
    else:
        run_every_op()
        projected = module.fc2._forward(fc2_input, fc2_weight, fc2_bias)
    result = projected.float().mul_(256.0)
    if diagnostic_state is not None:
        _update_fp16_diagnostic(diagnostic_state, x, up, swiglu, result)
    return result


def _tensor_nbytes(value):
    if value is None:
        return 0
    return int(value.numel()) * int(value.element_size())


def _is_weight_pair_resource_error(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "out of memory", "vbar_fault", "vbar fault", "result 2",
            "cuda error: memory allocation", "cudamalloc",
        )
    )


@contextmanager
def _prepare_mlp_weight_pair(module, x, transformer_options):
    """Prepare and pin fc1/fc2 once for all chunks in this MLP invocation."""
    from comfy.ops import CastBiasWeightContext

    if not all(hasattr(linear, "_forward") for linear in (module.fc1, module.fc2)):
        raise _WeightPairUnsupported(
            "H3 MLP weight-pair reuse requires ComfyUI Linear._forward."
        )
    if any(
        getattr(linear, "pre_quant_scale", None) is not None
        for linear in (module.fc1, module.fc2)
    ):
        raise _WeightPairUnsupported(
            "H3 MLP weight-pair reuse does not accept pre_quant_scale."
        )

    controller = transformer_options.get(CONTROLLER_KEY)
    if controller is not None:
        controller.reserve("fc1", x.device)
        controller.reserve("fc2", x.device)
    kwargs = {
        "input": None,
        "dtype": torch.float16,
        "device": x.device,
        "bias_dtype": torch.float16,
        "offloadable": True,
        "compute_dtype": torch.float16,
        "want_requant": False,
    }
    with ExitStack() as stack:
        fc1_weight, fc1_bias = stack.enter_context(
            CastBiasWeightContext(module.fc1, **kwargs)
        )
        fc2_weight, fc2_bias = stack.enter_context(
            CastBiasWeightContext(module.fc2, **kwargs)
        )
        weights = (fc1_weight, fc1_bias, fc2_weight, fc2_bias)
        yield weights, sum(_tensor_nbytes(value) for value in weights)


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


def _select_chunk_tokens(
    tokens, x, module, experimental_fp16=False,
    native_headroom_policy=False,
):
    """Budget an H3 MLP call before the quantized fc1 allocation occurs."""
    device = x.device
    if device.type != "cuda":
        return tokens, "non_cuda_full", {}

    mib = 1024 ** 2
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    reusable_cache_bytes = max(0, reserved_bytes - allocated_bytes)
    # Dynamic mode budgets from real driver-free memory. Weight preparation is
    # reduced locally within one MLP invocation.
    credited_cache_bytes = (
        0 if native_headroom_policy else reusable_cache_bytes
    )
    cache_credit_allowed = credited_cache_bytes > 0
    effective_bytes = free_bytes + credited_cache_bytes

    # comfy_kitchen's SM70 eager INT8 fc1 materializes an INT32 accumulator.
    # Add the FP32 SwiGLU and FP32 result boundaries used by our validated path.
    # Keep this conservative model for scaled SwiGLU too: isolated V100 tests
    # found 3,584/6,400/8,448 rows within 0.2%, so spending its saved memory on
    # a larger cold-start chunk has no useful speed return.
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


def _make_forward(
    original_forward, block_index, transformer_options,
    expected_dynamic_vbar_controller=None,
):
    """Chunk the leading token dimension without retaining chunk outputs."""

    def chunked_forward(self, x):
        # Reclaim allocator cache before measuring driver-free memory. Measuring
        # first caused an unnecessary, irreversible drop to 512-token chunks.
        pre_trimmed, pre_trim_before, pre_trim_after, threshold = _trim_if_needed(
            x.device, transformer_options
        )
        controller = transformer_options.get(CONTROLLER_KEY)
        if expected_dynamic_vbar_controller is not None:
            if controller is not expected_dynamic_vbar_controller:
                raise RuntimeError(
                    "H3 native Dynamic VBAR policy binding was lost "
                    f"before MLP block {block_index}."
                )
            binding_key = id(expected_dynamic_vbar_controller)
            if binding_key not in _controller_bindings_reported:
                _controller_bindings_reported.add(binding_key)
                LOGGER.info(
                    "H3 native Dynamic VBAR policy binding confirmed: "
                    "block=%d controller_shared=True vbar_cache_credit=False "
                    "mlp_weight_pair_reuse=True.",
                    block_index,
                )
        if controller is not None:
            controller.begin_mlp(block_index, x.device)
        adaptive = bool(transformer_options.get(ADAPTIVE_KEY, False))
        if adaptive:
            selections = transformer_options.setdefault("v100_mlp_auto_selections", {})
            experimental_fp16 = bool(
                transformer_options.get(EXPERIMENTAL_FP16_KEY, False)
            )
            scaled_fp16_swiglu = bool(
                transformer_options.get(SCALED_FP16_SWIGLU_KEY, False)
            )
            selection_key = (
                x.device.type, x.device.index, int(x.shape[0]),
                experimental_fp16, scaled_fp16_swiglu,
            )
            selected, reason, details = _select_chunk_tokens(
                int(x.shape[0]), x, self, experimental_fp16,
                native_headroom_policy=controller is not None,
            )
            if controller is not None and selected > 8448:
                selected = _balanced_chunk_tokens(
                    int(x.shape[0]), 8448, alignment=256, minimum=640
                )
                reason = "native_dynamic_vbar_correctness_cap"
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
            report_selection_key = selection_key + (int(applied),)
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
        arithmetic_key = bool(
            transformer_options.get(SCALED_FP16_SWIGLU_KEY, False)
        )
        report_key = (block_index, full_tokens, arithmetic_key)
        collect_fp16 = bool(
            transformer_options.get(EXPERIMENTAL_FP16_KEY, False)
            and False
            and block_index == 0
            and report_key not in _fp16_reported
        )
        diagnostic_state = {} if collect_fp16 else None
        if chunk_tokens == 0 or x.ndim != 2 or x.shape[0] <= chunk_tokens:
            diagnostics = False
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
        diagnostics = False
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

        def execute_chunks(prepared_weights=None):
            output = None
            for chunk_start in range(0, tokens, chunk_tokens):
                part = _call_mlp(
                    self, original_forward,
                    x[chunk_start:chunk_start + chunk_tokens],
                    transformer_options, block_index, diagnostic_state,
                    prepared_weights=prepared_weights,
                )
                if output is None:
                    output = torch.empty(
                        (tokens,) + tuple(part.shape[1:]),
                        dtype=part.dtype,
                        device=part.device,
                    )
                output[chunk_start:chunk_start + part.shape[0]].copy_(part)
                del part
            return output

        pair_info = {
            "eligible": False, "prepared": False, "fallback": False,
            "reason": "not_dynamic_fp16_multichunk", "prep_cpu_ms": 0.0,
            "expanded_mib": 0.0, "driver_free_before_mib": 0.0,
            "driver_free_after_mib": 0.0,
        }
        pair_stats = transformer_options.setdefault(
            "v100_mlp_weight_pair_stats",
            {
                "attempts": 0, "prepared": 0, "fallbacks": 0,
                "skipped_floor": 0, "reused_projection_calls": 0,
                "prep_cpu_ms": 0.0, "expanded_mib_peak": 0.0,
            },
        )

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
            pair_info["eligible"] = bool(
                chunks > 1
                and x.is_cuda
                and transformer_options.get(EXPERIMENTAL_FP16_KEY, False)
                and transformer_options.get(CONTROLLER_KEY) is not None
            )
            if pair_info["eligible"]:
                free_before, _ = torch.cuda.mem_get_info(x.device)
                pair_info["driver_free_before_mib"] = free_before / (1024 ** 2)
                if pair_info["driver_free_before_mib"] < _MLP_WEIGHT_PAIR_DRIVER_FLOOR_MIB:
                    pair_info["reason"] = "driver_transfer_floor"
                    pair_stats["skipped_floor"] += 1
                    result = execute_chunks()
                else:
                    pair_info["reason"] = "prepared_once"
                    pair_stats["attempts"] += 1
                    prepare_started = time.perf_counter()
                    fallback_exc = None
                    try:
                        with _prepare_mlp_weight_pair(
                            self, x, transformer_options
                        ) as (prepared_weights, expanded_bytes):
                            pair_info["prep_cpu_ms"] = (
                                time.perf_counter() - prepare_started
                            ) * 1000.0
                            pair_info["expanded_mib"] = expanded_bytes / (1024 ** 2)
                            free_after, _ = torch.cuda.mem_get_info(x.device)
                            pair_info["driver_free_after_mib"] = free_after / (1024 ** 2)
                            pair_info["prepared"] = True
                            result = execute_chunks(prepared_weights)
                    except _WeightPairUnsupported as exc:
                        fallback_exc = exc
                    except Exception as exc:
                        if not _is_weight_pair_resource_error(exc):
                            raise
                        fallback_exc = exc
                    if fallback_exc is not None:
                        pair_info["fallback"] = True
                        pair_info["reason"] = type(fallback_exc).__name__
                        pair_stats["fallbacks"] += 1
                        fallback_key = (
                            x.device.index, block_index, type(fallback_exc).__name__,
                            str(fallback_exc)[:160],
                        )
                        if fallback_key not in _weight_pair_fallback_reported:
                            _weight_pair_fallback_reported.add(fallback_key)
                            LOGGER.warning(
                                "H3 Dynamic MLP weight-pair reuse fallback: "
                                "block=%d chunks=%d reason=%s. Restoring the "
                                "validated per-chunk cast path.",
                                block_index, chunks, str(fallback_exc),
                            )
                        if diagnostic_state is not None:
                            diagnostic_state.clear()
                        torch.cuda.synchronize(x.device)
                        torch.cuda.empty_cache()
                        result = execute_chunks()
                    else:
                        pair_stats["prepared"] += 1
                        pair_stats["reused_projection_calls"] += 2 * (chunks - 1)
                        pair_stats["prep_cpu_ms"] += pair_info["prep_cpu_ms"]
                        pair_stats["expanded_mib_peak"] = max(
                            pair_stats["expanded_mib_peak"],
                            pair_info["expanded_mib"],
                        )
            else:
                result = execute_chunks()

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
            LOGGER.info(
                "V100 diagnostics Dynamic MLP weight-pair reuse: block=%d, "
                "eligible=%s, prepared=%s, fallback=%s, reason=%s, chunks=%d, "
                "projection_prepares=%d, avoided_projection_prepares=%d, "
                "expanded_weight_pair=%.1f MiB, prep_CPU=%.3f ms, "
                "driver_free_before/after=%.1f/%.1f MiB, "
                "cumulative_prepared/fallback/skipped_floor=%d/%d/%d, "
                "cumulative_avoided_projection_prepares=%d.",
                block_index, pair_info["eligible"], pair_info["prepared"],
                pair_info["fallback"], pair_info["reason"], chunks,
                2 if pair_info["prepared"] else 0,
                2 * (chunks - 1) if pair_info["prepared"] else 0,
                pair_info["expanded_mib"], pair_info["prep_cpu_ms"],
                pair_info["driver_free_before_mib"],
                pair_info["driver_free_after_mib"],
                int(pair_stats.get("prepared", 0)),
                int(pair_stats.get("fallbacks", 0)),
                int(pair_stats.get("skipped_floor", 0)),
                int(pair_stats.get("reused_projection_calls", 0)),
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
    setattr(chunked_forward, CONTROLLER_ATTR, expected_dynamic_vbar_controller)
    return chunked_forward


class H3TokenwiseMLPChunking:
    """Adapter for H3's validated [tokens, hidden] token-independent MLPs."""

    def patch(self, model, chunk_tokens=512, cache_trim=True, cache_trim_threshold_mb=2048,
              adaptive=False, experimental_fp16=False, scaled_fp16_swiglu=False,
              dynamic_vbar_controller=None):
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
        transformer_options[SCALED_FP16_SWIGLU_KEY] = bool(
            experimental_fp16 and scaled_fp16_swiglu
        )
        if dynamic_vbar_controller is not None:
            transformer_options[CONTROLLER_KEY] = dynamic_vbar_controller
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
                    _make_forward(
                        base_forward, index, transformer_options,
                        dynamic_vbar_controller,
                    ), mlp
                ),
            )
            count += 1
        if dynamic_vbar_controller is not None:
            for index in range(count):
                key = f"diffusion_model.blocks.{index}.mlp.forward"
                function = getattr(patched.object_patches[key], "__func__", None)
                if getattr(function, CONTROLLER_ATTR, None) is not dynamic_vbar_controller:
                    raise RuntimeError(
                        "H3 native Dynamic VBAR integration check failed at "
                        f"MLP block {index}."
                    )
            LOGGER.info(
                "H3 native Dynamic VBAR policy integration armed: blocks=%d "
                "controller_shared=True vbar_cache_credit=False "
                "mlp_weight_pair_reuse=True driver_floor=%d MiB.",
                count,
                _MLP_WEIGHT_PAIR_DRIVER_FLOOR_MIB,
            )
        LOGGER.info(
            "H3 bounded-memory token-wise MLP active: blocks=%d, refreshed=%d, "
            "adaptive=%s, chunk_tokens=%d, "
            "cache_trim=%s, cache_trim_threshold=%d MiB. "
            "This reduces peak activation memory and may increase generation time.",
            count, refreshed, bool(adaptive), chunk_tokens, bool(cache_trim),
            max(0, int(cache_trim_threshold_mb)),
        )
        if experimental_fp16:
            if scaled_fp16_swiglu:
                LOGGER.warning(
                    "H3 validated scaled FP16 SwiGLU active: MLP fc1/SwiGLU/fc2 "
                    "use FP16 with branch scale=16 and fc2 scale=8."
                )
            else:
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
