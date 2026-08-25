"""Memory-bounded FP16 Sol-Attn reference path for MiniMax H3 validation.

This is deliberately a PyTorch reference implementation, not the final SM70
kernel.  It implements query-dependent block routing and centroid correction
without materializing a full token attention matrix.  The implementation is
useful for validating H3 quality policy and diagnostics before committing the
same math to the bundled CUTLASS extension.
"""

import logging
import math
import time

import torch

LOGGER = logging.getLogger("V100SolAttention")
DIAGNOSTICS_KEY = "_h3_v100_removed_diagnostics"
DIAGNOSTICS_INTERVAL_KEY = "_h3_v100_removed_diagnostics_interval"

MODE_KEY = "v100_attention_backend"
TAU_KEY = "v100_sol_attention_tau"
MIN_TOKENS_KEY = "v100_sol_attention_min_tokens"
BLOCK_SIZE_KEY = "v100_sol_attention_block_size"
PROBE_KEY = "v100_sol_attention_probe"
PREFIX_STOP_KEY = "v100_sol_attention_prefix_stop"
H3_VIDEO_GRID_KEY = "v100_sol_h3_video_grid"
BLOCK_INDEX_KEY = "v100_h3_block_index"
BLOCK_COUNT_KEY = "v100_h3_block_count"
SIGMA_START_KEY = "v100_sol_sigma_start"
SIGMA_END_KEY = "v100_sol_sigma_end"

MODE_AUTO = "auto"
MODE_FLASH = "flash_attn"
MODE_FLASH_FIXED = "flash_fixed"
MODE_SOL = "sol_attn"

_stats = {}
_probe_stats = {}
_morton_probe_reported = set()
_morton_perm_cache = {}


def support_reasons(
    q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs
):
    tensors = (q, k, v)
    reasons = []
    if mask is not None:
        reasons.append("mask-present")
    if not skip_reshape:
        reasons.append("layout-not-explicit-head")
    if skip_output_reshape:
        reasons.append("skip-output-reshape")
    if kwargs.get("enable_gqa", False):
        reasons.append("gqa-enabled")
    if not all(t.ndim == 4 for t in tensors):
        reasons.append("not-4d")
    if not all(t.is_cuda for t in tensors):
        reasons.append("not-cuda")
    if not all(t.dtype == torch.float16 for t in tensors):
        reasons.append("not-all-fp16")
    if any(t.requires_grad for t in tensors):
        reasons.append("requires-grad")
    if not (q.shape == k.shape == v.shape):
        reasons.append("qkv-shape-mismatch")
    if q.ndim == 4 and (q.shape[1] != heads or q.shape[-1] != 128):
        reasons.append("heads-or-head-dim-mismatch")
    if q.is_cuda and torch.cuda.get_device_capability(q.device) != (7, 0):
        reasons.append("not-sm70")
    return tuple(reasons)


def _block_centroids(x, block_size):
    """Compute small block statistics without padding/copying full K or V."""
    batch, heads, tokens, width = x.shape
    complete = tokens // block_size
    pieces = []
    if complete:
        body = x[:, :, :complete * block_size].view(
            batch, heads, complete, block_size, width
        )
        pieces.append(body.float().mean(dim=3))
    if complete * block_size < tokens:
        pieces.append(x[:, :, complete * block_size:].float().mean(dim=2, keepdim=True))
    return torch.cat(pieces, dim=2)


def _matmul_fp32_accum(left, right):
    """Low-precision GEMM with an FP32 result, matching Triton's accumulator.

    A plain torch.matmul on FP16 tensors returns FP16.  Its internal tensor-core
    accumulation may be FP32, but the pre-normalization result is rounded back
    to FP16 and can overflow before Sol's softmax denominator is applied.
    Triton's kernel keeps this value in FP32.  CUDA bmm's out_dtype preserves
    that behavior on the V100 path.
    """
    if left.is_cuda and left.dtype in (torch.float16, torch.bfloat16):
        prefix = left.shape[:-2]
        left_3d = left.reshape(-1, left.shape[-2], left.shape[-1])
        right_3d = right.reshape(-1, right.shape[-2], right.shape[-1])
        result = torch.bmm(left_3d, right_3d, out_dtype=torch.float32)
        return result.reshape(*prefix, left.shape[-2], right.shape[-1])
    return torch.matmul(left.float(), right.float())


def _native_routing_density(q, k_center, threshold, block_size, prefix_stop):
    """Reconstruct only the small block-routing matrix for diagnostics."""
    q_center = _block_centroids(q, block_size)
    route_scores = torch.matmul(
        q_center, k_center.transpose(-2, -1)
    )
    selected = route_scores > threshold.unsqueeze(-1)
    blocks = selected.shape[-1]
    indices = torch.arange(blocks, device=q.device)
    local = (indices[:, None] - indices[None, :]).abs() <= 1
    selected |= local.view(1, 1, blocks, blocks)
    prefix_blocks = min(blocks, math.ceil(prefix_stop / block_size))
    if prefix_blocks:
        selected[..., :prefix_blocks] = True
        selected[..., :prefix_blocks, :] = True
    density = selected.float().mean(dim=-1)
    return (
        float(selected.float().mean().item()),
        float(density.min().item()),
        float(density.max().item()),
    )


def _morton_permutation(grid, device, curve):
    """Return the H3 video-row Z-order used by the Triton reference project."""
    normalized = tuple(int(value) for value in grid)
    key = (normalized, str(curve))
    permutation = _morton_perm_cache.get(key)
    if permutation is None:
        frames, height, width = normalized
        linear = torch.arange(frames * height * width, dtype=torch.int64)
        area = height * width
        z = linear // area
        remainder = linear - z * area
        y = remainder // width
        x = remainder - y * width

        def part1by2(value):
            value = value & 0x1FFFFF
            value = (value | (value << 32)) & 0x1F00000000FFFF
            value = (value | (value << 16)) & 0x1F0000FF0000FF
            value = (value | (value << 8)) & 0x100F00F00F00F00F
            value = (value | (value << 4)) & 0x10C30C30C30C30C3
            return (value | (value << 2)) & 0x1249249249249249

        if curve == "2d_frame":
            code = (z << 42) | part1by2(x) | (part1by2(y) << 1)
        elif curve == "3d":
            code = part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)
        else:
            raise ValueError(f"Unsupported Morton curve: {curve}")
        permutation = linear[torch.argsort(code)]
        _morton_perm_cache[key] = permutation
    return permutation.to(device)


def _routing_selection(q_center, k_center, threshold, block_size, prefix_stop):
    route_scores = torch.matmul(q_center, k_center.transpose(-2, -1))
    selected = route_scores > threshold.unsqueeze(-1)
    blocks = selected.shape[-1]
    indices = torch.arange(blocks, device=q_center.device)
    selected |= ((indices[:, None] - indices[None, :]).abs() <= 1).view(
        1, 1, blocks, blocks
    )
    prefix_blocks = min(blocks, math.ceil(int(prefix_stop) / block_size))
    if prefix_blocks:
        selected[..., :prefix_blocks] = True
        selected[..., :prefix_blocks, :] = True
    return selected


def _routing_stats(selected):
    counts = selected.sum(dim=-1)
    blocks = selected.shape[-1]
    density = counts.float().div_(blocks)
    flat = density.flatten()
    quantiles = torch.quantile(
        flat, torch.tensor((0.1, 0.25, 0.5, 0.75, 0.9), device=flat.device)
    )
    previous = torch.zeros_like(selected)
    previous[..., 1:] = selected[..., :-1]
    run_count = int((selected & ~previous).sum().item())
    selected_count = int(counts.sum().item())
    return {
        "density": float(density.mean().item()),
        "density_min": float(density.min().item()),
        "density_max": float(density.max().item()),
        "all_dense_rows": int((counts == blocks).sum().item()),
        "rows": int(counts.numel()),
        "avg_run_length": selected_count / max(1, run_count),
        "quantiles": tuple(float(value) for value in quantiles.tolist()),
    }


def _morton_routing_probe(q, k, tau, block_size, prefix_stop, video_grid, curve):
    """Measure a Morton layout without changing the tensors returned to H3."""
    batch, _heads, tokens, _width = q.shape
    video_tokens = math.prod(int(value) for value in video_grid)
    video_start = int(prefix_stop)
    video_stop = video_start + video_tokens
    if batch != 1 or video_start < 0 or video_stop != tokens:
        return None

    video_order = _morton_permutation(video_grid, q.device, curve)
    pad = (-video_start) % block_size
    if pad:
        video_order = torch.roll(video_order, pad)
    order = torch.arange(tokens, device=q.device)
    order[video_start:video_stop] = video_order + video_start

    q_reordered = q.index_select(2, order)
    q_center = _block_centroids(q_reordered, block_size)
    del q_reordered
    k_reordered = k.index_select(2, order)
    k_center = _block_centroids(k_reordered, block_size)
    del k_reordered

    k_mean = k_center.mean(dim=2)
    k_var = (k_center - k_mean.unsqueeze(2)).square().mean(dim=2)
    threshold_mean = (q_center * k_mean.unsqueeze(2)).sum(dim=-1)
    threshold_variance = (
        q_center.square() * k_var.unsqueeze(2)
    ).sum(dim=-1)
    threshold = threshold_mean + float(tau) * (
        threshold_variance.clamp_min_(0.0) + 1e-6
    ).sqrt()
    selected = _routing_selection(
        q_center, k_center, threshold, block_size, prefix_stop
    )
    return _routing_stats(selected)


def _report_morton_probe(q, k, tau, block_size, prefix_stop, video_grid):
    key = (
        q.device.type, q.device.index, int(q.shape[2]), tuple(video_grid),
        int(block_size), float(tau), int(prefix_stop),
    )
    if key in _morton_probe_reported:
        return
    _morton_probe_reported.add(key)
    for curve in ("2d_frame", "3d"):
        stats = _morton_routing_probe(
            q, k, tau, block_size, prefix_stop, video_grid, curve
        )
        if stats is None:
            LOGGER.info(
                "V100 diagnostics Sol Morton probe skipped: curve=%s, tokens=%d, "
                "prefix_stop=%d, video_grid=%s.",
                curve, int(q.shape[2]), int(prefix_stop), tuple(video_grid),
            )
            continue
        LOGGER.info(
            "V100 diagnostics Sol Morton routing probe: curve=%s, "
            "diagnostic_only=True, video_grid=%s, density=%.4f, "
            "density_min/max=%.4f/%.4f, density_p10/p25/p50/p75/p90="
            "%.4f/%.4f/%.4f/%.4f/%.4f, all_dense_rows=%d/%d, "
            "avg_selected_run_length=%.3f.",
            curve, tuple(video_grid), stats["density"], stats["density_min"],
            stats["density_max"], *stats["quantiles"],
            stats["all_dense_rows"], stats["rows"],
            stats["avg_run_length"],
        )


def sol_attention_reference(
    q, k, v, *, tau=1.0, block_size=64, prefix_stop=0,
    exact_prefix_queries=True, return_stats=False, prepared=None,
):
    """Return approximate attention for explicit-head ``[B,H,T,D]`` tensors.

    Selected blocks use exact token attention.  Unselected blocks contribute a
    centroid score/value with the real block length in the shared softmax
    numerator and denominator.  Only one query block is expanded at a time.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Sol-Attn reference expects equal [B,H,T,D] Q/K/V")
    if q.shape[-1] != 128:
        raise ValueError("Sol-Attn reference requires head_dim=128")
    if block_size not in (64, 128, 256):
        raise ValueError("Sol-Attn reference block_size must be 64, 128, or 256")

    batch, heads, tokens, width = q.shape
    scale = width ** -0.5
    block_count = math.ceil(tokens / block_size)
    block_starts = torch.arange(block_count, device=q.device) * block_size
    lengths = (tokens - block_starts).clamp(min=0, max=block_size).float()
    if prepared is None:
        k_center = _block_centroids(k, block_size)
        v_center = _block_centroids(v, block_size)
        prepared_threshold = None
    else:
        k_center, v_center, prepared_threshold = prepared
    # Match the validated Sol routing preprocess: estimate each query block's
    # threshold from the per-channel diagonal variance of K centroids. This is
    # intentionally not the variance of the already projected route scores;
    # the two select different blocks when K channels are correlated.
    k_center_mean = k_center.mean(dim=2)
    k_center_var_diag = (
        k_center - k_center_mean.unsqueeze(2)
    ).square().mean(dim=2)
    # Centroids are reused by every query block.  Keep the tensor-core inputs in
    # the source dtype while thresholds and online-softmax state remain FP32.
    k_center_mm = k_center.to(q.dtype)
    v_center_mm = v_center.to(v.dtype)
    prefix_blocks = min(
        block_count, max(0, math.ceil(int(prefix_stop) / block_size))
    )

    output = torch.empty_like(q)
    selected_total = 0
    selected_slots = 0
    density_min = 1.0
    density_max = 0.0

    for query_block, start in enumerate(range(0, tokens, block_size)):
        stop = min(tokens, start + block_size)
        query = q[:, :, start:stop]
        query_center = query.float().mean(dim=2).to(q.dtype)
        route_scores = _matmul_fp32_accum(
            query_center.unsqueeze(2), k_center_mm.transpose(-2, -1)
        ).squeeze(2)
        query_center_fp32 = query.float().mean(dim=2)
        if prepared_threshold is None:
            threshold_mean = (
                query_center_fp32 * k_center_mean
            ).sum(dim=-1, keepdim=True)
            threshold_variance = (
                query_center_fp32.square() * k_center_var_diag
            ).sum(dim=-1, keepdim=True)
            threshold = threshold_mean + float(tau) * (
                threshold_variance.clamp_min_(0.0) + 1e-6
            ).sqrt()
        else:
            threshold = prepared_threshold[:, :, query_block].unsqueeze(-1)
        selected = route_scores > threshold
        local_start = max(0, query_block - 1)
        local_stop = min(block_count, query_block + 2)
        selected[:, :, local_start:local_stop] = True
        if prefix_blocks:
            selected[:, :, :prefix_blocks] = True
        # H3 packs text/reference/audio before video. Keeping those query rows
        # dense prevents approximate video state from feeding back into audio.
        # The one boundary block may include a few video rows; over-selecting it
        # is conservative and avoids splitting a kernel block.
        if exact_prefix_queries and start < int(prefix_stop):
            selected.fill_(True)

        counts = selected.sum(dim=-1)
        density = counts.float() / block_count
        density_min = min(density_min, float(density.min().item()))
        density_max = max(density_max, float(density.max().item()))
        selected_total += int(counts.sum().item())
        selected_slots += counts.numel() * block_count
        max_selected = max(1, int(counts.max().item()))

        # Selected indices are packed to the front independently for every head.
        order = torch.argsort(selected.to(torch.int8), dim=-1, descending=True)
        indices = order[:, :, :max_selected]
        slot_valid = torch.arange(max_selected, device=q.device).view(1, 1, -1)
        slot_valid = slot_valid < counts.unsqueeze(-1)
        token_index = indices[..., None] * block_size + torch.arange(
            block_size, device=q.device
        ).view(1, 1, 1, -1)
        exact_valid = slot_valid[..., None] & (token_index < tokens)
        gather_tokens = token_index.clamp(max=tokens - 1).reshape(batch, heads, -1)
        gather_tokens = gather_tokens[..., None].expand(-1, -1, -1, width)
        exact_k = torch.gather(k, 2, gather_tokens)
        exact_v = torch.gather(v, 2, gather_tokens)
        exact_valid = exact_valid.reshape(batch, heads, -1)
        exact_logits = _matmul_fp32_accum(
            query, exact_k.transpose(-2, -1)
        ).mul_(scale)
        exact_logits.masked_fill_(~exact_valid.unsqueeze(2), -torch.inf)

        approximate_logits = _matmul_fp32_accum(
            query, k_center_mm.transpose(-2, -1)
        ).mul_(scale)
        approximate_logits.masked_fill_(selected.unsqueeze(2), -torch.inf)

        row_max = torch.maximum(
            exact_logits.amax(dim=-1), approximate_logits.amax(dim=-1)
        ).unsqueeze(-1)
        exact_weight = torch.exp(exact_logits - row_max)
        exact_weight.masked_fill_(~exact_valid.unsqueeze(2), 0.0)
        approximate_weight = torch.exp(approximate_logits - row_max)
        approximate_weight.masked_fill_(selected.unsqueeze(2), 0.0)
        approximate_weight.mul_(lengths.view(1, 1, 1, -1))

        denominator = exact_weight.sum(dim=-1, keepdim=True)
        denominator.add_(approximate_weight.sum(dim=-1, keepdim=True))
        exact_out = _matmul_fp32_accum(exact_weight.to(v.dtype), exact_v)
        approximate_out = _matmul_fp32_accum(
            approximate_weight.to(v.dtype), v_center_mm
        )
        output[:, :, start:stop].copy_(
            ((exact_out + approximate_out) / denominator.clamp_min_(1e-20)).to(
                output.dtype
            )
        )

    details = {
        "tokens": tokens,
        "block_size": block_size,
        "blocks": block_count,
        "prefix_blocks": prefix_blocks,
        "exact_prefix_queries": bool(exact_prefix_queries),
        "threshold": "diag_k_centroid_variance",
        "exact_density": selected_total / max(1, selected_slots),
        "density_min": density_min,
        "density_max": density_max,
    }
    result = output.transpose(1, 2).reshape(batch, tokens, heads * width)
    return (result, details) if return_stats else result


def _record(
    q, details, elapsed_ms, cuda_ms, block_index, mode, interval
):
    key = (
        q.device.type, q.device.index, int(q.shape[2]), int(details["block_size"]),
        float(details["tau"]), mode,
    )
    state = _stats.setdefault(
        key, {"calls": 0, "elapsed_ms": 0.0, "cuda_ms": 0.0}
    )
    state["calls"] += 1
    state["elapsed_ms"] += elapsed_ms
    state["cuda_ms"] += cuda_ms
    stage_ms = details.get("stage_ms") or {}
    stage_totals = state.setdefault("stage_ms", {})
    for name, value in stage_ms.items():
        stage_totals[name] = stage_totals.get(name, 0.0) + value
    if state["calls"] == 1 or (interval > 0 and state["calls"] % interval == 0):
        allocated = torch.cuda.memory_allocated(q.device) / (1024 ** 2)
        reserved = torch.cuda.memory_reserved(q.device) / (1024 ** 2)
        peak = torch.cuda.max_memory_allocated(q.device) / (1024 ** 2)
        stage_text = ""
        if stage_ms:
            stage_text = ", stages=" + "/".join(
                "%s=%.3f(avg %.3f)" % (
                    name, value, stage_totals[name] / state["calls"]
                )
                for name, value in stage_ms.items()
            ) + " ms"
        route_text = ""
        route = details.get("route_diagnostics")
        if route:
            quantiles = route["tail_nnz_quantiles"]
            bands = route["tail_bitword_popcount_bands"]
            route_text = (
                ", sparse_tail_rows/blocks=%d/%d, tail_density=%.4f, "
                "tail_nnz_mean/std/cv=%.2f/%.2f/%.3f, "
                "tail_nnz_p0/p10/p25/p50/p75/p90/p100=%s, "
                "tail_dense_rows=%d, selected_runs=%d, avg_run=%.3f, "
                "bitword_mean_popcount=%.3f, bitword_empty/full=%.4f/%.4f, "
                "bitword_bands_0/1_8/9_16/17_24/25_31/32=%s"
            ) % (
                route["tail_rows"], route["tail_blocks"], route["tail_density"],
                route["tail_nnz_mean"], route["tail_nnz_std"], route["tail_nnz_cv"],
                "/".join("%.0f" % value for value in quantiles),
                route["tail_all_dense_rows"], route["tail_run_count"],
                route["tail_avg_run_length"], route["tail_bitword_mean_popcount"],
                route["tail_bitword_empty_fraction"], route["tail_bitword_full_fraction"],
                "/".join(str(value) for value in bands),
            )
        LOGGER.info(
            "V100 diagnostics Sol-Attn reference: mode=%s, block=%s, calls=%d, "
            "sequence=%d, heads=%d, block_size=%d, blocks=%d, tau=%.3f, "
            "prefix_stop=%d, prefix_blocks=%d, exact_density=%.4f, "
            "density_min/max=%.4f/%.4f, CPU=%.3f ms, CUDA=%.3f ms, "
            "CPU_avg=%.3f ms, CUDA_avg=%.3f ms, allocated=%.1f MiB, "
            "reserved=%.1f MiB, peak=%.1f MiB, implementation=%s%s%s.",
            mode, block_index if block_index is not None else "unknown",
            state["calls"], details["tokens"], int(q.shape[1]),
            details["block_size"], details["blocks"], details["tau"],
            details["prefix_stop"], details["prefix_blocks"],
            details["exact_density"], details["density_min"],
            details["density_max"], elapsed_ms, cuda_ms,
            state["elapsed_ms"] / state["calls"],
            state["cuda_ms"] / state["calls"], allocated, reserved, peak,
            details.get("implementation", "pytorch_reference"), stage_text,
            route_text,
        )


def run_reference(
    q, k, v, transformer_options, *, exact_backend=None, mode=MODE_SOL
):
    tau = float(transformer_options.get(TAU_KEY, 1.0))
    block_size = int(transformer_options.get(BLOCK_SIZE_KEY, 64))
    prefix_stop = int(transformer_options.get(PREFIX_STOP_KEY, 0) or 0)
    block_index = transformer_options.get(BLOCK_INDEX_KEY)
    diagnostics = False
    probe = bool(transformer_options.get(PROBE_KEY, False))
    interval = max(
        0, int(transformer_options.get(DIAGNOSTICS_INTERVAL_KEY, 50))
    )
    before = time.perf_counter()
    cuda_start = cuda_end = None
    if diagnostics:
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cuda_start.record()
    native = None
    try:
        from . import backend
        if q.is_cuda:
            native = backend.try_native_sol_attention(
                q, k, v, tau=tau, block_size=block_size,
                prefix_stop=prefix_stop,
                profile=diagnostics or probe,
            )
    except (AttributeError, RuntimeError, OSError) as error:
        if int(q.shape[2]) >= 38_000:
            raise RuntimeError(
                "Native Sol-Attn failed for a long sequence; refusing the "
                "unbounded PyTorch reference path. "
                f"tokens={int(q.shape[2])}, block_size={block_size}, "
                f"native_error={type(error).__name__}: {error}"
            ) from error
        LOGGER.warning(
            "Native Sol-Attn unavailable; using the PyTorch reference for "
            "tokens=%d: %s: %s",
            int(q.shape[2]), type(error).__name__, error,
        )
    if native is None:
        if int(q.shape[2]) >= 38_000:
            raise RuntimeError(
                "Native Sol-Attn operators are unavailable for a long sequence; "
                "the PyTorch reference path is intentionally disabled. "
                f"tokens={int(q.shape[2])}, block_size={block_size}"
            )
        result, details = sol_attention_reference(
            q, k, v, tau=tau, block_size=block_size,
            prefix_stop=prefix_stop, exact_prefix_queries=True, return_stats=True,
        )
        implementation = "pytorch_reference"
    else:
        result, _kc, _threshold, stage_events, route_stats = native
        blocks = math.ceil(q.shape[2] / block_size)
        density = density_min = density_max = float("nan")
        if route_stats is not None:
            if isinstance(route_stats, dict):
                density = route_stats["density"]
                density_min = route_stats["density_min"]
                density_max = route_stats["density_max"]
            else:
                density, density_min, density_max = route_stats
        details = {
            "tokens": int(q.shape[2]), "block_size": block_size,
            "blocks": blocks,
            "prefix_blocks": min(blocks, math.ceil(prefix_stop / block_size)),
            "exact_prefix_queries": True,
            "threshold": "diag_k_centroid_variance",
            "exact_density": density,
            "density_min": density_min, "density_max": density_max,
        }
        if isinstance(route_stats, dict) and "tail_rows" in route_stats:
            details["route_diagnostics"] = route_stats
        implementation = (
            "sm70_cuda_hybrid" if hasattr(
                torch.ops.comfy_v100_flash_attn_cuda, "sol_sparse_exact"
            ) else "sm70_cuda_fused"
        )
    if cuda_end is not None:
        cuda_end.record()
        cuda_end.synchronize()
    elapsed_ms = (time.perf_counter() - before) * 1000.0
    cuda_ms = cuda_start.elapsed_time(cuda_end) if cuda_end is not None else 0.0
    details.update({"tau": tau, "prefix_stop": prefix_stop})
    if native is not None and stage_events is not None:
        details["stage_ms"] = {
            name: start.elapsed_time(end)
            for name, (start, end) in stage_events.items()
        }
    details["implementation"] = implementation
    if diagnostics:
        _record(q, details, elapsed_ms, cuda_ms, block_index, mode, interval)

    if probe and exact_backend is not None:
        exact = exact_backend()
        sol = result.float()
        reference = exact.float()
        delta = sol - reference
        rel_l2 = float(delta.norm() / reference.norm().clamp_min(1e-20))
        cosine = float(torch.nn.functional.cosine_similarity(
            sol.flatten(), reference.flatten(), dim=0
        ))
        max_abs = float(delta.abs().max())
        stats_key = (
            q.device.type, q.device.index, tuple(q.shape), block_size, tau,
            prefix_stop, int(block_index) if block_index is not None else -1,
        )
        entry = _probe_stats.setdefault(
            stats_key, {"calls": 0, "rel_l2": 0.0, "max_abs": 0.0,
                        "density": 0.0},
        )
        entry["calls"] += 1
        entry["rel_l2"] += rel_l2
        entry["max_abs"] = max(entry["max_abs"], max_abs)
        entry["density"] += details["exact_density"]
        LOGGER.warning(
            "V100 diagnostics Sol-Attn clean-input probe: sequence=%d, block=%s, "
            "tau=%.3f, block_size=%d, exact_density=%.4f, rel_l2=%.6g, "
            "cosine=%.8f, max_abs=%.6g, samples=%d. Returning exact Flash "
            "Attention so sparse error cannot contaminate later blocks.",
            int(q.shape[2]), block_index if block_index is not None else "unknown",
            tau, block_size, details["exact_density"], rel_l2, cosine, max_abs,
            entry["calls"],
        )
        block_count = transformer_options.get(BLOCK_COUNT_KEY)
        if block_count is not None and block_index == int(block_count) - 2:
            matching = []
            prefix = stats_key[:6]
            for key, values in _probe_stats.items():
                if key[:6] == prefix:
                    matching.append((
                        values["rel_l2"] / values["calls"], key[-1],
                        values["density"] / values["calls"],
                        values["max_abs"], values["calls"],
                    ))
            matching.sort(reverse=True)
            LOGGER.warning(
                "V100 diagnostics Sol-Attn block sensitivity (worst first): %s",
                "; ".join(
                    "block=%d rel_l2=%.6g density=%.4f max_abs=%.6g samples=%d" %
                    (index, error, density, peak, calls)
                    for error, index, density, peak, calls in matching
                ),
            )
        return exact
    return result
