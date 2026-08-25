import logging
import math
from pathlib import Path

import torch

_loaded = False
_cu_seqlens_cache = {}
_sol_split_reported = set()
_sol_k_prefix_split_reported = set()
_sol_k_prefix_chunk_reported = set()
_sol_csr_reported = set()
_sol_route_detail_reported = set()
_sol_csr_centroid_reported = set()
LOGGER = logging.getLogger("V100FlashBackend")

# The K-prefix partial result is mergeable one query range at a time.  Keeping
# the entire tail result live alongside the full sparse output costs about 1.3
# GiB at H3's 102,970-token validation shape.
_SOL_DENSE_TAIL_CHUNK_TOKENS = 4096
_SOL_DENSE_TAIL_LARGE_CHUNK_TOKENS = 8192
_SOL_DENSE_TAIL_MIN_FREE_MIB = 2048


def _select_sol_dense_tail_chunk(q, tokens):
    """Choose a larger merge chunk only when driver free VRAM has headroom."""
    if tokens <= _SOL_DENSE_TAIL_CHUNK_TOKENS or not q.is_cuda:
        return _SOL_DENSE_TAIL_CHUNK_TOKENS, "baseline-or-cpu"
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(q.device)
    except (RuntimeError, AttributeError):
        return _SOL_DENSE_TAIL_CHUNK_TOKENS, "memory-query-unavailable"
    free_mib = free_bytes / (1024 ** 2)
    if free_mib < _SOL_DENSE_TAIL_MIN_FREE_MIB:
        return _SOL_DENSE_TAIL_CHUNK_TOKENS, "free-vram-below-2048-mib"
    return _SOL_DENSE_TAIL_LARGE_CHUNK_TOKENS, "free-vram-headroom"


def load_extension():
    global _loaded
    if _loaded:
        return
    # H3_V100 and ComfyUI-V100 ship the same Torch extension. Windows cannot
    # initialize a second physical copy of that DLL in one process, even though
    # both copies have identical contents. Reuse the already registered op.
    if hasattr(torch.ops.comfy_v100_flash_attn_cuda, "varlen_fwd"):
        _loaded = True
        return
    binaries = list(Path(__file__).resolve().parent.glob("comfy_v100_flash_attn_cuda*.pyd"))
    if len(binaries) != 1:
        raise RuntimeError(
            "Expected exactly one comfy_v100_flash_attn_cuda CPython extension, "
            f"found {len(binaries)}."
        )
    torch.ops.load_library(str(binaries[0]))
    _loaded = True


def has_lse_free_inference():
    """Whether the loaded extension exposes the inference-only Flash API."""
    load_extension()
    return hasattr(
        torch.ops.comfy_v100_flash_attn_cuda, "varlen_fwd_inference"
    )


def _varlen_forward_inference(
    q, k, v, out, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, softmax_scale,
):
    """Run exact non-causal inference without materializing final global LSE."""
    ops = torch.ops.comfy_v100_flash_attn_cuda
    if max_seqlen_q > 1 and hasattr(ops, "varlen_fwd_inference"):
        return ops.varlen_fwd_inference(
            q, k, v, out, cu_seqlens_q, cu_seqlens_k,
            int(max_seqlen_q), int(max_seqlen_k), float(softmax_scale),
        )
    result, _lse, *_ = ops.varlen_fwd(
        q, k, v, out, cu_seqlens_q, cu_seqlens_k,
        None, None, None, None,
        int(max_seqlen_q), int(max_seqlen_k), 0.0, float(softmax_scale),
        False, False, -1, -1, 0.0, False, 1, None,
    )
    return result


def _varlen_forward(
    q, k, v, out, cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k, softmax_scale,
):
    """Established exact varlen path; experimental ops stay benchmark-only."""
    result, _lse, *_ = torch.ops.comfy_v100_flash_attn_cuda.varlen_fwd(
        q, k, v, out, cu_seqlens_q, cu_seqlens_k,
        None, None, None, None,
        int(max_seqlen_q), int(max_seqlen_k), 0.0, float(softmax_scale),
        False, False, -1, -1, 0.0, False, 1, None,
    )
    return result


def _varlen_partial_head_layout(q, k, v, softmax_scale):
    """Return normalized output and LSE for rectangular explicit-head inputs."""
    batch, heads, q_tokens, width = q.shape
    k_tokens = k.shape[2]
    q_flat = q.permute(0, 2, 1, 3).contiguous().view(
        batch * q_tokens, heads, width
    )
    k_flat = k.permute(0, 2, 1, 3).contiguous().view(
        batch * k_tokens, heads, width
    )
    v_flat = v.permute(0, 2, 1, 3).contiguous().view(
        batch * k_tokens, heads, width
    )
    out, lse, *_ = torch.ops.comfy_v100_flash_attn_cuda.varlen_fwd(
        q_flat, k_flat, v_flat, None,
        _cu_seqlens(batch, q_tokens, q.device),
        _cu_seqlens(batch, k_tokens, q.device),
        None, None, None, None,
        int(q_tokens), int(k_tokens), 0.0, float(softmax_scale),
        False, False, -1, -1, 0.0, False, 1, None,
    )
    out = out.view(batch, q_tokens, heads, width).permute(0, 2, 1, 3)
    return out, lse[..., :q_tokens]


def _dense_rect_head_layout(q, k, v, query_start, query_tokens, key_tokens,
                            softmax_scale):
    """Run rectangular dense Flash directly from contiguous BHDT tensors."""
    ops = torch.ops.comfy_v100_flash_attn_cuda
    if hasattr(ops, "sol_dense_rect"):
        return ops.sol_dense_rect(
            q, k, v, int(query_start), int(query_tokens), int(key_tokens),
            float(softmax_scale),
        )
    return _varlen_partial_head_layout(
        q[:, :, query_start:query_start + query_tokens],
        k[:, :, :key_tokens], v[:, :, :key_tokens], softmax_scale,
    )


def sol_prepare(q, k, v, tau=1.0, block_size=64):
    """Native SM70 Sol block statistics in explicit-head [B,H,T,128] layout."""
    load_extension()
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise RuntimeError("native Sol-Attn requires contiguous head dimensions")
    return torch.ops.comfy_v100_flash_attn_cuda.sol_prepare(
        q, k, v, float(tau), int(block_size)
    )


def try_sol_prepare(q, k, v, tau=1.0, block_size=64):
    """Return native Sol statistics, or None with an older installed binary."""
    load_extension()
    if not hasattr(torch.ops.comfy_v100_flash_attn_cuda, "sol_prepare"):
        return None
    return sol_prepare(q, k, v, tau=tau, block_size=block_size)


def try_native_sol_attention(
    q, k, v, *, tau=1.0, block_size=64, prefix_stop=0, scale=None,
    profile=False,
):
    """Run the fused SM70 Sol forward, or return None for an older binary."""
    load_extension()
    ops = torch.ops.comfy_v100_flash_attn_cuda
    if not hasattr(ops, "sol_prepare"):
        return None
    if block_size != 64:
        return None
    # H3 exposes [B,H,T,D] views over token-major BTHD storage.  The SM70
    # Sol kernels carry explicit row/head strides, so forcing BHDT contiguity
    # here would create one full sequence copy for each of Q/K/V.
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise RuntimeError("native Sol-Attn requires contiguous head dimensions")
    stage_events = None
    route_stats = None
    if profile:
        names = ("prepare", "route", "sparse_exact", "centroid_merge")
        stage_events = {
            name: (torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True))
            for name in names
        }
        stage_events["prepare"][0].record()
    qc, kc, vc, threshold = ops.sol_prepare(
        q, k, v, float(tau), int(block_size)
    )
    if stage_events is not None:
        stage_events["prepare"][1].record()
    softmax_scale = float(scale if scale is not None else 1.0 / math.sqrt(q.shape[-1]))
    if hasattr(ops, "sol_sparse_exact"):
        if stage_events is not None:
            stage_events["route"][0].record()
        batch, heads, tokens, width = q.shape
        blocks = math.ceil(tokens / block_size)
        lengths = (
            tokens - torch.arange(blocks, device=q.device) * block_size
        ).clamp(min=0, max=block_size).float()
        # Avoid keeping the full FP32 [B,H,blocks,blocks] route-score matrix
        # live on long sequences.  The GEMM/threshold semantics are unchanged;
        # each query-block slice is written directly into the final mask.
        route_chunk_blocks = 512 if blocks > 1024 else blocks
        selected = torch.empty(
            (batch, heads, blocks, blocks), device=q.device, dtype=torch.bool
        )
        kc_t = kc.transpose(-2, -1)
        threshold_t = threshold.unsqueeze(-1)
        for route_start in range(0, blocks, route_chunk_blocks):
            route_stop = min(blocks, route_start + route_chunk_blocks)
            route_scores = torch.matmul(
                qc[:, :, route_start:route_stop], kc_t
            )
            selected[:, :, route_start:route_stop].copy_(
                route_scores > threshold_t[:, :, route_start:route_stop]
            )
            del route_scores
        del qc, kc_t, threshold_t
        block_ids = torch.arange(blocks, device=q.device)
        selected |= (
            (block_ids.view(-1, 1) - block_ids.view(1, -1)).abs() <= 1
        ).view(1, 1, blocks, blocks)
        prefix_blocks = min(blocks, max(0, math.ceil(int(prefix_stop) / block_size)))
        if prefix_blocks:
            selected[..., :prefix_blocks] = True
            selected[..., :prefix_blocks, :] = True
        if profile:
            # `selected` is already needed by the centroid merge.  Capture
            # diagnostics now instead of reconstructing FP32 Q centroids and
            # the full route score matrix after this native call returns.
            selected_per_query = selected.sum(dim=-1, dtype=torch.int32)
            route_stats = {
                "density": float(selected_per_query.float().mean().item() / blocks),
                "density_min": float(selected_per_query.min().item() / blocks),
                "density_max": float(selected_per_query.max().item() / blocks),
            }
            detail_key = (
                q.device.type, q.device.index, tokens, heads, prefix_blocks
            )
            if detail_key not in _sol_route_detail_reported:
                _sol_route_detail_reported.add(detail_key)
        def packed_route(route, tail_prefix_blocks=0):
            if tail_prefix_blocks and hasattr(ops, "sol_pack_route_tail"):
                return ops.sol_pack_route_tail(
                    route.contiguous(), block_size, tail_prefix_blocks
                )
            if hasattr(ops, "sol_pack_route"):
                return ops.sol_pack_route(route.contiguous(), block_size)
            if tokens >= 38_000:
                raise RuntimeError(
                    "the loaded CUDA extension lacks sol_pack_route; restart "
                    "ComfyUI with the current ComfyUI-V100 binary and remove "
                    "older packages that register the same Torch operator namespace"
                )
            counts = route.sum(dim=-1, dtype=torch.int32).contiguous()
            offsets = torch.argsort(
                route.to(torch.int8), dim=-1, descending=True, stable=True
            ).to(torch.int32).mul_(block_size).contiguous()
            return counts, offsets

        use_csr_route = bool(
            hasattr(ops, "sol_pack_route_csr")
            and hasattr(ops, "sol_sparse_exact_csr")
        )

        use_k_prefix_split = bool(
            prefix_blocks >= 2
            and prefix_blocks < blocks
            and (hasattr(ops, "sol_sparse_exact_tail") or use_csr_route)
        )
        if use_k_prefix_split:
            if (not hasattr(ops, "sol_pack_route_tail") and not use_csr_route
                    and tokens >= 38_000):
                raise RuntimeError(
                    "the loaded CUDA extension lacks sol_pack_route_tail; restart "
                    "ComfyUI with the current ComfyUI-V100 binary"
                )
            if use_csr_route:
                counts, offsets = ops.sol_pack_route_csr(
                    selected.contiguous(), block_size, prefix_blocks
                )
            elif hasattr(ops, "sol_pack_route_tail"):
                counts, offsets = packed_route(selected, prefix_blocks)
            else:
                # Compatibility-only path for pre-tail-pack binaries at small
                # shapes. Long sequences reject it above rather than restoring
                # the route-mask clone peak.
                tail_selected = selected.clone()
                tail_selected[..., prefix_blocks:, :prefix_blocks] = False
                counts, offsets = packed_route(tail_selected)
                del tail_selected
        elif use_csr_route:
            counts, offsets = ops.sol_pack_route_csr(
                selected.contiguous(), block_size, 0
            )
        else:
            counts, offsets = packed_route(selected)
        csr_route = use_csr_route
        use_csr_centroid_merge = bool(
            csr_route and hasattr(ops, "sol_centroid_merge_csr")
        )
        if use_csr_centroid_merge and profile:
            centroid_key = (q.device.type, q.device.index, tokens, prefix_blocks)
            if centroid_key not in _sol_csr_centroid_reported:
                _sol_csr_centroid_reported.add(centroid_key)
                LOGGER.info(
                    "V100 diagnostics Sol CSR centroid merge active: "
                    "tokens=%d, prefix_blocks=%d, bool_route_released=True.",
                    tokens, prefix_blocks,
                )
        if profile and csr_route:
            report_key = (q.device.type, q.device.index, tokens)
            if report_key not in _sol_csr_reported:
                _sol_csr_reported.add(report_key)
                LOGGER.info(
                    "V100 diagnostics Sol CSR route metadata active: "
                    "tokens=%d, packed_offsets=%d, "
                    "omitted_dense_prefix_entries=%d.",
                    tokens, offsets.numel(),
                    batch * heads * prefix_blocks * blocks,
                )
        if stage_events is not None:
            stage_events["route"][1].record()
            stage_events["sparse_exact"][0].record()
        if use_csr_centroid_merge:
            del selected
        use_dense_prefix_split = bool(
            prefix_blocks >= 2 and hasattr(ops, "sol_sparse_exact_split")
        )
        if use_k_prefix_split:
            report_key = (q.device.type, q.device.index, tokens, prefix_blocks)
            if profile and report_key not in _sol_k_prefix_split_reported:
                _sol_k_prefix_split_reported.add(report_key)
                LOGGER.info(
                    "V100 diagnostics Sol K-prefix split active: "
                    "tokens=%d, prefix_blocks=%d, prefix_tokens=%d, "
                    "tail_query_blocks=%d.",
                    tokens, prefix_blocks, prefix_blocks * block_size,
                    blocks - prefix_blocks,
                )
            prefix_tokens = prefix_blocks * block_size
            use_csr_range = bool(
                csr_route and hasattr(ops, "sol_sparse_exact_csr_range")
            )
            if use_csr_range:
                # Stream sparse exact by query range.  The native range op
                # keeps CSR rows global while returning chunk-local output;
                # merge then releases each sparse chunk before the next one.
                tail_chunk_tokens, tail_chunk_reason = _select_sol_dense_tail_chunk(
                    q, tokens
                )
                exact_out = torch.empty_like(q)
                exact_lse = torch.empty(
                    (batch, heads, tokens), device=q.device, dtype=torch.float32
                )
                prefix_out, prefix_lse = _dense_rect_head_layout(
                    q, k, v, 0, prefix_tokens, tokens, softmax_scale
                )
                exact_out[:, :, :prefix_tokens].copy_(prefix_out)
                exact_lse[:, :, :prefix_tokens].copy_(prefix_lse)
                del prefix_out, prefix_lse
                for tail_start in range(prefix_tokens, tokens, tail_chunk_tokens):
                    tail_tokens = min(tail_chunk_tokens, tokens - tail_start)
                    query_block_start = tail_start // block_size
                    query_block_count = (tail_tokens + block_size - 1) // block_size
                    sparse_chunk_out, sparse_chunk_lse = ops.sol_sparse_exact_csr_range(
                        q, k, v, counts, offsets, softmax_scale,
                        query_block_start, query_block_count,
                    )
                    dense_tail_out, dense_tail_lse = _dense_rect_head_layout(
                        q, k, v, tail_start, tail_tokens,
                        prefix_tokens, softmax_scale,
                    )
                    # The final query block may be padded to block_size by the
                    # native kernel; merge only the real token rows.
                    sparse_chunk_out = sparse_chunk_out[:, :, :tail_tokens]
                    sparse_chunk_lse = sparse_chunk_lse[:, :, :tail_tokens]
                    ops.sol_merge_partial_tail(
                        dense_tail_out, dense_tail_lse,
                        sparse_chunk_out, sparse_chunk_lse, 0,
                    )
                    exact_out[:, :, tail_start:tail_start + tail_tokens].copy_(
                        sparse_chunk_out
                    )
                    exact_lse[:, :, tail_start:tail_start + tail_tokens].copy_(
                        sparse_chunk_lse
                    )
                    del (
                        sparse_chunk_out, sparse_chunk_lse,
                        dense_tail_out, dense_tail_lse,
                    )
                if not use_csr_centroid_merge:
                    del counts, offsets
                if profile and report_key not in _sol_k_prefix_chunk_reported:
                    _sol_k_prefix_chunk_reported.add(report_key)
                    LOGGER.info(
                        "V100 diagnostics Sol K-prefix tail chunking active: "
                        "tokens=%d, selected_chunk_tokens=%d, reason=%s, "
                        "native_query_range=True.",
                        tokens, tail_chunk_tokens, tail_chunk_reason,
                    )
                # Skip the legacy full sparse-output path below.
                sparse_out = sparse_lse = None
            else:
                sparse_out, sparse_lse = (ops.sol_sparse_exact_csr(
                    q, k, v, counts, offsets, softmax_scale, prefix_blocks,
                ) if csr_route else ops.sol_sparse_exact_tail(
                    q, k, v, counts, offsets, softmax_scale, prefix_blocks,
                ))
            # The native kernel has consumed the route metadata.  Do not keep
            # the potentially large CSR offset tensor live during the dense
            # K-prefix tail merge.
            if not use_csr_range and not use_csr_centroid_merge:
                del counts, offsets
            if (not use_csr_range) and hasattr(ops, "sol_merge_partial_tail"):
                # Allocate the dense prefix partial after sparse exact has
                # finished its output allocation, avoiding overlap at the
                # sparse-stage peak.
                prefix_out, prefix_lse = _dense_rect_head_layout(
                    q, k, v, 0, prefix_tokens, tokens, softmax_scale
                )
                # The dense K-prefix partial can be merged independently for
                # each query range.  Chunking it bounds temporary output/LSE
                # storage while preserving the exact LSE fusion formula.
                tail_chunk_tokens, tail_chunk_reason = _select_sol_dense_tail_chunk(
                    q, tokens
                )
                exact_out, exact_lse = sparse_out, sparse_lse
                # The protected prefix is already complete in the dense
                # partial.  Copy it before starting tail work so its partial
                # buffers can be released during the long merge loop.
                exact_out[:, :, :prefix_tokens].copy_(prefix_out)
                exact_lse[:, :, :prefix_tokens].copy_(prefix_lse)
                del prefix_out, prefix_lse
                for tail_start in range(
                    prefix_tokens, tokens, tail_chunk_tokens
                ):
                    tail_tokens = min(
                        tail_chunk_tokens, tokens - tail_start
                    )
                    dense_tail_out, dense_tail_lse = _dense_rect_head_layout(
                        q, k, v, tail_start, tail_tokens,
                        prefix_tokens, softmax_scale,
                    )
                    ops.sol_merge_partial_tail(
                        dense_tail_out, dense_tail_lse, sparse_out, sparse_lse,
                        tail_start,
                    )
                    del dense_tail_out, dense_tail_lse
                if profile and report_key not in _sol_k_prefix_chunk_reported:
                    _sol_k_prefix_chunk_reported.add(report_key)
                    LOGGER.info(
                        "V100 diagnostics Sol K-prefix tail chunking active: "
                        "tokens=%d, selected_chunk_tokens=%d, reason=%s.",
                        tokens, tail_chunk_tokens, tail_chunk_reason,
                    )
            elif not use_csr_range:
                prefix_out, prefix_lse = _dense_rect_head_layout(
                    q, k, v, 0, prefix_tokens, tokens, softmax_scale
                )
                dense_tail_out, dense_tail_lse = _dense_rect_head_layout(
                    q, k, v, prefix_tokens, tokens - prefix_tokens,
                    prefix_tokens, softmax_scale,
                )
                sparse_tail_out = sparse_out[:, :, prefix_tokens:]
                sparse_tail_lse = sparse_lse[:, :, prefix_tokens:]
                total_tail_lse = torch.logaddexp(dense_tail_lse, sparse_tail_lse)
                dense_scale = torch.exp(
                    dense_tail_lse - total_tail_lse
                ).unsqueeze(-1)
                sparse_scale = torch.exp(
                    sparse_tail_lse - total_tail_lse
                ).unsqueeze(-1)
                exact_out = torch.empty_like(q)
                exact_lse = torch.empty(
                    (batch, heads, tokens), device=q.device, dtype=torch.float32
                )
                exact_out[:, :, prefix_tokens:].copy_(
                    (
                        dense_tail_out.float() * dense_scale
                        + sparse_tail_out.float() * sparse_scale
                    ).to(q.dtype)
                )
                exact_lse[:, :, prefix_tokens:].copy_(total_tail_lse)
                exact_out[:, :, :prefix_tokens].copy_(prefix_out)
                exact_lse[:, :, :prefix_tokens].copy_(prefix_lse)
                del prefix_out, prefix_lse
        elif csr_route:
            exact_out, exact_lse = ops.sol_sparse_exact_csr(
                q, k, v, counts, offsets, softmax_scale, 0,
            )
            if not use_csr_centroid_merge:
                del counts, offsets
        elif use_dense_prefix_split:
            report_key = (q.device.type, q.device.index, tokens, prefix_blocks)
            if profile and report_key not in _sol_split_reported:
                _sol_split_reported.add(report_key)
                LOGGER.info(
                    "V100 diagnostics Sol dense-prefix split active: "
                    "tokens=%d, prefix_blocks=%d, dense_prefix_tokens=%d, "
                    "sparse_tail_blocks=%d.",
                    tokens, prefix_blocks, prefix_blocks * block_size,
                    blocks - prefix_blocks,
                )
            exact_out, exact_lse = ops.sol_sparse_exact_split(
                q, k, v,
                counts, offsets, softmax_scale, prefix_blocks,
            )
            del counts, offsets
        else:
            exact_out, exact_lse = ops.sol_sparse_exact(
                q, k, v,
                counts, offsets, softmax_scale,
            )
            del counts, offsets
        if stage_events is not None:
            stage_events["sparse_exact"][1].record()
            stage_events["centroid_merge"][0].record()
        output = exact_out
        dense_prefix_tokens = prefix_blocks * block_size
        if use_csr_centroid_merge:
            output, exact_lse = ops.sol_centroid_merge_csr(
                q, kc, vc, counts, offsets, output, exact_lse, softmax_scale,
                dense_prefix_tokens, prefix_blocks, 16,
            )
            del counts, offsets
            if stage_events is not None:
                stage_events["centroid_merge"][1].record()
            flattened = output.transpose(1, 2).reshape(
                batch, tokens, heads * width
            )
            return flattened, kc, threshold, stage_events, route_stats
        if hasattr(ops, "sol_centroid_merge"):
            output, exact_lse = ops.sol_centroid_merge(
                q, kc, vc, selected, output, exact_lse, softmax_scale,
                dense_prefix_tokens, 16,
            )
            if stage_events is not None:
                stage_events["centroid_merge"][1].record()
            flattened = output.transpose(1, 2).reshape(
                batch, tokens, heads * width
            )
            return flattened, kc, threshold, stage_events, route_stats
        log_lengths = lengths.log().view(1, 1, 1, blocks)
        kc_t = kc.transpose(-2, -1)
        combine_chunk = 1024
        # The protected H3 conditioning prefix is routed to every K/V block.
        # For those complete query chunks, sol_sparse_exact already produced
        # the final normalized dense-attention result, so centroid correction
        # is mathematically a no-op. Keep the exact output and avoid the extra
        # Q-to-centroid GEMM, logsumexp, and blend work.
        for start in range(0, tokens, combine_chunk):
            stop = min(tokens, start + combine_chunk)
            if stop <= dense_prefix_tokens:
                continue
            query = q[:, :, start:stop].float()
            approximate_logits = torch.matmul(query, kc_t).mul_(softmax_scale)
            approximate_logits.add_(log_lengths)
            query_blocks = torch.arange(
                start, stop, device=q.device, dtype=torch.long
            ).div_(block_size, rounding_mode="floor")
            selected_rows = selected.index_select(2, query_blocks)
            approximate_logits.masked_fill_(
                selected_rows, -torch.inf
            )
            approximate_lse = torch.logsumexp(approximate_logits, dim=-1)
            exact_lse_block = exact_lse[:, :, start:stop]
            total_lse = torch.logaddexp(exact_lse_block, approximate_lse)
            exact_scale = torch.exp(exact_lse_block - total_lse).unsqueeze(-1)
            approximate_weight = torch.exp(
                approximate_logits - total_lse.unsqueeze(-1)
            )
            approximate_out = torch.matmul(approximate_weight, vc)
            output[:, :, start:stop].copy_(
                (exact_out[:, :, start:stop].float() * exact_scale + approximate_out)
                .to(output.dtype)
            )
        if stage_events is not None:
            stage_events["centroid_merge"][1].record()
        flattened = output.transpose(1, 2).reshape(batch, tokens, heads * width)
        return flattened, kc, threshold, stage_events, route_stats
    if not hasattr(ops, "sol_fwd_validation"):
        return None
    output = ops.sol_fwd_validation(
        q, k, v, kc, vc, threshold, int(prefix_stop), softmax_scale
    )
    batch, heads, tokens, width = output.shape
    flattened = output.transpose(1, 2).reshape(batch, tokens, heads * width)
    return flattened, kc, threshold, stage_events, route_stats


def _cu_seqlens(batch, length, device):
    key = (device.type, device.index, batch, length)
    value = _cu_seqlens_cache.get(key)
    if value is None:
        value = torch.arange(
            0, (batch + 1) * length, length, device=device, dtype=torch.int32
        )
        _cu_seqlens_cache[key] = value
    return value


def supports(q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs):
    return not support_reasons(
        q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs
    )


def support_reasons(q, k, v, heads, mask, skip_reshape, skip_output_reshape, kwargs):
    tensors = (q, k, v)
    is_head_layout = (
        skip_reshape
        and all(t.ndim == 4 for t in tensors)
        and q.shape[1] == k.shape[1] == v.shape[1] == heads
        and q.shape[-1] == k.shape[-1] == v.shape[-1] == 128
        and k.shape[2] == v.shape[2]
    )
    is_standard_layout = (
        not skip_reshape
        and all(t.ndim == 3 for t in tensors)
        and all(t.shape[-1] == heads * 128 for t in tensors)
        and k.shape[1] == v.shape[1]
    )
    reasons = []
    if mask is not None:
        reasons.append("mask-present")
    if skip_output_reshape:
        reasons.append("skip-output-reshape")
    if kwargs.get("enable_gqa", False):
        reasons.append("gqa-enabled")
    if not all(t.is_cuda for t in tensors):
        reasons.append("not-cuda")
    if not all(t.dtype == torch.float16 for t in tensors):
        reasons.append("not-all-fp16")
    if any(t.requires_grad for t in tensors):
        reasons.append("requires-grad")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        reasons.append("batch-mismatch")
    if not (is_head_layout or is_standard_layout):
        reasons.append("unsupported-layout-or-head-dim")
    if q.is_cuda and torch.cuda.get_device_capability(q.device) != (7, 0):
        reasons.append("not-sm70")
    return tuple(reasons)


def comfy_attention(q, k, v, heads, scale=None):
    load_extension()
    if q.ndim == 3:
        batch, q_len, _ = q.shape
        k_len = k.shape[1]
        head_dim = 128
        q = q.view(batch, q_len, heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(batch, k_len, heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(batch, k_len, heads, head_dim).permute(0, 2, 1, 3)
    batch, _, q_len, head_dim = q.shape
    k_len = k.shape[2]
    q_flat = q.permute(0, 2, 1, 3).contiguous().view(batch * q_len, heads, head_dim)
    k_flat = k.permute(0, 2, 1, 3).contiguous().view(batch * k_len, heads, head_dim)
    v_flat = v.permute(0, 2, 1, 3).contiguous().view(batch * k_len, heads, head_dim)
    softmax_scale = float(scale if scale is not None else 1.0 / math.sqrt(head_dim))

    out = _varlen_forward(
        q_flat,
        k_flat,
        v_flat,
        None,
        _cu_seqlens(batch, q_len, q.device),
        _cu_seqlens(batch, k_len, q.device),
        q_len,
        k_len,
        softmax_scale,
    )
    return out.view(batch, q_len, heads * head_dim)
