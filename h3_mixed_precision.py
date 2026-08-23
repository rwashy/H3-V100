"""Workflow-scoped MiniMax H3 mixed-precision patch for NVIDIA V100.

SPDX-License-Identifier: GPL-3.0-only

The attention precision split is based on the community-tested Plan 2 profile
from Icbears/minimax-h3-v100-patch.  Unlike the original file patcher, this
module applies the replacement to a cloned ComfyUI MODEL via object patches.
"""

import logging
import math
import types

import torch

import comfy.model_management as model_management
import comfy.quant_ops
from comfy.ldm.modules.attention import attention_pytorch, optimized_attention

from .diagnostics import (
    DIAGNOSTICS_INTERVAL_KEY,
    DIAGNOSTICS_KEY,
    FLASH_BENCHMARK_OCCURRED_KEY,
    run_h3_block_diagnostic,
    enqueue_deferred_block_sample,
)
from .sol_attention import BLOCK_COUNT_KEY, BLOCK_INDEX_KEY


LOGGER = logging.getLogger("H3V100MixedPrecision")
PATCH_MARKER = "_h3_v100_mixed_precision_plan2"
OPTION_KEY = "h3_v100_mixed_precision_v100_only"
AUDIO_RANGES_OPTION_KEY = "minimax_h3_fp32_audio_ranges"
QKV_CHUNKING_OPTION_KEY = "v100_h3_qkv_chunking"
QKV_CHUNK_TOKENS_OPTION_KEY = "v100_h3_qkv_chunk_tokens"
QKV_CHUNK_THRESHOLD_OPTION_KEY = "v100_h3_qkv_chunk_threshold"
QKV_CACHE_TRIM_THRESHOLD_OPTION_KEY = "v100_h3_qkv_cache_trim_threshold_mb"
EXPERIMENTAL_FP16_OPTION_KEY = "v100_h3_experimental_fp16_linear"
BLOCK_PATCH_MARKER = "_h3_v100_audio_ranges"
BLOCK_ORIGINAL_FORWARD_ATTR = "_h3_v100_audio_ranges_original_forward"
_diagnostic_shapes = set()
_diagnostic_stats = {}
_qkv_diagnostic_shapes = set()
_missing_audio_ranges_reported = set()

# Extreme compatibility is deliberately outside the normal operating policy.
# The 16 GiB reference boundary is scaled by physical VRAM; live pressure may
# enter the guard earlier, but never changes the path for ordinary sequences.
_EXTREME_QKV_REFERENCE_TOKENS = 110_000
_EXTREME_QKV_REFERENCE_VRAM = 16 * 1024 ** 3


def _extreme_qkv_policy(tokens, device):
    if device.type != "cuda":
        return False, 1024, 0
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    scaled_threshold = max(
        48_000,
        int(_EXTREME_QKV_REFERENCE_TOKENS * total_bytes / _EXTREME_QKV_REFERENCE_VRAM),
    )
    # If less than 6.25% of physical VRAM remains, protect somewhat earlier.
    pressure_threshold = max(48_000, int(scaled_threshold * 0.85))
    extreme = tokens >= scaled_threshold or (
        free_bytes < total_bytes // 16 and tokens >= pressure_threshold
    )
    return extreme, (512 if extreme else 1024), scaled_threshold


def _unwrap_our_block_forward(value):
    current = value
    seen = set()
    while current is not None:
        function = getattr(current, "__func__", current)
        if not getattr(function, BLOCK_PATCH_MARKER, False):
            return current
        identity = id(function)
        if identity in seen:
            raise RuntimeError("H3 block diagnostics detected a cyclic wrapper chain.")
        seen.add(identity)
        current = getattr(function, BLOCK_ORIGINAL_FORWARD_ATTR, None)
    raise RuntimeError("H3 block diagnostics could not recover its original forward.")


def _slice_rope(rope_freqs, start, stop):
    return rope_freqs[:, start:stop]


def _normalize_rope_pair(self, q, k, rope_freqs, device):
    """Run H3 Q/K RMSNorm+RoPE in FP32 for one token window."""
    qw = model_management.cast_to(self.q_norm.weight, dtype=q.dtype, device=device)
    kw = model_management.cast_to(self.k_norm.weight, dtype=k.dtype, device=device)
    rope = rope_freqs.to(q.dtype) if rope_freqs.dtype != q.dtype else rope_freqs
    rot = rope.shape[-3] * 2
    return comfy.quant_ops.ck.rms_rope_split_half(
        q, k, rope, qw, kw, epsilon=self.q_norm.eps, rot_dim=rot
    )


def _chunked_qk_rope(self, q, k, rope_freqs, transformer_options):
    """Normalize/RoPE Q/K in bounded FP32 windows and overwrite FP16 inputs."""
    tokens = int(q.shape[0])
    chunk_tokens = max(1, int(transformer_options.get(QKV_CHUNK_TOKENS_OPTION_KEY, 1024)))
    q_out = q.view(tokens, self.heads, self.head_dim)
    k_out = k.view(tokens, self.heads, self.head_dim)
    for start in range(0, tokens, chunk_tokens):
        stop = min(tokens, start + chunk_tokens)
        qc = q_out[start:stop].unsqueeze(0).float()
        kc = k_out[start:stop].unsqueeze(0).float()
        qc, kc = _normalize_rope_pair(
            self, qc, kc, _slice_rope(rope_freqs, start, stop), q.device
        )
        q_out[start:stop].copy_(qc[0])
        k_out[start:stop].copy_(kc[0])
    key = ("qk_rope", q.device.type, q.device.index, tokens, chunk_tokens)
    if key not in _qkv_diagnostic_shapes:
        _qkv_diagnostic_shapes.add(key)
        LOGGER.info(
            "V100 adaptive Q/K RMSNorm+RoPE: tokens=%d, chunk_tokens=%d, "
            "chunks=%d, fp32_workspace=True, retained_dtype=torch.float16, "
            "in_place=True, avoided_full_buffers=2.",
            tokens, chunk_tokens, math.ceil(tokens / chunk_tokens),
        )
    return q_out, k_out


def _streaming_audio_attention(q, k, v, audio_ranges, heads, head_dim, key_chunk=1024, query_chunk=64):
    """Exact online-softmax attention with bounded FP32 workspaces for audio rows."""
    outputs = []
    scale = head_dim ** -0.5
    for range_start, range_stop in audio_ranges:
        for qs in range(range_start, range_stop, query_chunk):
            qe = min(range_stop, qs + query_chunk)
            qf = q[:, :, qs:qe].float()
            shape = qf.shape[:-1] + (1,)
            running_max = torch.full(shape, -torch.inf, dtype=torch.float32, device=q.device)
            running_sum = torch.zeros(shape, dtype=torch.float32, device=q.device)
            running_out = torch.zeros(qf.shape, dtype=torch.float32, device=q.device)
            for ks in range(0, k.shape[2], key_chunk):
                ke = min(k.shape[2], ks + key_chunk)
                scores = torch.matmul(qf, k[:, :, ks:ke].float().transpose(-2, -1)).mul_(scale)
                block_max = scores.amax(dim=-1, keepdim=True)
                new_max = torch.maximum(running_max, block_max)
                correction = torch.exp(running_max - new_max)
                probs = torch.exp(scores - new_max)
                running_sum.mul_(correction).add_(probs.sum(dim=-1, keepdim=True))
                running_out.mul_(correction).add_(torch.matmul(probs, v[:, :, ks:ke].float()))
                running_max = new_max
            outputs.append((qs, qe, (running_out / running_sum).transpose(1, 2).reshape(1, qe - qs, heads * head_dim)))
    return outputs


def _is_v100_device(device):
    """Return True only for Volta compute capability 7.0 CUDA devices."""
    try:
        return torch.cuda.get_device_capability(device) == (7, 0)
    except Exception:
        return False


def _trim_qkv_cache_if_needed(device, transformer_options):
    if device.type != "cuda":
        return None
    threshold = float(
        transformer_options.get(QKV_CACHE_TRIM_THRESHOLD_OPTION_KEY, 2048)
    )
    free_before, _ = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    cached = max(0, reserved_before - allocated)
    if free_before >= threshold * 1024 ** 2 or cached < 512 * 1024 ** 2:
        return None
    torch.cuda.empty_cache()
    free_after, _ = torch.cuda.mem_get_info(device)
    reserved_after = torch.cuda.memory_reserved(device)
    return (
        free_before / 1024 ** 2,
        free_after / 1024 ** 2,
        reserved_before / 1024 ** 2,
        reserved_after / 1024 ** 2,
    )


def _trim_and_log(device, transformer_options, reason):
    trimmed = _trim_qkv_cache_if_needed(device, transformer_options)
    if trimmed is not None and transformer_options.get(DIAGNOSTICS_KEY, False):
        LOGGER.info(
            "V100 diagnostics adaptive cache trim: "
            "cuda_free_before/after=%.1f/%.1f MiB, "
            "reserved_before/after=%.1f/%.1f MiB; reason=%s.",
            trimmed[0], trimmed[1], trimmed[2], trimmed[3], reason,
        )
    return trimmed


def _qkv_projection(self, proj_x, transformer_options):
    """Project QKV, using separate contiguous outputs for long inference."""
    tokens = int(proj_x.shape[0])
    enabled = bool(transformer_options.get(QKV_CHUNKING_OPTION_KEY, False))
    threshold = max(
        0, int(transformer_options.get(QKV_CHUNK_THRESHOLD_OPTION_KEY, 38_000))
    )
    chunk_tokens = max(
        1, int(transformer_options.get(QKV_CHUNK_TOKENS_OPTION_KEY, 1024))
    )
    extreme, extreme_chunk_tokens, extreme_threshold = _extreme_qkv_policy(
        tokens, proj_x.device
    )
    if extreme:
        chunk_tokens = min(chunk_tokens, extreme_chunk_tokens)
    width = self.heads * self.head_dim
    if not enabled or tokens <= threshold:
        return self.qkv_proj(proj_x).split(width, dim=-1)

    trimmed = _trim_qkv_cache_if_needed(proj_x.device, transformer_options)
    if extreme and proj_x.device.type == "cuda":
        # This guard is intentionally unconditional in the extreme tier. The
        # quantized ConvRot projection needs a transient dequantized weight,
        # and allocator cache that looks reusable is not usable by that path.
        before_free, _ = torch.cuda.mem_get_info(proj_x.device)
        before_reserved = torch.cuda.memory_reserved(proj_x.device)
        torch.cuda.empty_cache()
        after_free, _ = torch.cuda.mem_get_info(proj_x.device)
        after_reserved = torch.cuda.memory_reserved(proj_x.device)
        trimmed = (
            before_free / 1024 ** 2, after_free / 1024 ** 2,
            before_reserved / 1024 ** 2, after_reserved / 1024 ** 2,
        )
    if torch.is_grad_enabled() and proj_x.requires_grad:
        result = torch.cat(
            [self.qkv_proj(part) for part in proj_x.split(chunk_tokens, dim=0)],
            dim=0,
        )
        q_out, k_out, v_out = result.split(width, dim=-1)
        layout = "combined-autograd"
    else:
        q_out = k_out = v_out = None
        for start in range(0, tokens, chunk_tokens):
            input_part = proj_x[start:start + chunk_tokens]
            # Avoid retaining a full-sequence FP16 copy in the extreme tier.
            # Normal sequences keep the existing whole-tensor conversion.
            if extreme and input_part.dtype == torch.float32:
                input_part = input_part.half()
            part = self.qkv_proj(input_part)
            q_part, k_part, v_part = part.split(width, dim=-1)
            if q_out is None:
                output_shape = (tokens, width)
                q_out = torch.empty(
                    output_shape,
                    dtype=part.dtype,
                    device=part.device,
                )
                k_out = torch.empty_like(q_out)
                v_out = torch.empty_like(q_out)
            stop = start + part.shape[0]
            q_out[start:stop].copy_(q_part)
            k_out[start:stop].copy_(k_part)
            v_out[start:stop].copy_(v_part)
        layout = "split-contiguous"

    key = (proj_x.device.type, proj_x.device.index, tokens, chunk_tokens)
    if key not in _qkv_diagnostic_shapes:
        _qkv_diagnostic_shapes.add(key)
        LOGGER.info(
            "V100 adaptive QKV projection: tokens=%d, chunk_tokens=%d, chunks=%d, "
            "input_dtype=%s, output_dtype=%s, device=%s, pre_trim=%s, "
            "layout=%s, flash_repack_copies=0.",
            tokens, chunk_tokens, math.ceil(tokens / chunk_tokens), proj_x.dtype,
            q_out.dtype, proj_x.device, bool(trimmed), layout,
        )
        if extreme:
            LOGGER.info(
                "V100 extreme-sequence QKV compatibility: tokens=%d, "
                "scaled_threshold=%d, total_vram_scaled=True, "
                "chunk_local_fp16=%s.",
                tokens, extreme_threshold, proj_x.dtype == torch.float32,
            )
        if (
            trimmed is not None
            and transformer_options.get(DIAGNOSTICS_KEY, False)
        ):
            LOGGER.info(
                "V100 diagnostics adaptive QKV cache trim: "
                "cuda_free_before/after=%.1f/%.1f MiB, "
                "reserved_before/after=%.1f/%.1f MiB.",
                trimmed[0], trimmed[1], trimmed[2], trimmed[3],
            )
    return q_out, k_out, v_out


def _out_projection(self, out, transformer_options):
    """Bound the quantized output-projection accumulation above 38K tokens."""
    tokens = int(out.shape[0])
    enabled = bool(transformer_options.get(QKV_CHUNKING_OPTION_KEY, False))
    threshold = max(
        0, int(transformer_options.get(QKV_CHUNK_THRESHOLD_OPTION_KEY, 38_000))
    )
    chunk_tokens = max(
        1, int(transformer_options.get(QKV_CHUNK_TOKENS_OPTION_KEY, 1024))
    )
    extreme, extreme_chunk_tokens, _ = _extreme_qkv_policy(tokens, out.device)
    if extreme:
        chunk_tokens = min(chunk_tokens, extreme_chunk_tokens)
    # Keep attention output projection on its validated FP32 boundary. The
    # former scaled FP16 experiment produced black/noisy output and must not be
    # coupled to the independently validated FP16 MLP path.
    def project(part):
        return self.out_proj(part)

    if not enabled or tokens <= threshold:
        return project(out)

    trimmed = _trim_qkv_cache_if_needed(out.device, transformer_options)
    if torch.is_grad_enabled() and out.requires_grad:
        result = torch.cat(
            [project(part) for part in out.split(chunk_tokens, dim=0)],
            dim=0,
        )
    else:
        result = None
        for start in range(0, tokens, chunk_tokens):
            part = project(out[start:start + chunk_tokens])
            if result is None:
                result = torch.empty(
                    (tokens,) + tuple(part.shape[1:]),
                    dtype=part.dtype,
                    device=part.device,
                )
            result[start:start + part.shape[0]].copy_(part)

    key = ("out_proj", out.device.type, out.device.index, tokens, chunk_tokens)
    if key not in _qkv_diagnostic_shapes:
        _qkv_diagnostic_shapes.add(key)
        LOGGER.info(
            "V100 adaptive attention output projection: tokens=%d, "
            "chunk_tokens=%d, chunks=%d, input_dtype=%s, output_dtype=%s, "
            "device=%s, pre_trim=%s.",
            tokens, chunk_tokens, math.ceil(tokens / chunk_tokens), out.dtype,
            result.dtype, out.device, bool(trimmed),
        )
    return result


def h3_v100_attention_forward(self, x, rope_freqs=None, transformer_options={}):
    """Plan 2: FP16 QKV/attention with FP32 norm, RoPE and residual stream."""
    s = x.shape[0]
    residual_dtype = x.dtype
    v100_only = True
    if isinstance(transformer_options, dict):
        v100_only = transformer_options.get(OPTION_KEY, True)

    use_fp16 = (
        x.device.type == "cuda"
        and x.dtype == torch.float32
        and (not v100_only or _is_v100_device(x.device))
    )
    audio_ranges = ()
    if isinstance(transformer_options, dict):
        audio_ranges = transformer_options.get(AUDIO_RANGES_OPTION_KEY, ())
    use_fp32_audio_attention = (
        use_fp16 and bool(audio_ranges) and not model_management.in_training
    )
    missing_audio_metadata = (
        use_fp16 and not audio_ranges and not model_management.in_training
    )
    diagnostics = bool(
        isinstance(transformer_options, dict)
        and transformer_options.get(DIAGNOSTICS_KEY, False)
    )
    profile_stages = False
    stage_events = None
    if diagnostics:
        diagnostics_interval = max(
            0, int(transformer_options.get(DIAGNOSTICS_INTERVAL_KEY, 50))
        )
        stats_key = (x.device.index, s, tuple(audio_ranges), bool(use_fp16))
        calls = _diagnostic_stats.get(stats_key, 0) + 1
        _diagnostic_stats[stats_key] = calls
        profile_stages = bool(
            x.is_cuda
            and diagnostics_interval > 0
            and calls % diagnostics_interval == 0
        )
        if profile_stages:
            stage_events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            stage_events[0].record()
        if diagnostics_interval > 0 and calls % diagnostics_interval == 0:
            allocated = torch.cuda.memory_allocated(x.device) / (1024 ** 2) if x.is_cuda else 0
            reserved = torch.cuda.memory_reserved(x.device) / (1024 ** 2) if x.is_cuda else 0
            peak = torch.cuda.max_memory_allocated(x.device) / (1024 ** 2) if x.is_cuda else 0
            LOGGER.info(
                "V100 diagnostics H3 summary: calls=%d, sequence=%d, "
                "audio_ranges=%s, mixed_precision=%s, memory_allocated=%.1f MiB, "
                "memory_reserved=%.1f MiB, peak=%.1f MiB.",
                calls, s, tuple(audio_ranges), bool(use_fp16), allocated, reserved, peak,
            )

    # Feeding FP16 into qkv_proj is the important memory/performance island.
    # The quantized ComfyUI operator chooses its activation/output path from
    # the input dtype, while the FP32 residual tensor remains untouched.
    extreme_qkv, _, _ = _extreme_qkv_policy(s, x.device)
    # Normal workloads retain the established whole-tensor FP16 path. Only
    # the VRAM-scaled extreme tier converts each QKV input slice locally.
    proj_x = x.half() if use_fp16 and not extreme_qkv else x
    q, k, v = _qkv_projection(self, proj_x, transformer_options)
    if profile_stages:
        stage_events[1].record()
    if proj_x is not x:
        del proj_x

    adaptive_qk = bool(
        use_fp16
        and transformer_options.get(QKV_CHUNKING_OPTION_KEY, False)
        and s > int(transformer_options.get(QKV_CHUNK_THRESHOLD_OPTION_KEY, 38_000))
        and rope_freqs is not None
    )

    # Q/K norm and RoPE remain FP32 for numerical stability.  V stays FP16
    # until optimized_attention when the mixed-precision path is active.
    if use_fp16 and not adaptive_qk:
        q = q.float()
        k = k.float()

    v = v.view(s, self.heads, self.head_dim)
    if adaptive_qk:
        q, k = _chunked_qk_rope(self, q, k, rope_freqs, transformer_options)
    elif rope_freqs is not None:
        q = q.view(1, s, self.heads, self.head_dim)
        k = k.view(1, s, self.heads, self.head_dim)
        qw = model_management.cast_to(
            self.q_norm.weight, dtype=q.dtype, device=x.device
        )
        kw = model_management.cast_to(
            self.k_norm.weight, dtype=k.dtype, device=x.device
        )
        rope = rope_freqs.to(q.dtype) if rope_freqs.dtype != q.dtype else rope_freqs
        rot = rope.shape[-3] * 2
        # Preserve the validated release integration here.  The later
        # inference-only in-place variant changed the tensors presented to the
        # native attention backend and regressed real H3 Flash/Sol output,
        # despite the CUDA kernel itself remaining unchanged.  Keep separate
        # normalized/RoPE outputs for ordinary sequences; the bounded-memory
        # adaptive path above remains responsible for very long sequences.
        q, k = comfy.quant_ops.ck.rms_rope_split_half(
            q,
            k,
            rope,
            qw,
            kw,
            epsilon=self.q_norm.eps,
            rot_dim=rot,
        )
        q = q[0]
        k = k[0]
    else:
        q = self.q_norm(q.view(s, self.heads, self.head_dim))
        k = self.k_norm(k.view(s, self.heads, self.head_dim))

    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)
    if profile_stages:
        stage_events[2].record()

    if adaptive_qk:
        _trim_and_log(
            x.device, transformer_options, "contiguous-attention-output"
        )

    if use_fp32_audio_attention:
        diagnostic_key = (x.device.index, s, tuple(audio_ranges))
        measure = diagnostics and diagnostic_key not in _diagnostic_shapes
        if measure:
            fp16_start = torch.cuda.Event(enable_timing=True)
            fp16_end = torch.cuda.Event(enable_timing=True)
            audio_start = torch.cuda.Event(enable_timing=True)
            audio_end = torch.cuda.Event(enable_timing=True)
            fp16_start.record()
        attention_out = optimized_attention(
            q.half(),
            k.half(),
            v.half(),
            self.heads,
            mask=None,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
        if measure:
            fp16_end.record()
            audio_start.record()
        # ComfyUI 0.31.1's audio carry/sampler path is sensitive to FP16
        # attention for target/reference audio queries. Keep the fast FP16
        # result for text/video, then recompute only those query rows in FP32.
        if adaptive_qk:
            audio_outputs = _streaming_audio_attention(
                q, k, v, audio_ranges, self.heads, self.head_dim,
                key_chunk=max(1, int(transformer_options.get(QKV_CHUNK_TOKENS_OPTION_KEY, 1024))),
            )
            del q, k, v
            _trim_and_log(
                x.device, transformer_options, "fp32-attention-promotion"
            )
            out = attention_out.to(residual_dtype)
            del attention_out
            for start, stop, audio_out in audio_outputs:
                out[:, start:stop] = audio_out.to(residual_dtype)
            del audio_outputs
        else:
            out = attention_out.to(residual_dtype)
            del attention_out
            audio_v = v.to(residual_dtype)
            for start, stop in audio_ranges:
                if 0 <= start < stop <= s:
                    out[:, start:stop] = attention_pytorch(
                        q[:, :, start:stop],
                        k,
                        audio_v,
                        self.heads,
                        mask=None,
                        skip_reshape=True,
                    )
        if measure:
            audio_end.record()
            if not transformer_options.get(FLASH_BENCHMARK_OCCURRED_KEY, False):
                audio_end.synchronize()
                _diagnostic_shapes.add(diagnostic_key)
                LOGGER.info(
                    "V100 diagnostics H3 attention: sequence=%d, heads=%d, head_dim=%d, "
                    "audio_ranges=%s, audio_rows=%d, FP16 full attention=%.3f ms, "
                    "FP32 audio recompute=%.3f ms.",
                    s,
                    self.heads,
                    self.head_dim,
                    tuple(audio_ranges),
                    sum(stop - start for start, stop in audio_ranges),
                    fp16_start.elapsed_time(fp16_end),
                    audio_start.elapsed_time(audio_end),
                )
    elif missing_audio_metadata:
        warning_key = (x.device.index, s)
        if warning_key not in _missing_audio_ranges_reported:
            _missing_audio_ranges_reported.add(warning_key)
            LOGGER.warning(
                "H3 audio row metadata is missing or invalid for sequence=%d; "
                "falling back to full FP32 attention for audio safety.",
                s,
            )
        out = optimized_attention(
            q.float(),
            k.float(),
            v.float(),
            self.heads,
            mask=None,
            skip_reshape=True,
            transformer_options=transformer_options,
        ).to(residual_dtype)
    elif use_fp16:
        attention_out = optimized_attention(
            q.half(),
            k.half(),
            v.half(),
            self.heads,
            mask=None,
            skip_reshape=True,
            transformer_options=transformer_options,
        )
        if adaptive_qk:
            del q, k, v
            _trim_and_log(
                x.device, transformer_options, "fp32-attention-promotion"
            )
        out = attention_out.to(residual_dtype)
        del attention_out
    else:
        out = optimized_attention(
            q,
            k,
            v,
            self.heads,
            mask=None,
            skip_reshape=True,
            transformer_options=transformer_options,
        )

    if profile_stages:
        stage_events[3].record()
    result = _out_projection(self, out.squeeze(0), transformer_options)
    if profile_stages:
        stage_events[4].record()
        stage_events[4].synchronize()
        LOGGER.info(
            "V100 diagnostics H3 attention stages: sequence=%d, heads=%d, "
            "QKV=%.3f ms, norm_rope_repack=%.3f ms, attention_and_audio=%.3f ms, "
            "output_projection=%.3f ms, total=%.3f ms.",
            s,
            self.heads,
            stage_events[0].elapsed_time(stage_events[1]),
            stage_events[1].elapsed_time(stage_events[2]),
            stage_events[2].elapsed_time(stage_events[3]),
            stage_events[3].elapsed_time(stage_events[4]),
            stage_events[0].elapsed_time(stage_events[4]),
        )
    return result


setattr(h3_v100_attention_forward, PATCH_MARKER, True)
# Lets attention-composition nodes recognize that this forward delegates to
# ComfyUI's optimized_attention rather than replacing the backend itself.
h3_v100_attention_forward._uses_optimized_attention = True


def _audio_ranges_from_mod_segments(mod_segments, sequence_length):
    """Return H3 audio spans from its (start, stop, timestep*3+modality) table."""
    ranges = []
    for segment in mod_segments or ():
        if not isinstance(segment, (tuple, list)) or len(segment) != 3:
            continue
        start, stop, row = segment
        if not all(isinstance(value, int) for value in (start, stop, row)):
            continue
        if row % 3 == 2 and 0 <= start < stop <= sequence_length:
            ranges.append((start, stop))
    return tuple(ranges)


def _make_h3_block_forward(original_forward, block_index, block_count):
    """Expose audio row ranges only while the corresponding block is running."""

    def h3_v100_block_forward(
        self, x, t_emb, mod_segments, rope_freqs, transformer_options={}
    ):
        if not isinstance(transformer_options, dict):
            return original_forward(
                x, t_emb, mod_segments, rope_freqs,
                transformer_options=transformer_options,
            )

        missing = object()
        previous = transformer_options.get(AUDIO_RANGES_OPTION_KEY, missing)
        previous_block = transformer_options.get(BLOCK_INDEX_KEY, missing)
        previous_block_count = transformer_options.get(BLOCK_COUNT_KEY, missing)
        transformer_options[AUDIO_RANGES_OPTION_KEY] = _audio_ranges_from_mod_segments(
            mod_segments, x.shape[0]
        )
        transformer_options[BLOCK_INDEX_KEY] = int(block_index)
        transformer_options[BLOCK_COUNT_KEY] = int(block_count)
        try:
            def call_original():
                result = original_forward(
                    x, t_emb, mod_segments, rope_freqs,
                    transformer_options=transformer_options,
                )
                if transformer_options.get(DIAGNOSTICS_KEY, False):
                    if block_index == 0:
                        transformer_options["v100_sol_nonfinite_reported"] = False
                    enqueue_deferred_block_sample(
                        transformer_options, block_index, result
                    )
                return result

            if transformer_options.get(DIAGNOSTICS_KEY, False):
                interval = max(
                    0, int(transformer_options.get(DIAGNOSTICS_INTERVAL_KEY, 50))
                )
                return run_h3_block_diagnostic(
                    block_index, x, interval, transformer_options, call_original
                )
            return call_original()
        finally:
            if previous is missing:
                transformer_options.pop(AUDIO_RANGES_OPTION_KEY, None)
            else:
                transformer_options[AUDIO_RANGES_OPTION_KEY] = previous
            if previous_block is missing:
                transformer_options.pop(BLOCK_INDEX_KEY, None)
            else:
                transformer_options[BLOCK_INDEX_KEY] = previous_block
            if previous_block_count is missing:
                transformer_options.pop(BLOCK_COUNT_KEY, None)
            else:
                transformer_options[BLOCK_COUNT_KEY] = previous_block_count

    setattr(h3_v100_block_forward, BLOCK_PATCH_MARKER, True)
    setattr(
        h3_v100_block_forward,
        BLOCK_ORIGINAL_FORWARD_ATTR,
        _unwrap_our_block_forward(original_forward),
    )
    return h3_v100_block_forward


def _is_our_patch(value):
    function = getattr(value, "__func__", value)
    return bool(getattr(function, PATCH_MARKER, False))


class H3V100MixedPrecision:
    """Apply the V100 precision split to one cloned MiniMax H3 MODEL."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "v100_only": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "V100"
    DESCRIPTION = (
        "Validated H3 V100 precision split: FP16 projection/attention islands; "
        "FP32 norm, RoPE, output projection, audio safety and residual stream. "
        "The H3 V100 node applies its bounded MLP policy separately."
    )

    def patch(self, model, enabled=True, v100_only=True):
        if not enabled:
            return (model,)

        patched_model = model.clone()
        try:
            diffusion_model = patched_model.get_model_object("diffusion_model")
        except Exception as exc:
            raise RuntimeError(
                "MiniMax H3 V100 patch could not access diffusion_model."
            ) from exc

        blocks = getattr(diffusion_model, "blocks", None)
        if not blocks:
            raise RuntimeError(
                "MiniMax H3 V100 patch expected diffusion_model.blocks, but none were found."
            )

        for index, block in enumerate(blocks):
            attention = getattr(block, "attn", None)
            if attention is None or not hasattr(attention, "qkv_proj"):
                raise RuntimeError(
                    "MiniMax H3 V100 patch rejected this model: "
                    f"block {index} has no compatible attention/QKV projection."
                )

        transformer_options = patched_model.model_options.setdefault(
            "transformer_options", {}
        )
        transformer_options[OPTION_KEY] = bool(v100_only)

        newly_patched = 0
        already_patched = 0
        refreshed_blocks = 0
        for index, block in enumerate(blocks):
            block_key = f"diffusion_model.blocks.{index}.forward"
            existing_block = patched_model.object_patches.get(block_key)
            if existing_block is not None:
                block_function = getattr(existing_block, "__func__", existing_block)
                if not getattr(block_function, BLOCK_PATCH_MARKER, False):
                    raise RuntimeError(
                        "MiniMax H3 V100 audio safety patch found another block patch at "
                        f"{block_key}. Remove that patch before this node."
                    )
                base_block_forward = _unwrap_our_block_forward(existing_block)
                refreshed_blocks += 1
            else:
                base_block_forward = _unwrap_our_block_forward(block.forward)
            patched_model.add_object_patch(
                block_key,
                types.MethodType(
                    _make_h3_block_forward(base_block_forward, index, len(blocks)), block
                ),
            )

            key = f"diffusion_model.blocks.{index}.attn.forward"
            existing = patched_model.object_patches.get(key)
            if existing is not None:
                if _is_our_patch(existing):
                    already_patched += 1
                    continue
                raise RuntimeError(
                    "MiniMax H3 V100 patch found another Attention patch at "
                    f"{key}. Remove the other attention/low-VRAM patch before this node."
                )

            patched_model.add_object_patch(
                key, types.MethodType(h3_v100_attention_forward, block.attn)
            )
            newly_patched += 1

        LOGGER.info(
            "MiniMax H3 V100 mixed precision active: %d blocks patched, "
            "%d block wrappers refreshed, %d attention patches reused, "
            "v100_only=%s. Audio query attention remains FP32; "
            "TE-Speed wrappers are preserved.",
            newly_patched,
            refreshed_blocks,
            already_patched,
            bool(v100_only),
        )
        return (patched_model,)
