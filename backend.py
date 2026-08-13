import math
from pathlib import Path

import torch


_loaded = False
_cu_seqlens_cache = {}


def load_extension():
    global _loaded
    if _loaded:
        return
    binaries = list(Path(__file__).resolve().parent.glob("comfy_v100_flash_attn_cuda*.pyd"))
    if len(binaries) != 1:
        raise RuntimeError(
            "Expected exactly one comfy_v100_flash_attn_cuda CPython extension, "
            f"found {len(binaries)}."
        )
    torch.ops.load_library(str(binaries[0]))
    _loaded = True


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

    out, _lse, *_ = torch.ops.comfy_v100_flash_attn_cuda.varlen_fwd(
        q_flat,
        k_flat,
        v_flat,
        None,
        _cu_seqlens(batch, q_len, q.device),
        _cu_seqlens(batch, k_len, q.device),
        None,
        None,
        None,
        None,
        q_len,
        k_len,
        0.0,
        softmax_scale,
        False,
        False,
        -1,
        -1,
        0.0,
        False,
        1,
        None,
    )
    return out.view(batch, q_len, heads * head_dim)

