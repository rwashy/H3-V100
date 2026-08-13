import importlib
import math

import torch


_extension = importlib.import_module("comfy_v100_flash_attn_cuda")
_cu_seqlens_cache = {}


def _cu_seqlens(batch, length, device):
    key = (device.type, device.index, batch, length)
    value = _cu_seqlens_cache.get(key)
    if value is None:
        value = torch.arange(
            0, (batch + 1) * length, length, device=device, dtype=torch.int32
        )
        _cu_seqlens_cache[key] = value
    return value


def is_supported(q, k, v, *, causal=False, mask=None):
    tensors = (q, k, v)
    return (
        all(t.is_cuda and t.dtype == torch.float16 for t in tensors)
        and all(t.ndim == 4 for t in tensors)
        and q.shape[-1] == k.shape[-1] == v.shape[-1]
        and q.shape[-1] in (128, 256)
        and q.shape[0] == k.shape[0] == v.shape[0]
        and q.shape[1] == k.shape[1] == v.shape[1]
        and k.shape[2] == v.shape[2]
        and not causal
        and mask is None
        and torch.cuda.get_device_capability(q.device) == (7, 0)
    )


def flash_attn_blhd(q, k, v, *, scale=None):
    """Return [batch, q_length, heads, head_dim] from BHLD FP16 inputs."""
    if not is_supported(q, k, v):
        raise ValueError("unsupported input for the SM70 head_dim=128/256 fast path")

    batch, heads, q_len, head_dim = q.shape
    k_len = k.shape[2]
    q_flat = q.permute(0, 2, 1, 3).contiguous().view(batch * q_len, heads, head_dim)
    k_flat = k.permute(0, 2, 1, 3).contiguous().view(batch * k_len, heads, head_dim)
    v_flat = v.permute(0, 2, 1, 3).contiguous().view(batch * k_len, heads, head_dim)
    cu_q = _cu_seqlens(batch, q_len, q.device)
    cu_k = _cu_seqlens(batch, k_len, q.device)
    softmax_scale = float(scale if scale is not None else 1.0 / math.sqrt(head_dim))

    out, _lse, *_ = torch.ops.comfy_v100_flash_attn_cuda.varlen_fwd(
        q_flat, k_flat, v_flat, None, cu_q, cu_k,
        None, None, None, None,
        q_len, k_len, 0.0, softmax_scale, False,
        False, -1, -1, 0.0, False, 1, None,
    )
    return out.view(batch, q_len, heads, head_dim)


def flash_attn_bhld(q, k, v, *, scale=None):
    """Return a contiguous BHLD tensor for callers outside ComfyUI."""
    return flash_attn_blhd(q, k, v, scale=scale).permute(0, 2, 1, 3).contiguous()


def comfy_attention(q, k, v, heads, *, scale=None):
    """Return ComfyUI's default [batch, length, heads * head_dim] layout."""
    out = flash_attn_blhd(q, k, v, scale=scale)
    return out.reshape(out.shape[0], out.shape[1], heads * out.shape[-1])
