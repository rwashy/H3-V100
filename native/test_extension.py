import argparse
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_v100_flash_attn import flash_attn_bhld


def timed(fn, warmup=3, iterations=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dims", nargs="+", type=int, default=[128, 256])
    args = parser.parse_args()

    assert torch.cuda.get_device_capability() == (7, 0)
    torch.manual_seed(1234)
    for head_dim in args.head_dims:
        for length in args.lengths:
            shape = (1, args.heads, length, head_dim)
            q = torch.randn(shape, device="cuda", dtype=torch.float16)
            k = torch.randn(shape, device="cuda", dtype=torch.float16)
            v = torch.randn(shape, device="cuda", dtype=torch.float16)
            reference = F.scaled_dot_product_attention(q, k, v)
            actual = flash_attn_bhld(q, k, v)
            difference = (reference - actual).abs().float()
            fa_ms = timed(lambda: flash_attn_bhld(q, k, v))
            sdpa_ms = timed(lambda: F.scaled_dot_product_attention(q, k, v))
            print(
                f"head_dim={head_dim} length={length} "
                f"max_diff={difference.max().item():.6f} "
                f"mean_diff={difference.mean().item():.6f} "
                f"v100_fa_ms={fa_ms:.3f} sdpa_ms={sdpa_ms:.3f} "
                f"speedup={sdpa_ms/fa_ms:.2f}x"
            )
            assert torch.isfinite(actual).all()
            assert difference.max().item() <= 8e-2
            assert difference.mean().item() <= 8e-3


if __name__ == "__main__":
    main()
