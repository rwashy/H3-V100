# H3_V100

[简体中文](README_zh-CN.md) | English

`H3_V100` is a workflow-scoped ComfyUI custom node for MiniMax H3 inference
on NVIDIA Tesla V100 (SM70). It adds a validated mixed-precision path, explicit
exact Flash and sparse Sol attention routes, and always-on adaptive memory
guards for long video sequences.

The node does not edit ComfyUI or other custom nodes. It validates the expected
50-block MiniMax H3 model layout and applies patches only to the cloned `MODEL`
object passed through it. Unknown model layouts are rejected instead of being
patched speculatively.

## Features

- Optional H3 mixed precision: FP16 QKV, main attention and validated FP16 MLP
  linear boundaries; FP32 is retained for Q/K normalization, RoPE, SwiGLU,
  attention output projection, residual-sensitive work and audio-query
  recomputation.
- Explicit attention backend selection: exact SM70 Flash Attention or the
  validated Flash-Sol-Flash long-sequence schedule. Unsupported calls fall back
  safely, and Flash mode never enters Sol implicitly.
- Always-on adaptive memory: runtime-sized MLP chunks, bounded QKV and output
  projections, cache trimming under pressure, and conservative dynamic-weight
  prefetch behavior.
- VRAM-aware balanced chunk selection: the policy derives the minimum safe
  chunk count from physical VRAM, live driver memory, allocator cache and the
  measured V100 capacity guard, then aligns FP16 MLP rows for SM70 GEMMs.
- Audio-safe inference: audio query attention stays on the stable FP32 path.

## Node controls

The public node exposes one precision switch, one backend selector and the
quality/schedule controls needed by explicit Sol mode:

| Control | Default | Function |
|---|---:|---|
| `mixed_precision` | On | Enables the validated H3 mixed-precision and FP16-linear path. |
| `attention_backend` | `flash_attn` | Selects exact Flash or explicit `sol_attn`. |
| `sol_tau` | `1.0` | Controls the Sol sparse threshold. |
| `sol_start_percent` | `0.2` | Starts the Sol diffusion window. |
| `sol_end_percent` | `0.8` | Ends the Sol diffusion window. |

Adaptive memory protection is always enabled and is independent of both
controls. Sol controls are hidden while Flash is selected. Turning
`mixed_precision` off therefore retains bounded-memory
execution. To bypass every optimization, remove or bypass the node itself.

## Recommended resolution and duration tiers

Approximate 16:9 workload guidance for a 16 GiB V100 is based on measured H3
token boundaries, not a guarantee: up to roughly 32K tokens may avoid MLP
chunking; 32K-64K uses moderate chunking; 64K-96K uses conservative 8K chunks;
above 96K uses 4K chunks and can become extremely slow. Resolution, duration,
audio/reference inputs, other resident models, CUDA fragmentation and ComfyUI
version all affect the actual boundary. Start with 864x480 or 960x544 at 5-10
seconds, then increase one dimension at a time. Extremely large combinations
such as 1280x736 at 15 seconds remain experimental even with memory guards.

This matrix pairs every 0.2-2.0 MP preset with 5-15 seconds. Cell values are
estimated packed sequence lengths in thousands of tokens. Colors indicate the
approximate physical-VRAM region needed to avoid MLP chunking; they are not
total-memory requirements or a guarantee that a workflow will fit.

![MiniMax H3 resolution-duration operating regions](assets/h3-resolution-duration-vram-regions.png)

| Tier | Resolution/duration examples | Guidance |
|---|---|---|
| Recommended | 0.2-0.5 MP at 5-10 s; 0.2-0.3 MP up to 15 s | Best balance of detail, duration and generation time. |
| Extended | 0.4-0.6 MP at 10-15 s; 0.7-0.9 MP at 5-8 s | Chunking is common; expect substantially longer runs. |
| Extreme | Around 0.9 MP at 15 s, or above 1.0 MP for most durations | Heavy 4K/8K chunking; validate short runs first. |
| Not recommended | 1.5-2.0 MP at long durations | Theoretical coverage only; runtime and OOM risk are disproportionate. |

## Usage profiles

| Goal | Mixed precision | Attention backend | Guidance |
|---|---:|---|---|
| Exact optimized route | On | `flash_attn` | Uses exact attention with validated V100 mixed precision. |
| Explicit sparse route | On | `sol_attn` | Uses Sol only inside its selected block and diffusion window. |
| Precision comparison | Off | `flash_attn` | Keeps adaptive memory; unsupported FP32 native calls fall back safely. |

## Requirements

- NVIDIA Tesla V100 / compute capability 7.0 (SM70).
- Windows x64 for the bundled native extension.
- CPython 3.12.
- A CUDA-enabled PyTorch build compatible with the host ComfyUI installation.
  Development validation used PyTorch 2.8.0 with CUDA 12.8.
- A ComfyUI build containing the MiniMax H3 model implementation expected by
  this node.
- A sufficiently recent NVIDIA driver for the selected PyTorch CUDA runtime.

No additional pip packages are required. PyTorch is supplied by ComfyUI; do
not install another Torch build into a working portable installation merely to
install this node.

The bundled `.pyd` is ABI-specific. Other Python versions, Linux, or a different
Torch/CUDA ABI are not supported by this prebuilt release.

## Published benchmark configuration

The performance and boundary data in this repository were collected with the
following reproducible configuration. Model names are public project names;
machine-specific paths and workflow-local data are intentionally omitted.

| Component | Benchmark configuration |
|---|---|
| GPUs | 2 x NVIDIA Tesla V100-SXM2 16 GiB |
| System memory | 128 GiB RAM |
| H3 diffusion model | MiniMax H3 Ref2VA INT8 ConvRot, assigned to the second V100 |
| Text encoder | Qwen3-VL 32B MiniMax H3 INT8 ConvRot, assigned to CPU |
| Video VAE | MiniMax H3 Video VAE FP16, assigned to the first V100 |
| Audio VAE | MiniMax H3 Audio VAE FP32, assigned to the first V100 |
| Acceleration LoRA | MiniMax H3 FL2V Turbo 8-step BF16, strength 0.75 |
| Sampling | 8 sampler steps, no step skipping |
| Global FP16 accumulation | Disabled |

Reported timings are therefore **dual-V100 workflow results**, not single-GPU
throughput claims. For optimization A/B tests, keep the prompt, references,
seed, model placement and sampler settings identical and compare hot runs. The
LoRA's BF16 distribution label does not mean that
the complete H3 runtime executes in BF16.

### Published v1.1.2 measured speed improvement

In the fixed-seed 864x480, 5-second, 18,376-token hot-run comparison above,
the sampler ran 8 steps with no step skipping, and
the original path without this node took **738 seconds**. Enabling mixed
precision reduced the run to **380.11 seconds**: throughput improved by
**94.2%** (1.94x) and elapsed time fell by **48.5%**. Enabling Flash Attention
on top of mixed precision saved another **39.94 seconds**, reducing the run to
**340.17 seconds**; that is a further **11.7% throughput improvement** and
**10.5% elapsed-time reduction** relative to mixed precision alone. Overall,
the fully enabled node delivered **117.0% higher throughput** (2.17x) than the
original path and reduced total elapsed time by **53.9%**.

The GPUs experienced thermal throttling during testing, so the measured
elapsed times and speedups above should be treated as conservative. With
adequate cooling and stable clocks, the effective optimization gain may be
higher.

These numbers are retained from the published v1.1.2 baseline. They must not be
used as v1.3.0 performance claims. Updated Flash/Sol, 17K/25K and cold/hot
results will be added only after the release-candidate test matrix is complete.

### v1.3.0 candidate single-V100 test

The following warm-run test uses a different environment from the published
dual-V100 baseline and must be read separately:

| Item | Configuration |
|---|---|
| Workflow | MiniMax H3 text-to-video |
| Output | 608x352, 15 seconds |
| Sampling | 8 steps |
| GPU | One NVIDIA V100 16 GiB; no manual component assignment |
| Loaders | Default CLIP, diffusion-model and VAE loaders; CPU and GPU 0 |
| Launcher | `--lowvram --disable-dynamic-vram` |
| Start state | Warm; cold model-loading time intentionally excluded |
| Thermal state | GPU frequency was reduced by cooling-related throttling |

| Route | Final sampler average | Estimated 8-step sampler time | Full prompt time |
|---|---:|---:|---:|
| No H3 V100 node | 214.71 s/step | 1,717.68 s | 1,774.00 s (29:34) |
| H3 V100 `flash_attn` | 62.21 s/step | 497.68 s | 550.68 s |
| H3 V100 `sol_attn` | 62.51 s/step | 500.08 s | 552.33 s |

In this run, Flash reduced measured sampler elapsed time by 71.0% relative to
the no-node compatibility reference (3.45x sampler throughput), while Sol
reduced it by 70.9% (3.43x). Sol was about 0.5% slower than Flash in sampler
time and 0.3% slower in full prompt time, which is too small and too workload-specific
to claim a general backend advantage.

The no-node result is not a clean measurement of ComfyUI's fastest native path:
the two required launcher options disable parts of ComfyUI's dynamic-VRAM
optimization. They were kept enabled so all three runs used the same process
configuration and so the V100 16 GiB workflow remained runnable. Treat the
no-node row as a same-launcher compatibility reference, not an unrestricted
native baseline. This is one thermally throttled run per route; repeated hot
runs remain pending. Non-sampler time was internally consistent at 56.32,
53.00 and 52.25 seconds, which supports that the reported difference is located
in the denoising stage rather than VAE decoding or other workflow overhead.

## Installation

1. Copy the `H3_V100` directory into `ComfyUI/custom_nodes/`.
2. Confirm that `comfy_v100_flash_attn_cuda.cp312-win_amd64.pyd` is present in
   the directory when Flash Attention is required.
3. Restart ComfyUI.
4. Insert `H3_V100` after the H3 model loader and before the sampler.

Do not stack this node with another extension that independently patches the
same H3 attention, MLP, QKV or output-projection methods. Cache/step-skipping
extensions are outside the default exact route; when TE-Speed is evaluated,
place it before H3 V100 and keep the required launcher flags below.

## Important launcher setting

For the validated 16 GiB V100 route, use these two ComfyUI launcher options
together:

```text
--disable-dynamic-vram --lowvram
```

This is a compatibility requirement for the tested V100 16 GiB configuration,
not a performance switch. `--disable-dynamic-vram` prevents dynamic weight
staging/prefetch from expanding during an H3 activation peak, while `--lowvram`
keeps the resident weight set small enough for AIMDO streaming. PyTorch
reserved cache is not equivalent to driver-free VRAM: AIMDO can fail on the
next 64 MiB direct weight copy even when several GiB appear reusable inside the
allocator. The flags do not disable this node's adaptive MLP/QKV chunking.
Larger-VRAM cards and other weight-loading backends remain outside the current
validation boundary. These are process-wide startup options and cannot be
enabled by this node.

Do **not** enable ComfyUI's global `--fast fp16_accumulation` option on V100.
It is separate from this node's mixed precision. V100 validation showed
no meaningful QKV gain, a small MLP slowdown, and measurable numerical changes.
Leaving it disabled preserves FP32 accumulation for ordinary FP16 GEMMs and
does not disable this node's FP16, mixed precision or Flash Attention paths.

Also avoid treating global `--fast` options as node-local settings: they can
affect every compatible model in the ComfyUI process.

## Limits

Fixed-seed and repeated H3 tests covered short and long sequences, cold and hot
starts, mixed-precision/Flash ablations, video output and generated audio. No
obvious picture or audio degradation was observed in the tested optimized
profiles. This does not prove bit-identical output or guarantee every workflow.

The adaptive policy includes measured guards from 16 GiB V100 tests:

- A 32,917-token case completed without MLP chunking.
- A 36,176-token full allocation failed; adaptive chunking completed it.
- A 102,623-token case exposed allocator-cache overestimation; direct AIMDO
  copies now use driver-free memory rather than PyTorch cache as their budget.

Extreme sequences: under the published dual Tesla V100-SXM2 16 GiB and
128 GiB system-memory configuration, 1280x736 at 15 seconds (102,623 tokens)
completed all 8 sampling steps and produced a video in 1:51:58. The run used
4,096-token MLP chunks and 1,024-token QKV/output-projection chunks. This is
the highest fully validated feasibility tier, not a recommended everyday tier.

At 1376x768 and 15 seconds (114,607 tokens), the former policy OOMed while
expanding the first QKV weight. The VRAM-scaled extreme compatibility policy
then used 512-token QKV/output-projection chunks, a pre-QKV cache trim, and
chunk-local FP16 conversion, and successfully completed the first sampling
step with about 14,023 MiB peak PyTorch allocation and finite numerical checks.
The run was intentionally stopped afterward, so it validates first-step
compatibility only—not a completed video, final quality, or full runtime. The
reference extreme threshold is about 110K tokens on a 16 GiB V100 and scales
with physical VRAM and live memory pressure on other configurations.

## References and acknowledgements

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI) provides the host model and
  execution framework.
- The sparse route is informed by NVIDIA's
  [Sol-Attn project](https://nvlabs.github.io/Sana/Sol-Attn/) and
  [paper](https://arxiv.org/abs/2607.24027), a training-free on-the-fly
  block-sparsification method. This project provides an independent H3 and
  V100/SM70 adaptation; Sol is explicitly approximate, and results reported
  for other models and hardware are not performance claims for this node.
- The H3 mixed-precision split is adapted from
  [Icbears/minimax-h3-v100-patch](https://github.com/Icbears/minimax-h3-v100-patch).
  This project converts that source-patching approach into a workflow-scoped
  node and adds audio-safe execution and adaptive memory guards.
- The native SM70 attention work is derived from
  [Icbears/flash-attention-v100](https://github.com/Icbears/flash-attention-v100).
- Native kernel building uses [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass)
  4.2.0 headers, including CuTe.
- PyTorch SDPA is used as the correctness/performance fallback and benchmark
  reference where applicable.

See `NOTICE.md` and `licenses/` for attribution and license information.

## License

This distribution is GPL-3.0-only because the H3 mixed-precision-derived
portion is GPL-3.0-only. BSD-3-Clause components retain their notices. See
`NOTICE.md` and `licenses/`.
