# Changelog

## v1.3.0

- Added explicit `flash_attn` and `sol_attn` backends, including Sol threshold
  and diffusion-window controls.
- Added adaptive, VRAM-aware MLP/QKV/normalization/RoPE/output-projection
  chunking and safer cache/weight-streaming coordination for V100 16 GiB.
- Consolidated the public interface to one mixed-precision switch, one
  attention selector and the required Sol controls.
- Preserved validated FP32 boundaries and exact fallback paths; Flash remains
  the exact route, while Sol is an explicitly selected sparse approximation.
- Removed first-run attention auto-benchmarking and improved compatibility with
  ComfyUI model offloading and external wrappers.
- Moved repetitive cache-trim telemetry behind internal diagnostics so normal
  generation logs remain concise.
- Documented the required validated launcher pair:
  `--lowvram --disable-dynamic-vram`.

## v1.1.2

- Initial public MiniMax H3 optimization node for NVIDIA V100 / SM70.
- Added validated mixed precision, V100 Flash Attention and bounded-memory
  execution for long video sequences.
