# Changelog

## v1.4.1

- Fixed repeat-run AIMDO `hostbuf_read_file_slice` process aborts by releasing
  an inactive, node-managed H3 DynamicVRAM model before the next prompt loads
  its text encoder.
- Cleans completed prefetch queues at that phase boundary before detaching H3,
  preventing stale VBAR pins or stream state from crossing into the next run.
- Limits the new load guard to models explicitly marked by H3 V100 Optimize;
  unrelated DynamicVRAM models retain ComfyUI's normal residency behavior.

## v1.4.0

- Promoted the validated DynamicVRAM profile to stable.
- Made Dynamic VBAR ownership and scaled FP16 SwiGLU built-in behavior.
- Added MLP-invocation-local fc1/fc2 expanded-weight reuse, eliminating the
  large cold-start penalty caused by repeated preparation for every chunk.
- Added role/state-based release of inactive prior-stage CUDA Dynamic models.
- Updated the validated launcher profile: no `--disable-dynamic-vram`, no
  `--lowvram`, and no global
  `--fast fp16_accumulation` on V100.
