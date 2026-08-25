# Changelog

## v1.4.0

- Promoted the validated DynamicVRAM profile to stable.
- Made Dynamic VBAR ownership and scaled FP16 SwiGLU built-in behavior.
- Added MLP-invocation-local fc1/fc2 expanded-weight reuse, eliminating the
  large cold-start penalty caused by repeated preparation for every chunk.
- Added role/state-based release of inactive prior-stage CUDA Dynamic models.
- Updated the validated launcher profile: no `--disable-dynamic-vram`, no
  `--lowvram`, and no global
  `--fast fp16_accumulation` on V100.
