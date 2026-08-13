# Changelog

All notable changes to this project are documented in this file.

## 1.1.2

- Finalized the standalone MiniMax H3-only public node.
- Kept adaptive bounded-memory execution always enabled and independent of the
  two exposed controls.
- Synchronized the latest VRAM-aware long-sequence QKV, output-projection and
  MLP chunking policy.
- Preserved the validated FP32 attention output-projection boundary.
- Removed development diagnostics and unrelated model-specific code.
- Added bilingual benchmark, operating-tier, ABI, attribution and thermal-
  throttling documentation.
- Ensured the README operating-region image is tracked by Git.
- Ensured the tested Windows CPython 3.12 native extension is included in the
  source repository.
