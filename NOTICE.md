# Notices

`h3_mixed_precision.py` is distributed under GPL-3.0-only. Its full license is
stored in `licenses/H3_MIXED_PRECISION_GPL-3.0.txt`.

The H3 V100 mixed-precision split is adapted from:
https://github.com/Icbears/minimax-h3-v100-patch

This package reimplements that file patch as a workflow-scoped ComfyUI MODEL
patch and adds audio-safe handling and bounded-memory execution. Do not enable
both projects on the same model because both replace H3 attention.

The V100 FlashAttention CUDA implementation and related Python integration are
distributed under the BSD 3-Clause license. Its full license is stored in
`licenses/FLASH_ATTENTION_BSD-3-CLAUSE.txt`.

The SM70 FlashAttention implementation is derived from:
https://github.com/Icbears/flash-attention-v100

CUTLASS/CuTe headers used to build the bundled binary came from NVIDIA CUTLASS
4.2.0. Applicable attribution and license terms are preserved in this notice
and the bundled license files.
