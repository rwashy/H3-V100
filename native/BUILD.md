# Isolated Windows build

This source is a narrow ComfyUI proof of concept derived from
`Icbears/flash-attention-v100`. It builds a uniquely named extension and never
installs or removes the standard `flash_attn` package.

Supported fast path:

- SM70 / NVIDIA V100
- FP16 forward inference
- non-causal kernel, no native mask, no dropout
- `head_dim=128` and `head_dim=256`
- no split-KV or paged KV

The `H3_V100` package uses the original H3 `head_dim=128` selection rules.
Unsupported shapes are never passed to this kernel.

The tested interpreter is ComfyUI's CPython 3.12.10 with PyTorch 2.8.0+cu128.
Visual Studio 2022 C++ tools, Windows SDK, and CUDA Toolkit 12.8 are required.

From this source directory, prepare build-only Python packages in the sibling
directory expected by `build_isolated.py`:

```powershell
$python = 'C:\path\to\ComfyUI\python_embeded\python.exe'
& $python -m pip install --target ..\sm70-build-deps `
  'setuptools>=77,<81' ninja packaging wheel jinja2 psutil
```

Download the matching Python development files into another sibling directory:

```powershell
curl.exe -L 'https://www.nuget.org/api/v2/package/python/3.12.10' `
  -o ..\python.3.12.10.nupkg
New-Item -ItemType Directory -Force ..\python-dev-3.12.10 | Out-Null
tar.exe -xf ..\python.3.12.10.nupkg -C ..\python-dev-3.12.10
```

Build and test without installation:

```powershell
& $python .\build_isolated.py
& $python .\test_extension.py
& $python .\benchmark_comfy.py --lengths 2048 8192 --heads 56 --iterations 10
```

Optional environment variables:

- `CUDA_PATH`: CUDA Toolkit root
- `V100_FA_BUILD_DEPS`: isolated build-package directory
- `V100_FA_PYTHON_DEV`: directory containing Python `include` and `libs`

The resulting `comfy_v100_flash_attn_cuda*.pyd` is ABI-specific to Python,
PyTorch, and CUDA. Rebuild it after changing any of those components.
