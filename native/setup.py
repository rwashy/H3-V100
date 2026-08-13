import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
CSRC = ROOT / "csrc"
FLASH = CSRC / "flash_attn"
PYTHON_DEV = Path(
    os.environ.get(
        "V100_FA_PYTHON_DEV", ROOT.parent / "python-dev-3.12.10" / "tools"
    )
)
sdk_candidates = sorted(
    Path(r"C:\Program Files (x86)\Windows Kits\10\bin").glob(r"*\x64\rc.exe")
)
WINDOWS_SDK_BIN = sdk_candidates[-1].parent if sdk_candidates else Path()


class WindowsSdkBuildExtension(BuildExtension):
    def build_extensions(self):
        if not self.compiler.initialized:
            self.compiler.initialize()
        self.compiler._paths = os.pathsep.join(
            [str(WINDOWS_SDK_BIN), self.compiler._paths]
        )
        super().build_extensions()

extension = CUDAExtension(
    name="comfy_v100_flash_attn_cuda",
    sources=[
        "csrc/flash_attn/flash_api.cpp",
        "csrc/flash_attn/flash_api_torch_lib.cpp",
        "csrc/flash_attn/src/flash_fwd_hdim128_sm70.cu",
        "csrc/flash_attn/src/flash_fwd_hdim256_sm70.cu",
    ],
    include_dirs=[
        str(FLASH),
        str(FLASH / "src"),
        str(CSRC / "common"),
        str(ROOT / "third_party" / "cutlass" / "include"),
        str(PYTHON_DEV / "include"),
    ],
    library_dirs=[str(PYTHON_DEV / "libs")],
    define_macros=[
        ("FLASH_NAMESPACE", "comfy_v100_flash"),
        ("FLASHATTENTION_DISABLE_BACKWARD", None),
        ("FLASHATTENTION_DISABLE_DROPOUT", None),
        ("FLASHATTENTION_DISABLE_ALIBI", None),
        ("FLASHATTENTION_DISABLE_SOFTCAP", None),
        ("FLASHATTENTION_DISABLE_LOCAL", None),
        ("FLASHATTENTION_DISABLE_PYBIND", None),
        ("COMFY_V100_HDIM128_256_ONLY", None),
        ("CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL", "1"),
        ("NOMINMAX", None),
    ],
    extra_compile_args={
        "cxx": ["/O2", "/std:c++17", "/Zc:preprocessor"],
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-lineinfo",
            "-gencode=arch=compute_70,code=sm_70",
            "-Xcompiler=/Zc:preprocessor",
            "-DFLASH_NAMESPACE=comfy_v100_flash",
            "-DFLASHATTENTION_DISABLE_BACKWARD",
            "-DFLASHATTENTION_DISABLE_DROPOUT",
            "-DFLASHATTENTION_DISABLE_ALIBI",
            "-DFLASHATTENTION_DISABLE_SOFTCAP",
            "-DFLASHATTENTION_DISABLE_LOCAL",
            "-DFLASHATTENTION_DISABLE_PYBIND",
            "-DCOMFY_V100_HDIM128_256_ONLY",
            "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
            "-DNOMINMAX",
        ],
    },
)

setup(
    name="comfy-v100-flash-attn",
    version="0.1.0",
    ext_modules=[extension],
    cmdclass={"build_ext": WindowsSdkBuildExtension.with_options(use_ninja=True)},
)
