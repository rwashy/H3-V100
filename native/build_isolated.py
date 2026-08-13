import os
from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parent
DEPS = Path(os.environ.get("V100_FA_BUILD_DEPS", ROOT.parent / "sm70-build-deps"))
CUDA = Path(
    os.environ.get(
        "CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    )
)
sdk_candidates = sorted(
    Path(r"C:\Program Files (x86)\Windows Kits\10\bin").glob("*\x64\rc.exe")
)
WINDOWS_SDK_BIN = sdk_candidates[-1].parent if sdk_candidates else Path()

sys.path.insert(0, str(DEPS))
try:
    import ninja

    NINJA_BIN = Path(ninja.BIN_DIR)
except ImportError:
    NINJA_BIN = DEPS / "bin"
os.environ["CUDA_HOME"] = str(CUDA)
os.environ["CUDA_PATH"] = str(CUDA)
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0"
os.environ["MAX_JOBS"] = os.environ.get("MAX_JOBS", "1")
os.environ["PATH"] = os.pathsep.join(
    [str(NINJA_BIN), str(CUDA / "bin"), str(WINDOWS_SDK_BIN), os.environ["PATH"]]
)

sys.argv = [
    str(ROOT / "setup.py"),
    "build_ext",
    "--inplace",
    "--build-temp",
    str(ROOT.parent / "bt"),
]
os.chdir(ROOT)
runpy.run_path(str(ROOT / "setup.py"), run_name="__main__")
