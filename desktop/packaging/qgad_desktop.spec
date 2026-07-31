# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

ROOT = Path.cwd()

CUDA_BINARY_PATTERNS = (
    "cublas",
    "cudart",
    "cudnn",
    "cufft",
    "cufile",
    "curand",
    "cusolver",
    "cusparse",
    "cusparselt",
    "cudss",
    "magma",
    "nccl",
    "nvjitlink",
    "torch_cuda",
)


def _is_cuda_binary(entry) -> bool:
    name = Path(entry[0]).name.lower()
    return any(name.startswith(prefix) for prefix in CUDA_BINARY_PATTERNS)


def _filter_cpu_only_binaries(binaries):
    return [entry for entry in binaries if not _is_cuda_binary(entry)]

datas = [
    (str(ROOT / "desktop" / "assets"), "desktop/assets"),
    (str(ROOT / "desktop" / "resources"), "desktop/resources"),
    (str(ROOT / "desktop" / "storage"), "desktop/storage"),
]

hiddenimports = [
    "numpy",
    "pandas",
    "networkx",
    "scipy",
    "data.financial_dataset",
    "data.elliptic_dataset",
    "gbs.gbs_kernel",
    "models.hybrid_classifier",
    "utils.graph_utils",
    "utils.helpers",
]

excludes = [
    "IPython",
    "PIL",
    "PySide2",
    "PySide6",
    "matplotlib",
    "matplotlib_inline",
    "notebook",
    "pyarrow",
    "qiskit",
    "statsmodels",
    "tkinter",
    "torchvision",
    "tornado",
    "xgboost",
    "sklearn",
]

a = Analysis(
    [str(ROOT / "desktop" / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
a.binaries = _filter_cpu_only_binaries(a.binaries)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="QGADDesktop",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="QGADDesktop",
)
