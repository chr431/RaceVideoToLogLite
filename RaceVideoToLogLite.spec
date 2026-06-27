# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for RaceVideoToLogLite — single-file executable, v5 models only."""
from __future__ import annotations
from pathlib import Path
import rapidocr_onnxruntime as _rr

_rr_root = Path(_rr.__file__).parent

# ── Only include v5 models + essential configs ──
_datas: list[tuple[str, str]] = []
for _pattern in [
    "config.yaml",
    "models/ch_PP-OCRv5_mobile_det_infer.onnx",
    "models/ch_PP-OCRv5_mobile_rec_infer.onnx",
]:
    for _f in _rr_root.glob(_pattern):
        _dest = str(Path(_f).relative_to(_rr_root.parent).parent)
        _datas.append((str(_f), _dest))

a = Analysis(
    ["lite.py"],
    pathex=[],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        "rapidocr_onnxruntime",
        "rapidocr_onnxruntime.ch_ppocr_v3_det",
        "rapidocr_onnxruntime.ch_ppocr_v3_rec",
        "rapidocr_onnxruntime.ch_ppocr_v2_cls",
        "rapidocr_onnxruntime.utils",
        "onnxruntime",
        "onnxruntime.capi",
        "cv2",
        "numpy",
        "yaml",
        "shapely",
        "pyclipper",
        "PIL",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "unittest",
        "test",
        "pydoc",
        "distutils",
        "setuptools",
        "pip",
        "wheel",
        "email",
        "http",
        "html",
        "xmlrpc",
        "urllib",
        "ftplib",
        "smtplib",
        "socketserver",
        "ssl",
        "_ssl",
        "asyncio",
        "multiprocessing",
        "concurrent.futures",
        "ctypes.test",
        "lib2to3",
        "sqlite3",
        "ensurepip",
        "venv",
        "wsgiref",
        "turtle",
        "turtledemo",
        "idlelib",
        "zoneinfo",
    ],
    noarchive=False,
    optimize=2,
)

# ── Strip OpenCV FFmpeg DLL — Windows native DShow/MSMF handles video I/O ──
_SKIP_BINARIES = {"opencv_videoio_ffmpeg"}
a.binaries = [
    (src, dest, typecode)
    for (src, dest, typecode) in a.binaries
    if not any(skip in Path(dest).name for skip in _SKIP_BINARIES)
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="RaceVideoToLogLite",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
