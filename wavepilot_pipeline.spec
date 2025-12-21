# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for WavePilot training pipeline

from pathlib import Path

spec_dir = Path(SPECPATH)
main_script = spec_dir / "pipeline.py"

block_cipher = None

# Collect all local Python modules needed by pipeline.py
local_modules = [
    'constants.py',
    'data.py',
    'logger.py',
    'model.py',
    'optimize.py',
    'serialization.py',
    'utils.py',
]

datas = [(str(spec_dir / mod), '.') for mod in local_modules if (spec_dir / mod).exists()]

a = Analysis(
    [str(main_script)],
    pathex=[str(spec_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'numpy',
        'torch',
        'scipy',
        'scipy.spatial',
        'scipy.spatial.distance',
        'scipy.interpolate',
        'pandas',
        'optuna',
        'sklearn',
        'sklearn.model_selection',
        'sklearn.preprocessing',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',  # Not needed for training
        'tkinter',
        'PyQt5',
        'PyQt6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='wavepilot_pipeline',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir='.',
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
