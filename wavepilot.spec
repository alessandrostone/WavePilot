# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Get the directory containing this spec file
spec_dir = Path(SPECPATH)

block_cipher = None

# Define the main script
main_script = spec_dir / "wavepilot_bridge_main.py"

# Collect all Python files in the wavepilot directory
wavepilot_files = []
for py_file in spec_dir.glob("*.py"):
    if py_file.name != "wavepilot_bridge_main.py":
        wavepilot_files.append((str(py_file), "."))

# Essential imports only - focusing on the bridge functionality
hidden_imports = [
    'json',
    'pathlib',
    'argparse',
    'traceback',
    'threading',
    'sys',
    'os',
]

a = Analysis(
    [str(main_script)],
    pathex=[str(spec_dir)],
    binaries=[],
    datas=wavepilot_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='wavepilot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)