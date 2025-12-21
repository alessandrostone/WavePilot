# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

spec_dir = Path(SPECPATH)
main_script = spec_dir / "minimal_server.py"

block_cipher = None

a = Analysis(
    [str(main_script)],
    pathex=[str(spec_dir)],
    binaries=[],
    datas=[], # No other python files needed
    hiddenimports=[],
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
    name='minimal_server', # New name
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