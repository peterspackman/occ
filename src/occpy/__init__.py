from pathlib import Path
import site
import warnings
from ._occpy import *

# Set up logging first, only log critical errors
set_log_level(LogLevel.CRITICAL)

# Get site-packages directory and construct path to data
_site_packages = Path(site.getsitepackages()[0])
_data_dir = _site_packages / "share" / "occ"

# Warn if directory not found, but continue
if not _data_dir.exists():
    warnings.warn(f"OCC data directory not found at expected location: {_data_dir}")

set_data_directory(str(_data_dir))

def run_occ_executable():
    import subprocess
    import sys
    import os
    from pathlib import Path

    env = os.environ.copy()
    site_packages = Path(site.getsitepackages()[0])
    data_dir = site_packages / "share" / "occ"
    occ_path = site_packages / "bin" / "occ"

    if data_dir.exists():
        env["OCC_DATA_PATH"] = str(data_dir)

    subprocess.run([str(occ_path)] + sys.argv[1:], env=env)
