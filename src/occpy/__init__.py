from pathlib import Path
import site
import warnings
from ._occpy import setup_logging, set_data_directory, set_num_threads
from .core import Atom, Element, Molecule, Dimer
from .crystal import Crystal
from .qm import AOBasis, HartreeFock, Shell, Wavefunction
from .dft import DFT

# Set up logging first
setup_logging(0)

# Get site-packages directory and construct path to data
_site_packages = Path(site.getsitepackages()[0])
_data_dir = _site_packages / "share" / "occ"

# Warn if directory not found, but continue
if not _data_dir.exists():
    warnings.warn(f"OCC data directory not found at expected location: {_data_dir}")

set_data_directory(str(_data_dir))

__all__ = [
    "Atom",
    "AOBasis",
    "calculate_crystal_growth_energies",
    "core",
    "crystal",
    "Crystal",
    "dft",
    "DFT",
    "Dimer",
    "Element",
    "HartreeFock",
    "Molecule",
    "qm",
    "set_data_directory",
    "set_num_threads",
    "setup_logging",
    "Shell",
    "Wavefunction",
]
