from pathlib import Path
import site
import warnings
from ._occpy import setup_logging, set_data_directory
from ._occpy import *

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
    "AOBasis",
    "AsymmetricUnit",
    "Atom",
    "BeckeGridSettings",
    "calculate_crystal_growth_energies",
    "Crystal",
    "CrystalAtomRegion",
    "CrystalDimers",
    "CrystalGrowthConfig",
    "CGDimer",
    "CGEnergyTotal",
    "CGResult",
    "DFT",
    "Dimer",
    "DimerSolventTerm",
    "Element",
    "HartreeFock",
    "HF",
    "HKL",
    "KS",
    "LatticeConvergenceSettings",
    "MolecularOrbitals",
    "Molecule",
    "set_data_directory",
    "set_num_threads",
    "setup_logging",
    "Shell",
    "SymmetryRelatedDimer",
    "UnitCell",
    "Wavefunction",
]
