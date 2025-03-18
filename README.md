# Open Computational Chemistry (OCC)
<img src="https://github.com/peterspackman/occ/raw/main/docs/static/occ.png" width=640/>

[![Build & Test](https://github.com/peterspackman/occ/actions/workflows/build_test.yml/badge.svg)](https://github.com/peterspackman/occ/actions/workflows/build_test.yml)
[![PyPI version](https://badge.fury.io/py/occpy.svg)](https://badge.fury.io/py/occpy)
[![PyPI Downloads](https://img.shields.io/pypi/dm/occ)](https://pypi.org/project/occ/)
[![DOI](https://zenodo.org/badge/292276139.svg)](https://zenodo.org/doi/10.5281/zenodo.10703204)

A next-generation quantum chemistry and crystallography program and library, designed for modern computational workflows.

> **Note**: OCC is in active development and undergoes frequent changes. The API and features are not yet stable.

## Installation

### From PyPI

The easiest way to install OCC is via pip:

```bash
pip install occpy
```

Supported Python versions:
- Python 3.10, 3.11, 3.12, 3.13

Pre-built wheels are available for:
- Linux (x86_64)
- macOS (x86_64 and ARM64/Apple Silicon via universal2 wheels)

## Features

### Quantum Chemistry

OCC provides comprehensive functionality for ground-state single-point calculations:

- **Electronic Structure Methods**
  - Hartree-Fock (Restricted, Unrestricted, and General Spinorbitals)
  - Density-Functional Theory (Restricted & Unrestricted)
    - Supported approximations: LDA, GGA, meta-GGA
    - Global hybrid functionals (range-separated support planned)
  - Density fitting (RI-JK) with auxiliary basis sets
  - Implicit solvation via SMD
  - XDM dispersion model

- **Property Calculations**
  - Molecular and atomic multipole moments (up to hexadecapole)
  - Electron density, Electrostatic potential
  - CHELPG charges
  - Isosurfaces, generation of volumetric data and more...

### Crystal Structure Analysis

- CIF file processing (via gemmi)
- Advanced periodic analysis:
  - Fast periodic bond detection
  - Symmetry-unique molecule generation
  - Dimer identification
- Energy calculations:
  - CrystalExplorer model energies
  - Automatic direct space lattice energy summation
  - Wolf summation for neutral molecular crystals
- Surface analysis:
  - Hirshfeld surfaces
  - Promolecule surfaces

### Additional Features

- Spherical harmonic transforms (FFT-based)
- Molecular point group detection
- File format support:
  - Gaussian fchk files (read/write)
  - Molden files
  - NumPy `.npy` arrays (write)
  - QCSchema JSON
  - Basic Gaussian input files
- Geometric algorithms:
  - Marching cubes
  - Morton codes for linear-hashed octrees
- Electronegativity equilibration method for charges
- Python bindings via nanobind

## Python API Examples

```python
import occpy
from occpy import Crystal, Molecule, AOBasis, HartreeFock, DFT
from occpy.qm import SpinorbitalKind

# Set up basic configuration
occpy.set_log_level(occpy.LogLevel.WARN)  # Configure logging level
# occpy.set_data_directory("/path/to/basis/sets")  # Optional: Set basis set path

# Load molecule from XYZ file
mol = Molecule.from_xyz_file("h2o.xyz")

# Basic Restricted Hartree-Fock calculation
basis = AOBasis.load(mol.atoms(), "6-31G")
hf = HartreeFock(basis)
scf = hf.scf()
energy = scf.run()
wfn = scf.wavefunction()

# DFT calculation
dft = DFT("B3LYP", basis)
ks = dft.scf(SpinorbitalKind.Unrestricted)
ks.set_charge_multiplicity(0, 1)
energy = ks.run()

# Crystal structure analysis
crystal = Crystal.from_cif_file("structure.cif")
dimers = crystal.symmetry_unique_dimers(10.0)  # Get unique dimers within 10 Å
```

For more examples and detailed API documentation, please refer to the [documentation](docs_url_here).

## Build from Source

### Prerequisites

- C++17 compliant compiler (GCC 10+ recommended)
- CMake 3.15+
- Ninja (recommended) or Make

### Dependencies

OCC uses modern C++ libraries to provide its functionality:

| Library | Version | Description |
|---------|---------|-------------|
| [CLI11](https://github.com/CLIUtils/CLI11) | 2.4.2 | Command line parser |
| [Eigen3](https://eigen.tuxfamily.org/) | 3.4.0+ | Linear algebra |
| [fmt](https://github.com/fmtlib/fmt) | 11.0.2 | String formatting |
| [gemmi](https://gemmi.readthedocs.io/) | 0.6.5 | Crystallographic file handling |
| [LBFGS++](https://lbfgspp.statr.me/) | master | Optimization algorithms |
| [libcint](https://github.com/sunqm/libcint) | 6.1.2 | Gaussian integrals |
| [libxc](http://www.tddft.org/programs/libxc/) | 6.2.2 | Exchange-correlation functionals |
| [nlohmann/json](https://github.com/nlohmann/json) | 3.11.3 | JSON handling |
| [scnlib](https://github.com/eliaskosunen/scnlib) | 4.0.1 | String parsing |
| [spdlog](https://github.com/gabime/spdlog) | 1.15.0 | Logging |
| [unordered_dense](https://github.com/martinus/unordered_dense) | 4.5.0 | Hash containers |

Optional dependencies:
- [nanobind](https://github.com/wjakob/nanobind) (2.4.0) - For Python bindings

Most dependencies are automatically handled through [CPM](https://github.com/cpm-cmake/CPM.cmake). System-installed versions of Eigen3 and libxc can be used if available.

### Build Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/peterspackman/occ.git
   cd occ
   ```

2. Configure dependency caching (recommended):
   ```bash
   export CPM_SOURCE_CACHE="$HOME/.cache/cpm"
   ```

3. Build with CMake:
   ```bash
   mkdir build && cd build
   
   # Using system dependencies (if available)
   cmake .. -GNinja

   # OR download all dependencies
   cmake .. -GNinja -DUSE_SYSTEM_LIBXC=OFF -DUSE_SYSTEM_EIGEN=OFF

   # Build the executable
   ninja occ
   ```

### CMake Options

- `USE_SYSTEM_LIBXC`: Use system-installed libxc (default: ON)
- `USE_SYSTEM_EIGEN`: Use system-installed Eigen3 (default: ON)
- `WITH_PYTHON_BINDINGS`: Build Python bindings (default: OFF)
- `USE_MLX`: Enable MLX integration (default: OFF)
- `USE_QCINT`: Use QCInt instead of libcint (default: OFF)
- `ENABLE_HOST_OPT`: Enable host-specific optimizations (default: OFF)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Citation

If you use OCC in your research, please cite the appropriate papers for all functionals, methods etc. you use, along with the citations for the core dependencies here.
