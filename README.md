# Open Computational Chemistry (OCC)
<img src="https://github.com/peterspackman/occ/raw/main/docs/static/occ.png" width=640/>

[![Build & Test](https://github.com/peterspackman/occ/actions/workflows/build_test.yml/badge.svg)](https://github.com/peterspackman/occ/actions/workflows/build_test.yml)

A next generation quantum chemistry and crystallography program and library.

**Note**: occ is in early development, and is undergoing substantial changes regularly - it is not stable, and features are being added & developed rapidly.

## Features

### Quantum chemistry 

Occ is already a fairly fully featured program for ground-state single point calculations in quantum chemistry, including:
- Hartree-Fock (Restricted, Unrestricted and General Spinorbitals)
- Density-Functional Theory (Restricted & Unrestricted Spinorbitals)
    - The LDA, GGA and meta-GGA approximations are supported
    - Global hybrid functionals (range-separated will be added in the future)
- Density fitting (RI-JK) using an auxiliary basis for all above methods
- Implicit solvation via SMD

Seminumerical exchange (i.e. chain of spheres/COSX) has been implemented, but the performance is not yet good enough to be useful.

Property calculations that are currently available
- Molecular and atomic multipole moments up to hexadecapole (only Mulliken partitioning is implemented)
- Electrostatic potential calculations
- Electron density (of course)
- CHELPG charges

I've recently added an implementation of the XDM dispersion model, which will be properly interfaced and made convenient to use
in the future.

Not yet implemented:
- Gradients (and optimization of geometries)
- Perturbation theory (e.g. MP2)
- Coupled-cluster methods

### Crystal structures

- Reading CIF files (via gemmi)
- Fast periodic bond detection, generation of symmetry unique molecules, dimers and more...
- CrystalExplorer model energies
- Automatic direct space summation of lattice energies for neutral molecular crystals including Wolf summation.
- Hirshfeld surfaces, and promolecule surfaces

### Misc
- Spherical harmonic transforms using FFTs
- Molecular point group detection/determination
- Reading/writing Gaussian fchk files (including MO normalization and reordering of basis functions)
- Reading molden files (including MO normalization and reordering of basis functions)
- Writing numpy `.npy` arrays
- Reading QCSchema formatted JSON files.
- Reading basic Gaussian input files
- Marching cubes
- Morton codes for linear-hashed octrees
- Electronegativity equilibration method for charges

First steps have been taken, with a proof of concept python interface for convenience & scripting using pybind11.

## Compilation

occ requires a compliant C++17 compiler e.g. GCC-10 or newer.

### Dependencies

occ makes use of the the following open source libraries:

- [cxxopts](https://github.com/jarro2783/cxxopts)
- [Eigen3](https://eigen.tuxfamily.org/)(`eigen3-dev`)
- [fmt](https://github.com/fmtlib/fmt)
- [gau2grid](https://github.com/dgasmith/gau2grid)
- [gemmi](https://gemmi.readthedocs.io/)
- [LBFGS++](https://lbfgspp.statr.me/)
- [libcint](https://github.com/sunqm/libcint)
- [libxc](http://www.tddft.org/programs/libxc/)
- [nanoflann](https://github.com/jlblancoc/nanoflann)
- [nlohmann/json](https://github.com/nlohmann/json)
- [pocketFFT](https://github.com/hayguen/pocketfft)
- [scnlib](https://github.com/eliaskosunen/scnlib)
- [spdlog](https://github.com/gabime/spdlog)
- [zlib](https://zlib.net/) (`zlibc`)

And for the library tests/benchmarks:

- [catch2](https://github.com/catchorg/Catch2)


### Getting the source code

First clone the repository:
```
git clone https://github.com/peterspackman/occ.git
```

### Getting dependencies

Most of the dependencies can be downloaded and compiled via [CPM](https://github.com/cpm-cmake/CPM.cmake),
but you may wish to use system installed dependencies for `libxc`, `zlibc` and `eigen3` which will be searched for by default.
Note `occ` requires eigen version 3.4 or newer, which most operating systems do not package by default.

### Caching dependency downloads

If you wish to download and compile all dependencies, but you're a developer or want to avoid downloading
the dependencies every new build, I'd recommend setting up a source cache for CPM
via the environment variable `CPM_SOURCE_CACHE` e.g. adding the following to your environment.

```
export CPM_SOURCE_CACHE="$HOME/.cache/cpm"
```

#### 

For building the repository I highly recommend using [ninja](https://ninja-build.org/) rather
than make.

Once the dependencies are installed, start an out-of-source build e.g.
```
mkdir build && cd build
cmake .. -GNinja
```

**OR**, if you'd rather download all dependencies you could call cmake with:

```
cmake .. -GNinja \
    -DUSE_SYSTEM_LIBXC=OFF \
    -DUSE_SYSTEM_ZLIB=OFF \
    -DUSE_SYSTEM_EIGEN=OFF
```

Generally, speedups can be achieved by allowing the compiler to optimize for your platform using `-march=native` or similar flags.

Finally, to build the binary `occ`, running

```
ninja occ
```

will result in the binary being built under the `bin` directory wherever
your build directory is located

## Usage

All following usage is a work in progress, expect significant changes
constantly for the time-being while the exact input format is decided.

### occ

By default `occ -h` will print out its usage options, but basic usage
given a geometry e.g. `h2o.xyz` format would be:

```
occ h2o.xyz b3lyp 6-31g
```

### Basis set locations
*note* The path the `occ` will use to search for basis sets can be configured with the `OCC_BASIS_PATH` environment variable,
or you can simply make the basis set available in your working directory.
