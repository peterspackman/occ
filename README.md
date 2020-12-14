# Tonto-NG

A next generation quantum chemistry and crystallography program and library.


## Compilation

Tonto-NG requires a compliant C++17 compiler e.g. GCC-10 or newer
with OpenMP.

### Dependencies

Tonto-NG depends on the following libraries:

- libint2
- libxc
- Eigen3
- boost.graph
- zlib

It also depends on open source libraries packaged as submodules in the
git repository:

- spdlog
- fmtlib
- scnlib
- toml11
- Fastor
- gemmi

For tests:

- catch2

First clone the repository:
```
git clone --recurse-submodules https://github.com/peterspackman/tonto-ng.git
```

Once the dependencies are installed, start an out-of-source build e.g.
```
mkdir build && cd build
cmake "$PATH_TO_TONTONG"
make
```

The resulting binaries `tonto-ng` and `tonto-ng-ce` for calculating
single point energies and CrystalExplorer model interaction energies
will be built.


## Usage

All following usage is a work in progress, expect significant changes
constantly for the time-being while the exact input format is decided.

### tonto-ng
By default `tonto-ng` will print out its usage options, but basic usage
given a geometry e.g. `h2o.xyz` format would be:

*note* currently tonto-ng uses the default libint2 library basis sets
and location. This will be modified in a future version to be configurable.

```
tonto-ng h2o.xyz --method b3lyp --basis 6-31g
```

### tonto-ng-ce

Expects a toml file, and one/two `.fchk` or `.molden` files for wavefunctions.
