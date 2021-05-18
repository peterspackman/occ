# Open Computational Chemistry (OCC)

A next generation quantum chemistry and crystallography program and library.


## Compilation

occ requires a compliant C++17 compiler e.g. GCC-10 or newer
with OpenMP.

### Dependencies

occ depends on the following libraries:

- [libint2](https://github.com/evaleev/libint/releases/tag/v2.7.0-beta.6)
- [libxc](http://www.tddft.org/programs/libxc/down.php?file=5.0.0/libxc-5.0.0.tar.gz)
- Eigen3 (`eigen3-dev`)
- boost.graph (`libboost-graph1.65-dev`)
- zlib (`zlibc`)

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
git clone --recurse-submodules https://github.com/peterspackman/occ.git
```

Once the dependencies are installed, start an out-of-source build e.g.
```
mkdir build && cd build
cmake "$PATH_TO_OCC"
make
```

The resulting binaries `occ` and `occ-pairs` for calculating
single point energies and CrystalExplorer model interaction energies
will be built.


## Usage

All following usage is a work in progress, expect significant changes
constantly for the time-being while the exact input format is decided.

### occ
By default `occ` will print out its usage options, but basic usage
given a geometry e.g. `h2o.xyz` format would be:

*note* currently occ uses the default libint2 library basis sets
and location. This will be modified in a future version to be configurable.

```
occ h2o.xyz --method b3lyp --basis 6-31g
```

### occ-pairs

Expects a toml file, and one/two `.fchk` or `.molden` files for wavefunctions.
