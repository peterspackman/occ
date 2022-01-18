# Open Computational Chemistry (OCC)

A next generation quantum chemistry and crystallography program and library.

*Note* occ is in early development, and is undergoing substantial changes regularly -
it is not stable.

## Compilation

occ requires a compliant C++17 compiler e.g. GCC-10 or newer.

### Dependencies

occ makes use of the the following open source libraries:

- [libint2](https://github.com/evaleev/libint/releases/tag/v2.7.1)
- [libxc](http://www.tddft.org/programs/libxc/down.php?file=5.1.7/libxc-5.1.7.tar.gz)
- [Eigen3](https://eigen.tuxfamily.org/)(`eigen3-dev`)
- [boost.graph](https://www.boost.org/doc/libs/1_78_0/libs/graph/doc/index.html)(`libboost-graph-dev`)
- [zlib](https://zlib.net/) (`zlibc`)
- [gemmi](https://gemmi.readthedocs.io/)
- [gau2grid](https://github.com/dgasmith/gau2grid)
- [LBFGS++](https://lbfgspp.statr.me/)
- [spdlog](https://github.com/gabime/spdlog)
- [fmt](https://github.com/fmtlib/fmt)
- [scnlib](https://github.com/eliaskosunen/scnlib)
- [nlohmann/json](https://github.com/nlohmann/json)
- [cxxopts](https://github.com/jarro2783/cxxopts)

And for the library tests/benchmarks:

- [catch2](https://github.com/catchorg/Catch2)


### Getting the source code

First clone the repository:
```
git clone https://github.com/peterspackman/occ.git
```

### Getting dependencies

Most of the dependencies can be downloaded and compiled via [CPM](https://github.com/cpm-cmake/CPM.cmake),
but you may wish to use system installed dependencies for `libint2`, `libxc`, `zlibc`, `eigen3` and
`boost`, which will be searched for by default.

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
cmake .. -GNinja -DUSE_SYSTEM_LIBINT2=OFF -DUSE_SYSTEM_LIBXC=OFF -DUSE_SYSTEM_BOOST=OFF -DUSE_SYSTEM_ZLIB=OFF -DUSE_SYSTEM_EIGEN=OFF
```

Likewise, you can pick and choose which dependencies to download. Note that if you're compiling libint2, it might take a while...

Finally, to build the binary `occ`, running

```
ninja occ
```

will result in the binary being built.

## Usage

All following usage is a work in progress, expect significant changes
constantly for the time-being while the exact input format is decided.

### occ

By default `occ -h` will print out its usage options, but basic usage
given a geometry e.g. `h2o.xyz` format would be:

```
occ h2o.xyz b3lyp 6-31g
```

*note* currently occ uses the default libint2 library basis sets
and location. This can be configured with the `OCC_BASIS_PATH` environment variable,
or you can simply make the basis set available in your working directory.
