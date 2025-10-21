---
title: 'Open Computational Chemistry (OCC) - A portable software library and program for quantum chemistry and crystallography'
tags:
  - C++
  - Python
  - WebAssembly
  - quantum chemistry
  - crystallography
  - density functional theory
  - crystal growth
authors:
  - name: Peter R. Spackman
    orcid: 0000-0002-6532-8571
    corresponding: true
    affiliation: 1
affiliations:
 - name: School of Molecular and Life Sciences, Curtin University, Perth, WA 6845, Australia
   index: 1
date: 21 October 2025
bibliography: paper.bib
---

# Summary

Open Computational Chemistry (OCC) is a modern software library designed for
calculating the properties and electronic structure and interactions of
molecules and molecular crystals. The software serves a dual purpose: it
provides an open, accessible platform for researchers developing new
computational methods, while also being production-ready and fast enough for
relatively large-scale calculations on personal computers and high-performance
computing systems alike.

Many traditional computational chemistry programs require complex installation
procedures or rely on large dependencies can only run on specific operating
systems. OCC is designed as a priority to be portable: it runs on Windows,
macOS, and Linux, provides interfaces for C++, Python, and JavaScript, and with
the capability to even run entirely within web browsers through WebAssembly
(WASM), making computational chemistry accessible without specialized hardware
or software installation. OCC is already widely used as the primary
computational backend of CrystalExplorer[@Spackman2021], a widely-used
graphical program for crystal structure analysis, providing capabilities such
as automatic calculation of interaction energies between molecules in crystals,
accurate determination of total crystal energies with automatic convergence
checking, and evaluation of electron density distributions. 

# Statement of need

The landscape of computational chemistry software is dominated by established
packages that, while powerful, often suffer from legacy codebases, limited
portability, and barriers to modification. Most existing quantum chemistry
software requires complex installation procedures and system-level
dependencies, limiting accessibility for educational purposes and rapid
prototyping. Furthermore, the integration between quantum chemistry and
crystallography remains fragmented across different specialized tools,
requiring researchers to chain together multiple incompatible programs for
multi-scale workflows.

OCC addresses these challenges as an open, portable, and modifiable platform
for computational chemistry and crystallography. The software serves dual
purposes: as a computational backend for production applications like
CrystalExplorer, a widely-adopted graphical tool for crystal structure
analysis, and as a research platform for developing new computational methods.
This distinguishes OCC from purely academic codes (often limited in scope,
modifiability or performance) and purely commercial alternatives (which lack
transparency and modifiability).

Key features enabling this versatility include:

**Portability and Accessibility**: OCC is, to our knowledge, the first
full-featured quantum chemistry package that can run entirely in web browsers
via WASM, enabling interactive calculations without installation or specialized hardware,
on computers, tablets, phones or any device with a browser supporting WASM.
Native builds support Windows, macOS, and
Linux with comprehensive continuous integration testing, while language
bindings for C++, Python, and JavaScript facilitate integration into diverse
workflows.

**Unified Framework**: By integrating electronic structure methods with
crystallographic analysis in a single modern C++ codebase, OCC eliminates the
friction of multi-scale workflows. Unique capabilities include crystal growth
free energy predictions bridging molecular-level calculations and macroscopic
properties, automatic lattice energy convergence, and pairwise interaction
energy calculations used in production by CrystalExplorer.

**Open and Modifiable**: The codebase follows contemporary software development
best practices including comprehensive continuous integration testing across
all platforms, automated releases for Python packages via PyPI, and Javascript
via NPM and well-documented APIs with dependency management via CPM.cmake. This
enables researchers to rapidly develop and test new methods. Unlike monolithic
legacy codes, OCC's modular architecture facilitates extension and modification
for specific research needs.

# Implementation and Features

OCC implements Hartree-Fock and Density Functional Theory (DFT) with support
for LDA, GGA, and meta-GGA functionals via libxc [@Lehtola2018], density
fitting (RI-JK) methods for coulomb and exchange interactions, implicit
solvation via COSMO[@Klamt1993] and SMD[@Marenich2009], and dispersion
corrections (XDM[@Johnson2006;@Becke2007;@OterodelaRoza2012],
DFTD4[@Caldeweyher2017;@Caldeweyher2019;@Caldeweyher2020]). For
crystallography, OCC provides CIF file processing via gemmi [@Wojdyr2022], fast
periodic bond detection, symmetry-unique molecule generation, CrystalExplorer
model energies [@Mackenzie2017; @Spackman2023CE], and automatic pair-based
lattice energy summation for neutral molecular crystals with symmetry.

Unique capabilities include crystal growth free energy
predictions[@Spackman2023CC] combining lattice energies, interaction energy
decomposition, and vibrational/configurational entropy contributions. The
distributed multipole analysis (DMA)[@Stone2005] implementation provides multipole
expansions up to hexadecapole level with GDMA-compatible output. WebAssembly
compilation via Emscripten [@Zakai2011] enables full-featured quantum
chemistry calculations directly in web browsers, supporting interactive
educational tools and client-side computational workflows without server
infrastructure.

OCC employs a modular C++ architecture using Eigen3 [@eigen] for linear
algebra, libcint [@Sun2015] for integral evaluation, and libecpint [@Shaw2017;@Shaw2021]
for effective core potentials.

# Performance

Native builds provide performance comparable to established quantum chemistry
packages through optimized integral evaluation, SIMD instructions, and parallel
execution, while WebAssembly builds offer unprecedented accessibility with
acceptable performance overhead for educational and prototyping use cases.

# Availability and Documentation

OCC is available as open-source software under the GNU General Public License
v3 at https://github.com/peterspackman/occ. Pre-built Python packages are
available via PyPI ([occpy](https://pypi.org/projects/occpy)). WebAssembly
builds can be integrated via
[npm](https://www.npmjs.com/package/@peterspackman/occjs) or used directly in
browsers. Comprehensive continuous integration testing ensures reliability
across Windows, macOS, and Linux platforms for all language bindings (C++,
Python, JavaScript/WASM). Documentation at
[https://getocc.xyz](https://getocc.xyz) includes interactive tutorials, where
the code runs in the browser, complete C++ API references, and examples.

# Conclusions

OCC provides a modern, accessible platform for quantum chemistry and
crystallography that serves both as a research tool for method development and
as a production-ready library powering applications like CrystalExplorer. Its
unique combination of traditional electronic structure methods,
crystallographic analysis, and WebAssembly support addresses key accessibility
and integration challenges in computational chemistry. Future development will
focus on expanding functional support, implementing excited state methods, and
adding GPU acceleration.

# Acknowledgements

OCC would not exist if it weren't for Tonto[@tonto] its predecessor library and
program at the backend of CrystalExplorer.

OCC builds upon a foundation of high-quality open-source libraries, and we
gratefully acknowledge the developers and maintainers of these projects:

**Core computational libraries**: Eigen3 [@eigen] for linear algebra
operations, libcint [@Sun2015] for Gaussian integral evaluation, libxc
[@Lehtola2018] for exchange-correlation functionals,
libecpint [@Shaw2021] for effective core potential
integrals, and gau2grid [@gau2grid] for efficient grid evaluation of gaussians.

**Crystallographic and optimization tools**: gemmi [@Wojdyr2022] for CIF file
handling and crystallographic operations, and LBFGS++ [@lbfgspp] for
optimization algorithms. The molecular geometry optimization implementation was
significantly informed by pyberny [@pyberny].

**Infrastructure and utilities**: fmt [@fmt] for string formatting, spdlog
[@spdlog] for logging, oneTBB [@tbb] for parallelization, CLI11 [@cli11] for
command-line parsing, nlohmann/json [@nlohmann_json] for JSON handling, scnlib
[@scnlib] for string parsing, and unordered_dense [@unordered_dense] for
efficient hash containers.

**Development tools**: nanobind [@nanobind] for Python bindings, Emscripten
[@Zakai2011] for WebAssembly compilation, and CPM.cmake [@cpm] for
dependency management.

# References
