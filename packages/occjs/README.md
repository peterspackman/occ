# @occ/core

JavaScript/WebAssembly bindings for OCC - a quantum chemistry and crystallography library.

## Installation

```bash
npm install @occ/core
```

## Quick Start

```javascript
const { loadOCC, createMolecule, Elements } = require('@occ/core');

// Initialize the WASM module
const Module = await loadOCC();

// Create a water molecule
const water = await createMolecule(
  [Elements.O, Elements.H, Elements.H],
  [
    [0.0, 0.0, 0.0],
    [0.757, 0.586, 0.0],
    [-0.757, 0.586, 0.0]
  ]
);

console.log(`Molecule: ${water.name}`);
console.log(`Atoms: ${water.size()}`);
console.log(`Molar mass: ${water.molarMass()} g/mol`);
```

## Features

- **Molecular Structure Manipulation**: Create and manipulate molecular structures
- **Quantum Chemistry Calculations**: Hartree-Fock, DFT, and post-HF methods
- **Crystallography**: Space groups, symmetry operations, and crystal structure analysis
- **Partial Charges**: EEM and EEQ charge models
- **Geometry Analysis**: Point groups, molecular properties, and transformations
- **Isosurface Generation**: Electron density and promolecule density visualization
- **Custom Logging**: Flexible logging system with callbacks and buffering

## Documentation

For complete API documentation, examples, and usage guides, see the [main OCC repository](https://github.com/peterspackman/occ#readme).

## Links

- [GitHub Repository](https://github.com/peterspackman/occ)
- [Documentation](https://github.com/peterspackman/occ#readme)
- [Issue Tracker](https://github.com/peterspackman/occ/issues)