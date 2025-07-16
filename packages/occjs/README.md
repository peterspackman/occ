# @occ/core

[![JavaScript Tests](https://github.com/peterspackman/occ/actions/workflows/build_nodejs.yml/badge.svg)](https://github.com/peterspackman/occ/actions/workflows/build_nodejs.yml)
[![npm version](https://badge.fury.io/js/@occ%2Fcore.svg)](https://badge.fury.io/js/@occ%2Fcore)

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

## Development

### Building from Source

1. **Prerequisites**: Ensure you have Emscripten installed
2. **Build WASM module**: `npm run build:wasm`
3. **Build JavaScript wrapper**: `npm run build:wrapper`
4. **Run tests**: `npm test`

### Testing

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage
npm run test:coverage

# Run linting
npm run lint

# Run type checking
npm run typecheck
```

### Available Test Suites

- **Core functionality**: Basic module loading and error handling
- **Logging**: Log level management and message handling  
- **Molecular structure**: Molecule creation and property access
- **Quantum chemistry**: Hartree-Fock, DFT calculations, and property analysis
- **Crystallography**: Space groups, symmetry operations, unit cells
- **Isosurfaces**: Surface generation and cube file export
- **Basis sets**: Built-in and custom JSON basis set loading

## Documentation

For complete API documentation, examples, and usage guides, see the [main OCC repository](https://github.com/peterspackman/occ#readme).

## Links

- [GitHub Repository](https://github.com/peterspackman/occ)
- [Documentation](https://github.com/peterspackman/occ#readme)
- [Issue Tracker](https://github.com/peterspackman/occ/issues)