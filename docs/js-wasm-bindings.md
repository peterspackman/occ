# JavaScript/WebAssembly Bindings

OCC provides JavaScript/WebAssembly bindings for quantum chemistry calculations in web browsers, Node.js, and other JavaScript environments. The bindings are built using Emscripten and mirror the Python API structure.

## Installation

### Prerequisites

- Emscripten SDK installed and activated
- CMake 3.16+
- Ninja build system (recommended)

### Build Instructions

1. Build WASM bindings using the provided script:
   ```bash
   ./scripts/build_wasm.sh
   ```

2. Or manually configure and build:
   ```bash
   emcmake cmake . -Bwasm -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -DENABLE_JS_BINDINGS=ON -GNinja
   cmake --build wasm --target occjs
   ```

This generates:
- `wasm/src/occjs.js` - JavaScript module
- `wasm/src/occjs.wasm` - WebAssembly binary

## Usage

### Browser (ES6 Modules)

```html
<!DOCTYPE html>
<html>
<head>
    <script type="module">
        import createOccModule from './occjs.js';
        
        createOccModule().then(Module => {
            // Create a water molecule
            const positions = new Module.Mat3N(3, 3);
            positions.set(0, 0, 0.0);
            positions.set(1, 0, 0.0);  
            positions.set(2, 0, 0.757);
            positions.set(0, 1, 0.0);
            positions.set(1, 1, 0.0);
            positions.set(2, 1, -0.757);
            positions.set(0, 2, 0.0);
            positions.set(1, 2, 1.511);
            positions.set(2, 2, 0.0);
            
            const atomicNumbers = new Module.IVec([8, 1, 1]);
            const water = new Module.Molecule(atomicNumbers, positions);
            
            console.log("Molecular mass:", water.molarMass());
            console.log("Center of mass:", water.centerOfMass());
        });
    </script>
</head>
<body>
    <h1>OCC WebAssembly Demo</h1>
</body>
</html>
```

### Node.js

```javascript
const createOccModule = require('./occjs.js');

createOccModule().then(Module => {
    // Set up logging
    Module.setLogLevel(Module.LogLevel.INFO);
    
    // Load a basis set
    const basis = Module.AOBasis.load("sto-3g", ["H", "H"]);
    console.log("Basis functions:", basis.nbf());
    
    // Create H2 molecule
    const positions = new Module.Mat3N(2, 3);
    positions.set(0, 0, 0.0);
    positions.set(1, 0, 0.0);
    positions.set(2, 0, 0.0);
    positions.set(0, 1, 0.0);
    positions.set(1, 1, 0.0);
    positions.set(2, 1, 1.4);
    
    const atomicNumbers = new Module.IVec([1, 1]);
    const h2 = new Module.Molecule(atomicNumbers, positions);
    
    // Perform SCF calculation
    const hf = new Module.HartreeFock(basis);
    const scf = hf.scf(Module.SpinorbitalKind.Restricted);
    const energy = scf.run();
    
    console.log("SCF Energy:", energy);
});
```

## API Reference

### Core Module

#### Element
- `constructor(symbol: string)` - Create element from symbol
- `constructor(atomicNumber: number)` - Create element from atomic number
- `symbol: string` - Element symbol
- `mass: number` - Atomic mass
- `name: string` - Element name
- `atomicNumber: number` - Atomic number

#### Atom
- `constructor(atomicNumber: number, x: number, y: number, z: number)`
- `atomicNumber: number` - Atomic number
- `x, y, z: number` - Cartesian coordinates
- `getPosition(): Vec3` - Get position vector
- `setPosition(pos: Vec3): void` - Set position

#### Molecule
- `constructor(atomicNumbers: IVec, positions: Mat3N)`
- `size(): number` - Number of atoms
- `name: string` - Molecule name
- `centerOfMass(): Vec3` - Center of mass
- `centroid(): Vec3` - Geometric centroid
- `molarMass(): number` - Molecular mass
- `translate(translation: Vec3): void` - Translate molecule
- `rotate(rotation: Mat3, origin?: Origin): void` - Rotate molecule
- Static methods:
  - `fromXyzFile(filename: string): Molecule`
  - `fromXyzString(contents: string): Molecule`

#### PointCharge
- `constructor(charge: number, x: number, y: number, z: number)`
- `charge: number` - Point charge value
- `getPosition(): Vec3` - Position vector

### QM Module

#### AOBasis
- Static `load(name: string, atoms: string[]): AOBasis` - Load basis set
- `nbf(): number` - Number of basis functions
- `shells(): Shell[]` - Basis function shells
- `atoms(): Atom[]` - Atoms in basis

#### MolecularOrbitals
- `kind: SpinorbitalKind` - Restricted/unrestricted
- `numAlpha: number` - Number of alpha electrons
- `numBeta: number` - Number of beta electrons
- `orbitalEnergies: Vec` - Orbital energies
- `densityMatrix: Mat` - Density matrix

#### Wavefunction
- `molecularOrbitals: MolecularOrbitals` - MO information
- `atoms: Atom[]` - Molecular geometry
- `basis: AOBasis` - Basis set
- `charge(): number` - Molecular charge
- `multiplicity(): number` - Spin multiplicity
- `mullikenCharges(): Vec` - Mulliken partial charges
- `electronDensity(points: Mat3N, derivatives?: number): Mat` - Electron density
- Static methods:
  - `load(filename: string): Wavefunction`
  - `fromFchk(filename: string): Wavefunction`

#### HartreeFock
- `constructor(basis: AOBasis)`
- `overlapMatrix(): Mat` - Overlap integrals
- `kineticMatrix(): Mat` - Kinetic energy integrals
- `nuclearAttractionMatrix(): Mat` - Nuclear attraction integrals
- `coulombMatrix(mo: MolecularOrbitals): Mat` - Coulomb matrix
- `nuclearRepulsion(): number` - Nuclear repulsion energy

#### SCF Methods
- `HF(hf: HartreeFock, kind?: SpinorbitalKind)` - SCF procedure
- `run(): number` - Perform SCF calculation
- `wavefunction(): Wavefunction` - Get converged wavefunction

### Enums

#### SpinorbitalKind
- `Restricted` - Closed shell
- `Unrestricted` - Open shell
- `General` - General case

#### LogLevel
- `TRACE`, `DEBUG`, `INFO`, `WARN`, `ERROR`, `CRITICAL`, `OFF`

## Utilities

```javascript
// Set number of threads (if threading enabled)
Module.setNumThreads(4);

// Configure logging
Module.setLogLevel(Module.LogLevel.INFO);
Module.setLogFile("occ.log");

// Set data directory for basis sets
Module.setDataDirectory("/path/to/data");
```

## Performance Considerations

1. **Memory Management**: Objects are automatically managed by Emscripten
2. **Threading**: Limited threading support in browsers
3. **File I/O**: Use Emscripten's virtual filesystem for file operations
4. **Matrix Operations**: Large matrices may impact performance

## Example Calculations

### Simple SCF Calculation

```javascript
createOccModule().then(Module => {
    // H2 molecule
    const positions = new Module.Mat3N(2, 3);
    // Set coordinates...
    const h2 = new Module.Molecule(new Module.IVec([1, 1]), positions);
    
    // Load basis
    const basis = Module.AOBasis.load("sto-3g", h2.atoms());
    
    // SCF calculation
    const hf = new Module.HartreeFock(basis);
    const scf = new Module.HF(hf);
    const energy = scf.run();
    
    console.log("Total energy:", energy);
});
```

### Density Calculation

```javascript
// Get converged wavefunction
const wfn = scf.wavefunction();

// Create grid points
const points = new Module.Mat3N(1000, 3);
// Fill grid points...

// Calculate electron density
const density = wfn.electronDensity(points);
console.log("Density at points:", density);
```

## Error Handling

```javascript
try {
    const molecule = Module.Molecule.fromXyzFile("nonexistent.xyz");
} catch (error) {
    console.error("Failed to load molecule:", error.message);
}
```

## Browser Compatibility

- Chrome 57+ (WebAssembly support)
- Firefox 52+
- Safari 11+
- Edge 16+

For older browsers, consider using a WebAssembly polyfill.