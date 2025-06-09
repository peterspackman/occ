# JavaScript/WASM Examples

Interactive examples demonstrating OCC's quantum chemistry capabilities in web browsers and Node.js environments.

## Examples

### Interactive Demo (`index.html`)

**Features:**
- Real-time XYZ coordinate input
- Molecular property calculations
- Point group symmetry analysis
- Partial charge methods (EEM/EEQ)
- Example molecules (H₂O, H₂, CH₄, NH₃)

### Simple Molecule Demo (`simple_molecule.html`)

**Features:**
- Basic molecular operations
- Coordinate transformations
- Property calculations

### SCF Calculation (`scf_calculation.js`)

**Features:**
- Hartree-Fock SCF calculations
- Basis set handling
- Wavefunction analysis
- Node.js execution

## Quick Start

1. **Build WASM bindings:**
   ```bash
   ./scripts/build_wasm.sh
   ```

2. **Copy files to examples directory:**
   ```bash
   cp wasm/src/occjs.* examples/javascript/
   ```

3. **Start the development server:**
   ```bash
   cd examples/javascript
   python3 serve.py
   ```

4. **Open your browser:**
   ```
   http://localhost:8000
   ```

## Manual Setup

### Build WASM Bindings
```bash
emcmake cmake . -Bwasm -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -DENABLE_JS_BINDINGS=ON -GNinja
cmake --build wasm --target occjs
```

### Run Browser Examples
```bash
# Using Python 3
python3 -m http.server 8000

# Using Node.js
npx http-server
```

### Run Node.js Examples
```bash
node scf_calculation.js
```

## Expected Output

### SCF Calculation Example
```
Loading OCC WASM module...
✓ OCC module loaded successfully

=== Creating H2 Molecule ===
Molecule: Hydrogen molecule
Number of atoms: 2
Molar mass: 2.0159 g/mol
Center of mass: [0.0000, 0.0000, 0.7000] Bohr

=== Loading Basis Set ===
Basis set: sto-3g
Number of shells: 2
Number of basis functions: 2

=== Setting up Hartree-Fock Calculation ===
Nuclear repulsion energy: 0.71428571 Hartree

Calculating one-electron integrals...
Overlap matrix size: 2 x 2
Kinetic energy matrix size: 2 x 2
Nuclear attraction matrix size: 2 x 2

=== SCF Calculation ===
Energy convergence threshold: 1e-8
Commutator convergence threshold: 1e-6
Starting SCF iterations...

=== Results ===
SCF converged!
Total energy: -1.11675930 Hartree
Electronic energy: -1.83104501 Hartree
Nuclear repulsion: 0.71428571 Hartree
Calculation time: 45 ms

=== Wavefunction Analysis ===
Spinorbital kind: Restricted
Number of alpha electrons: 1
Number of beta electrons: 1
Number of basis functions: 2

Orbital energies (Hartree):
  Orbital 1: -0.578395 (occupied)
  Orbital 2: 0.670898 (virtual)

Mulliken partial charges:
  Atom 1 (H): 0.000000 e
  Atom 2 (H): 0.000000 e
Total charge: 0.000000 e

✓ SCF calculation completed successfully!
```

## API Overview

### Core Classes
- `Element` - Chemical elements with properties
- `Atom` - Individual atoms with coordinates
- `Molecule` - Collections of atoms with molecular properties
- `PointCharge` - Point charges for electrostatic calculations

### QM Classes
- `AOBasis` - Atomic orbital basis sets
- `MolecularOrbitals` - Molecular orbital information
- `Wavefunction` - Complete quantum mechanical state
- `HartreeFock` - Hartree-Fock method implementation
- `IntegralEngine` - Quantum mechanical integral calculations

### Utility Functions
- `eemPartialCharges()` - EEM partial charge calculation
- `eeqPartialCharges()` - EEQ partial charge calculation
- `setLogLevel()` - Configure logging
- `setNumThreads()` - Set number of threads (where supported)

## Performance Notes

1. **Memory**: WASM has a memory limit (default 1GB), large calculations may need tuning
2. **Threading**: Limited in browser environments, full threading in Node.js
3. **File I/O**: Use Emscripten's virtual filesystem for file operations
4. **Precision**: Double precision throughout, same as native C++ code

## Troubleshooting

### Common Issues

1. **CORS errors in browser**: Serve files from a web server, don't open directly
2. **Module loading fails**: Ensure both `.js` and `.wasm` files are present
3. **Out of memory**: Increase memory limit in Emscripten build flags
4. **Performance**: Disable debug assertions for production builds

### Debug Mode

To enable more verbose output:
```javascript
Module.setLogLevel(Module.LogLevel.DEBUG);
```

## Further Examples

For more advanced examples, see:
- The Python examples (similar API structure)
- The C++ test files in `/tests/`
- The main OCC documentation