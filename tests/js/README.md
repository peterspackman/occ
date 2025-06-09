# JavaScript/WASM Tests

This directory contains the JavaScript test suite for the OCC WebAssembly bindings, validating core molecular chemistry operations and quantum mechanical calculations.

## Test Structure

### Test Framework (`test_framework.js`)
- Lightweight testing framework with assertion functions
- Performance timing utilities
- Memory usage monitoring (Node.js)
- Test suite organization and reporting

### Core Tests (`test_core.js`)
Tests for the core molecular chemistry functionality:
- Element properties and creation
- Atom manipulation
- Molecule construction and properties
- Point charge calculations
- Molecular transformations (rotation, translation, centering)
- Dimer analysis
- Point group symmetry determination
- Partial charge calculations (EEM/EEQ methods)

### QM Tests (`test_qm.js`)
Tests for quantum mechanical functionality:
- Basis set loading and properties
- Molecular orbital management
- Hartree-Fock integral calculations
- SCF convergence procedures
- Complete H₂ molecule SCF calculations
- Wavefunction analysis
- Integral engine functionality
- Operator enumerations

## Running Tests

### Automated (Recommended)
```bash
# From the project root
./scripts/test_javascript.sh
```

### Manual
```bash
# Build WASM bindings first
emcmake cmake . -Bwasm -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=OFF -DENABLE_JS_BINDINGS=ON -GNinja
cmake --build wasm --target occjs

# Copy files to test directory
cp wasm/src/occjs.* tests/js/

# Run tests
cd tests/js
node run_tests.js
```

## Prerequisites

1. **Node.js** (v14+ recommended)
   - Download from: https://nodejs.org/
   - Or install via package manager

2. **Emscripten SDK**
   - Install from: https://emscripten.org/docs/getting_started/downloads.html
   - Must be activated in current shell

3. **Build tools**
   - CMake 3.16+
   - Ninja build system (recommended)

## Test Output

### Successful Run
```
OCC JavaScript/WASM Test Suite

Loading OCC module from: ../../wasm/src/occjs.js
✓ OCC WASM module loaded successfully

=== Core Module Tests ===
✓ Element creation and properties
✓ Atom creation and manipulation
✓ PointCharge functionality
✓ H2 molecule creation and properties
✓ Water molecule creation and properties
✓ Molecular transformations
✓ Dimer creation and analysis
✓ Point group analysis
✓ Partial charge calculations

Results: 9/9 passed

=== QM Module Tests ===
✓ AOBasis loading and properties
✓ MolecularOrbitals creation and properties
✓ HartreeFock object creation and basic integrals
✓ SCFConvergenceSettings
✓ Complete H2 SCF calculation
✓ IntegralEngine functionality
✓ SpinorbitalKind enumeration
✓ Operator enumeration

Results: 8/8 passed

==================================================
FINAL RESULTS
==================================================
Total tests: 17
Passed: 17
Failed: 0
Success rate: 100.0%
Total time: 1234 ms
Final memory: RSS: 45MB, Heap: 23/28MB, External: 12MB

🎉 All tests passed!
```

### Failed Test Example
```
=== Core Module Tests ===
✓ Element creation and properties
✗ Atom creation and manipulation: expected 1, got 2

Results: 1/2 passed

Failures:
  - Atom creation and manipulation: expected 1, got 2
```

## Test Details

### Core Module Coverage
- **Elements**: Symbol/atomic number lookup, properties
- **Atoms**: Coordinate manipulation, position vectors
- **Molecules**: Geometry, mass calculations, transformations
- **Point Charges**: Electrostatic point charges
- **Symmetry**: Point group determination
- **Charges**: EEM/EEQ partial charge methods

### QM Module Coverage
- **Basis Sets**: STO-3G loading and validation
- **Integrals**: Overlap, kinetic, nuclear attraction
- **SCF**: Complete restricted Hartree-Fock calculation
- **Orbitals**: Energy analysis, occupation
- **Convergence**: Energy and commutator criteria
- **Engines**: Integral computation infrastructure

### Performance Benchmarks
Typical performance on modern hardware:
- Module loading: ~100-200ms
- H₂/STO-3G SCF: ~10-50ms
- All tests: ~1-3 seconds

### Memory Usage
- Initial overhead: ~20-30MB
- Peak usage: ~40-60MB
- WASM binary: ~2-5MB

## Debugging

### Enable Debug Logging
```javascript
// In test files, change log level
Module.setLogLevel(Module.LogLevel.DEBUG);
```

### Common Issues

1. **Module not found**
   - Ensure WASM bindings are built
   - Check file paths in `run_tests.js`

2. **Emscripten errors**
   - Verify Emscripten installation
   - Check system dependencies

3. **Memory errors**
   - Increase WASM memory limit
   - Check for memory leaks in C++ code

4. **Numerical failures**
   - Verify tolerance settings
   - Check reference values

### Adding New Tests

1. Create test functions following existing patterns
2. Use assertion functions from `test_framework.js`
3. Add to appropriate test file (`test_core.js` or `test_qm.js`)
4. Update this README if adding new modules

Example test:
```javascript
suite.test('New functionality test', () => {
    const result = Module.newFunction();
    test.assertEqual(result, expectedValue, 'New function result');
    test.assertTrue(result > 0, 'Result should be positive');
});
```

## Continuous Integration

For CI/CD pipelines, use:
```bash
# Non-interactive mode
./scripts/test_javascript.sh --ci
```

Exit codes:
- 0: All tests passed
- 1: Test failures or build errors

## Performance Profiling

The test framework includes basic performance monitoring:
- Individual test timing
- Memory usage tracking
- Overall execution time

For detailed profiling, use Node.js profiling tools or browser DevTools.