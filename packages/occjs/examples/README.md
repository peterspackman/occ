# OCC.js Examples

This directory contains examples of using the OCC.js library for various quantum chemistry calculations.

## Running the Examples

### Local Development

If you're developing locally, first build the project:

```bash
npm run build
npm run examples
```

This will start a local server and open the examples in your browser.

### Using the NPM Package

The examples can also work with the published npm package. There are several ways to use it:

#### 1. Via CDN (easiest for demos)

```html
<script type="module">
import * as OCC from 'https://unpkg.com/@peterspackman/occjs@latest/dist/index.browser.js';

await OCC.loadOCC();
// Use OCC functions
</script>
```

#### 2. With npm and a bundler

```bash
npm install @peterspackman/occjs
```

```javascript
import { loadOCC, moleculeFromXYZ, calculateDMA } from '@peterspackman/occjs';

await loadOCC();
// Use OCC functions
```

## Available Examples

### 1. Wavefunction Calculator (`wavefunction_calculator.html`)
Complete quantum chemistry calculator that runs entirely in your browser using a Web Worker for non-blocking calculations. Features:
- Upload XYZ files or paste coordinates directly
- Choose from multiple theory levels (HF, DFT with various functionals)
- Multiple basis sets (STO-3G, 3-21G, 6-31G, 6-31G(d,p), def2-SVP, def2-TZVP, cc-pVDZ, cc-pVTZ)
- Real-time calculation logging
- Matrix visualization (overlap, kinetic, nuclear attraction, Fock, density, MO coefficients)
- Orbital energy analysis (HOMO, LUMO, band gap)
- Export results as FCHK or JSON formats
- Export matrices as CSV files

**Try it online:** [https://unpkg.com/@peterspackman/occjs/examples/wavefunction_calculator.html](https://unpkg.com/@peterspackman/occjs/examples/wavefunction_calculator.html)

### 2. DMA Calculator (`dma_calculator.html`)
Interactive tool for Distributed Multipole Analysis calculations. Features:
- Drag-and-drop file upload (.fchk, .molden, .json formats)
- Configurable DMA parameters
- Visual results with downloadable punch files
- Timing information

### 3. Elastic Tensor Analyzer (`elastic.html`)
Tool for analyzing elastic tensors:
- Input elastic tensor in various formats
- Calculate elastic properties (Young's modulus, Poisson's ratio, etc.)
- 2D and 3D visualizations
- Directional property analysis

## Example Molecules

Try these simple molecules to test the Wavefunction Calculator:

### Water (H₂O)
```
3
Water molecule
O  0.000000  0.000000  0.000000
H  0.757000  0.586000  0.000000
H -0.757000  0.586000  0.000000
```

### Methane (CH₄)
```
5
Methane molecule
C  0.000000  0.000000  0.000000
H  1.089000  1.089000  1.089000
H -1.089000 -1.089000  1.089000
H -1.089000  1.089000 -1.089000
H  1.089000 -1.089000 -1.089000
```

### Hydrogen (H₂)
```
2
Hydrogen molecule
H  0.000000  0.000000  0.699199
H  0.000000  0.000000 -0.699199
```

## File Format Support

### Wavefunction Files (for DMA Calculator)
- **FCHK** (.fchk) - Formatted checkpoint files from Gaussian
- **Molden** (.molden) - Molden format files
- **JSON** (.json) - OCC's native JSON wavefunction format

### XYZ Files (for Wavefunction Calculator)
- Standard XYZ coordinate files with atom count, title, and coordinates

### Limitations
- Maximum 250 basis functions for web demos (to ensure reasonable performance)

## Development Notes

The examples are designed to work in two modes:

1. **Local Development Mode**: Uses relative imports from `../dist/`
2. **NPM Package Mode**: Uses the published package from npm/CDN

The `dma_calculator_npm.html` example shows how to create applications that work transparently in both modes.

## Browser Compatibility

- Modern browsers with ES6 module support
- WebAssembly support required
- Tested on Chrome, Firefox, Safari, and Edge
