import { loadOCC, moleculeFromXYZ } from './dist/index.js';
const Module = await loadOCC();
const mol = await moleculeFromXYZ(`3
Water
O -0.7022  -0.0561  0.00994
H -1.0223   0.8467 -0.01149
H  0.2575   0.04212 0.00522`);
const calc = Module.XtbCalculator.fromMolecule(mol);
calc.singlePoint();
const H = calc.hessian(0.005);
console.log(`Hessian: ${H.rows()}×${H.cols()}`);
const modes = calc.vibrationalModes(0.005, true);
const freqs = modes.frequencies_cm;  // bound as Vec
const fa = [];
for (let i = 0; i < freqs.size(); i++) fa.push(freqs.get(i));
fa.sort((a, b) => a - b);
console.log(`top 3 freqs: ${fa.slice(-3).map(f => f.toFixed(1)).join(', ')} cm⁻¹`);
