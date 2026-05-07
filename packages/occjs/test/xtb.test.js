import { describe, it, expect, beforeAll } from 'vitest';
import { loadOCC, moleculeFromXYZ } from '../dist/index.js';

// Reference numbers come from the C++ tests:
//   tests/xtb_native_tests.cpp "Full GFN2 SCC vs xtb (water, no dispersion)"
//   build/bin/occ tb BENZEN.cif (Γ-only)

describe('XTB GFN2 Calculator', () => {
  let Module;
  let waterMolecule;

  beforeAll(async () => {
    Module = await loadOCC();
    // Standard test water geometry, in Å (matches the helpers in
    // tests/xtb_native_tests.cpp).
    waterMolecule = await moleculeFromXYZ(`3
Water
O -0.7022  -0.0561  0.00994
H -1.0223   0.8467 -0.01149
H  0.2575   0.04212 0.00522`);
  });

  describe('Bindings present', () => {
    it('exposes XtbCalculator, XtbResult and XtbMethod', () => {
      expect(typeof Module.XtbCalculator).toBe('function');
      expect(typeof Module.XtbResult).toBe('function');
      // embind enums surface as constructor-style functions in JS.
      expect(Module.XtbMethod).toBeDefined();
      expect(Module.XtbMethod.GFN2).toBeDefined();
    });
  });

  describe('Molecular SCC (water)', () => {
    let calc;
    beforeAll(() => {
      calc = Module.XtbCalculator.fromMolecule(waterMolecule);
    });

    it('reports identity / configuration', () => {
      expect(calc.numAtoms()).toBe(3);
      expect(calc.isPeriodic()).toBe(false);
      expect(calc.methodName()).toBe('GFN2');
      expect(calc.backendName()).toBe('Native');
      expect(calc.charge).toBe(0);
      expect(calc.includeMultipoles).toBe(true);
      expect(calc.includeDispersion).toBe(true);
      const k = calc.kpoints();
      expect(k[0]).toBe(1);
      expect(k[1]).toBe(1);
      expect(k[2]).toBe(1);
    });

    it('property setters round-trip', () => {
      calc.charge = 0;
      calc.maxIterations = 100;
      calc.includeMultipoles = true;
      expect(calc.charge).toBe(0);
      expect(calc.maxIterations).toBe(100);
      expect(calc.includeMultipoles).toBe(true);
    });

    it('runs single-point and matches xtb reference', () => {
      const result = calc.singlePoint();
      expect(result.converged).toBe(true);
      // xtb full GFN2 total for water: -5.0702559 Eh; we match to ~µHa.
      expect(result.totalEnergy).toBeCloseTo(-5.0702559, 4);
      expect(result.atomicCharges.size()).toBe(3);
      expect(result.atomicCharges.get(0)).toBeLessThan(-0.3);  // O is negative
      expect(result.atomicCharges.get(1)).toBeGreaterThan(0.1);
      expect(result.atomicCharges.get(2)).toBeGreaterThan(0.1);
    });

    it('returns energy + analytical gradient', () => {
      const eg = calc.energyAndGradient(false, 1e-3);
      // The analytical gradient runs charge-only SCC for self-consistency,
      // so its energy is ~1 mHa above the multipole-on single_point total.
      // See xtb_calculator.h docstring on gradient().
      expect(eg.energy).toBeCloseTo(-5.07, 1);
      expect(eg.gradient.rows()).toBe(3);
      expect(eg.gradient.cols()).toBe(3);
    });
  });
});
