/**
 * Tests for crystal growth and openCOSMO-RS solvation
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { loadOCC, moleculeFromXYZ } from '../dist/index.js';
import {
  calculateCrystalGrowth,
  cosmoRsSolvation,
  availableCosmoRsSolvents
} from '../dist/crystal-growth.js';

const waterXYZ = `3
water
O    -0.7021961  -0.0560603   0.0099423
H    -1.0221932   0.8467758  -0.0114887
H     0.2575211   0.0421215   0.0052190
`;

describe('COSMO-RS solvation', () => {
  beforeAll(async () => {
    await loadOCC();
  });

  it('lists cached solvent ensembles, of which occ ships none', async () => {
    const solvents = await availableCosmoRsSolvents();
    expect(Array.isArray(solvents)).toBe(true);
    // Sorted, so a caller can present them directly. Empty unless the caller
    // has put ensembles on the search path themselves.
    expect([...solvents].sort()).toEqual(solvents);
  });

  it('assembles a solvation free energy from its terms', async () => {
    const mol = await moleculeFromXYZ(waterXYZ);
    // No ensemble ships, so the solvent is given as a geometry and its
    // conductor cavity computed alongside the solute's.
    const solventMol = await moleculeFromXYZ(waterXYZ);
    const result = await cosmoRsSolvation(mol, solventMol,
                                          { liquidVolume: 30.01 });

    expect(result.cavityArea).toBeGreaterThan(30);
    expect(result.cavityVolume).toBeGreaterThan(15);
    expect(result.numRings).toBe(0);

    const e = result.energy;
    const summed = e.dielectric + e.residual + e.combinatorial + e.vdw +
                   e.ring + e.referenceState + e.eta;
    expect(summed).toBeCloseTo(e.total, 6);
    expect(result.total).toBeCloseTo(e.total, 6);

    // Water in water is bound; the terms it must have are the conductor
    // stabilisation and the van der Waals surface term.
    expect(e.dielectric).toBeLessThan(0);
    expect(e.vdw).toBeLessThan(0);
    expect(e.total).toBeLessThan(0);
  }, 120000);

  it('rejects unknown settings rather than ignoring them', async () => {
    const mol = await moleculeFromXYZ(waterXYZ);
    await expect(cosmoRsSolvation(mol, mol, { liquidVolumes: 30 }))
      .rejects.toThrow(/unknown option 'liquidVolumes'/);
  });
});

describe('crystal growth', () => {
  beforeAll(async () => {
    await loadOCC();
  });

  it('requires a structure', async () => {
    await expect(calculateCrystalGrowth({}))
      .rejects.toThrow(/structure is required/);
  });

  it('rejects both cif and crystalFilename', async () => {
    await expect(calculateCrystalGrowth({ cif: 'x', crystalFilename: 'y' }))
      .rejects.toThrow(/not both/);
  });

  it('rejects unknown options rather than ignoring them', async () => {
    await expect(calculateCrystalGrowth({ cif: 'x', modelNmae: 'ce-b3lyp' }))
      .rejects.toThrow(/unknown option 'modelNmae'/);
  });

  it('exposes the solvation model on the native config', async () => {
    const Module = await loadOCC();
    const config = new Module.CrystalGrowthConfig();
    try {
      expect(config.solvationModel).toBe('smd');
      expect(config.temperature).toBeCloseTo(298.15, 6);
      config.solvationModel = 'cosmo-rs';
      expect(config.solvationModel).toBe('cosmo-rs');
    } finally {
      config.delete();
    }
  });
});
