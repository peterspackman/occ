import { describe, test, expect, beforeAll } from 'vitest';
import { loadOCC, createMolecule, Elements } from '../dist/index.js';

describe('DFT Bindings', () => {
  let module, molecule, basis;

  beforeAll(async () => {
    module = await loadOCC();
    
    // Create a simple H2 molecule
    const atomicNumbers = [Elements.H, Elements.H];
    const positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4 * 0.529177]]; // 1.4 Bohr = 0.74 Angstrom
    
    molecule = await createMolecule(atomicNumbers, positions);
    basis = module.AOBasis.load(molecule.atoms(), 'sto-3g');
  });

  test('DFT class should be available', () => {
    expect(module.DFT).toBeDefined();
  });

  test('should create DFT object with functional', () => {
    const dft = new module.DFT('b3lyp', basis);
    expect(dft).toBeDefined();
  });

  test('GridSettings should be available', () => {
    expect(module.GridSettings).toBeDefined();
    const gridSettings = new module.GridSettings();
    expect(gridSettings).toBeDefined();
  });

  test('should create DFT object with grid settings', () => {
    const gridSettings = new module.GridSettings();
    const dft = new module.DFT('pbe', basis, gridSettings);
    expect(dft).toBeDefined();
  });

  test('DFT should have scf method', () => {
    const dft = new module.DFT('b3lyp', basis);
    expect(dft.scf).toBeDefined();
    
    const ks = dft.scf(module.SpinorbitalKind.Restricted);
    expect(ks).toBeDefined();
  });

  test('Kohn-Sham SCF should have charge/multiplicity setting', () => {
    const dft = new module.DFT('b3lyp', basis);
    const ks = dft.scf(module.SpinorbitalKind.Restricted);
    expect(ks.setChargeMultiplicity).toBeDefined();
  });

  describe('Wavefunction Methods', () => {
    let wfn;

    beforeAll(async () => {
      // Run HF calculation to get wavefunction for testing
      const hf = new module.HartreeFock(basis);
      const scf = new module.HartreeFockSCF(hf, module.SpinorbitalKind.Restricted);
      scf.setChargeMultiplicity(0, 1);
      
      const energy = scf.run();
      expect(energy).toBeDefined();
      expect(typeof energy).toBe('number');
      
      wfn = scf.wavefunction();
      expect(wfn).toBeDefined();
    });

    test('should have homoEnergy method', () => {
      expect(wfn.homoEnergy).toBeDefined();
      const homo = wfn.homoEnergy();
      expect(typeof homo).toBe('number');
    });

    test('should have lumoEnergy method', () => {
      expect(wfn.lumoEnergy).toBeDefined();
      const lumo = wfn.lumoEnergy();
      expect(typeof lumo).toBe('number');
    });

    test('should have toJson method', () => {
      expect(wfn.toJson).toBeDefined();
      const json = wfn.toJson();
      expect(typeof json).toBe('string');
      expect(json.length).toBeGreaterThan(0);
      
      // Verify it's valid JSON
      expect(() => JSON.parse(json)).not.toThrow();
    });

    test('should have toMoldenString method', () => {
      expect(wfn.toMoldenString).toBeDefined();
      const molden = wfn.toMoldenString();
      expect(typeof molden).toBe('string');
      expect(molden.length).toBeGreaterThan(0);
    });
  });
});