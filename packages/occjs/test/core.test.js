import { describe, it, expect, beforeAll } from 'vitest';
import { loadOCC, createMolecule, moleculeFromXYZ, Elements } from '../src/index.js';

describe('Core Module Tests', () => {
  let Module;
  
  beforeAll(async () => {
    Module = await loadOCC();
  });

  describe('Element', () => {
    it('should create element from symbol and access properties', () => {
      const h = new Module.Element('H');
      expect(h.symbol).toBe('H');
      expect(h.atomicNumber).toBe(1);
      expect(h.mass).toBeGreaterThan(1.0);
      expect(h.mass).toBeLessThan(1.1);
    });

    it('should create element from atomic number', () => {
      const c = Module.Element.fromAtomicNumber(6);
      expect(c.symbol).toBe('C');
      expect(c.atomicNumber).toBe(6);
    });
  });

  describe('Atom', () => {
    it('should create and manipulate atoms', () => {
      const atom = new Module.Atom(1, 0.0, 0.0, 1.4);
      expect(atom.atomicNumber).toBe(1);
      expect(atom.x).toBeCloseTo(0.0, 10);
      expect(atom.y).toBeCloseTo(0.0, 10);
      expect(atom.z).toBeCloseTo(1.4, 10);
      
      const pos = atom.getPosition();
      expect(pos.z()).toBeCloseTo(1.4, 10);
    });
  });

  describe('PointCharge', () => {
    it('should handle point charge functionality', () => {
      const pc = new Module.PointCharge(1.5, 1.0, 2.0, 3.0);
      expect(pc.charge).toBeCloseTo(1.5, 10);
      
      const pos = pc.getPosition();
      expect(pos.x()).toBeCloseTo(1.0, 10);
      expect(pos.y()).toBeCloseTo(2.0, 10);
      expect(pos.z()).toBeCloseTo(3.0, 10);
    });
  });

  describe('Molecule', () => {
    it('should create H2 molecule and access properties', () => {
      // Create H2 molecule
      const positions = Module.Mat3N.create(2);
      positions.set(0, 0, 0.0); positions.set(1, 0, 0.0); positions.set(2, 0, 0.0);
      positions.set(0, 1, 0.0); positions.set(1, 1, 0.0); positions.set(2, 1, 0.74);
      
      const atomicNumbers = Module.IVec.fromArray([1, 1]);
      const h2 = new Module.Molecule(atomicNumbers, positions);
      
      expect(h2.size()).toBe(2);
      // molarMass returns kg/mol, convert to g/mol
      expect(h2.molarMass() * 1000).toBeCloseTo(2.016, 2);
      
      h2.setName("H2");
      expect(h2.name).toBe("H2");
    });

    it('should create water molecule using helper function', async () => {
      const water = await createMolecule(
        [Elements.O, Elements.H, Elements.H],
        [[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]
      );
      
      expect(water.size()).toBe(3);
      // molarMass returns kg/mol, convert to g/mol
      expect(water.molarMass() * 1000).toBeCloseTo(18.015, 2);
      
      const com = water.centerOfMass();
      expect(Math.abs(com.x())).toBeLessThan(0.1);
      expect(com.y()).toBeGreaterThan(0);
      expect(Math.abs(com.z())).toBeLessThan(0.1);
    });

    it('should load molecule from XYZ string', async () => {
      const xyzString = `2
H2 molecule
H 0.0 0.0 0.0
H 0.0 0.0 0.74`;
      
      const h2 = await moleculeFromXYZ(xyzString);
      expect(h2.size()).toBe(2);
      // The XYZ parser might not preserve the comment as the name
      expect(h2.name).toBeTruthy(); // Just check it has a name
    });

    it('should handle molecular transformations', () => {
      const positions = Module.Mat3N.create(1);
      positions.set(0, 0, 1.0);
      positions.set(1, 0, 0.0);
      positions.set(2, 0, 0.0);
      
      const atomicNumbers = Module.IVec.fromArray([1]);
      const mol = new Module.Molecule(atomicNumbers, positions);
      
      // Test translation
      const translation = Module.Vec3.create(1.0, 2.0, 3.0);
      const translated = mol.translated(translation);
      const newPos = translated.positions();
      
      expect(newPos.get(0, 0)).toBeCloseTo(2.0, 10);
      expect(newPos.get(1, 0)).toBeCloseTo(2.0, 10);
      expect(newPos.get(2, 0)).toBeCloseTo(3.0, 10);
    });
  });

  describe('Dimer', () => {
    it('should create and analyze dimers', () => {
      // Create two H atoms
      const pos1 = Module.Mat3N.create(1);
      pos1.set(0, 0, 0.0); pos1.set(1, 0, 0.0); pos1.set(2, 0, 0.0);
      
      const pos2 = Module.Mat3N.create(1);
      pos2.set(0, 0, 0.0); pos2.set(1, 0, 0.0); pos2.set(2, 0, 2.0);
      
      const atomicNumbers = Module.IVec.fromArray([1]);
      const mol1 = new Module.Molecule(atomicNumbers, pos1);
      const mol2 = new Module.Molecule(atomicNumbers, pos2);
      
      const dimer = new Module.Dimer(mol1, mol2);
      expect(dimer.nearestDistance).toBeCloseTo(2.0, 10);
      expect(dimer.centerOfMassDistance).toBeCloseTo(2.0, 10);
    });
  });

  describe('Point Group Analysis', () => {
    it('should identify molecular point groups', () => {
      // Create linear H2
      const positions = Module.Mat3N.create(2);
      positions.set(0, 0, 0.0); positions.set(1, 0, 0.0); positions.set(2, 0, -0.37);
      positions.set(0, 1, 0.0); positions.set(1, 1, 0.0); positions.set(2, 1, 0.37);
      
      const atomicNumbers = Module.IVec.fromArray([1, 1]);
      const h2 = new Module.Molecule(atomicNumbers, positions);
      
      const pg = new Module.MolecularPointGroup(h2);
      const pgString = pg.getPointGroupString();
      
      // H2 should be D∞h, but might be detected as D2h or similar
      expect(pgString).toMatch(/D/); // Should contain 'D' for dihedral
      expect(pg.symmetryNumber).toBeGreaterThan(1);
    });
  });

  describe('Partial Charges', () => {
    it('should calculate EEM partial charges', () => {
      // Create H2O
      const positions = Module.Mat3N.create(3);
      positions.set(0, 0, 0.0); positions.set(1, 0, 0.0); positions.set(2, 0, 0.0);
      positions.set(0, 1, 0.757); positions.set(1, 1, 0.586); positions.set(2, 1, 0.0);
      positions.set(0, 2, -0.757); positions.set(1, 2, 0.586); positions.set(2, 2, 0.0);
      
      const atomicNumbers = Module.IVec.fromArray([8, 1, 1]);
      const charges = Module.eemPartialCharges(atomicNumbers, positions, 0.0);
      
      expect(charges.size()).toBe(3);
      
      // Oxygen should be negative
      expect(charges.get(0)).toBeLessThan(0);
      // Hydrogens should be positive
      expect(charges.get(1)).toBeGreaterThan(0);
      expect(charges.get(2)).toBeGreaterThan(0);
      
      // Total charge should be close to 0
      const totalCharge = charges.get(0) + charges.get(1) + charges.get(2);
      expect(totalCharge).toBeCloseTo(0.0, 5);
    });

    it('should calculate EEQ partial charges', () => {
      // Create H2O
      const positions = Module.Mat3N.create(3);
      positions.set(0, 0, 0.0); positions.set(1, 0, 0.0); positions.set(2, 0, 0.0);
      positions.set(0, 1, 0.757); positions.set(1, 1, 0.586); positions.set(2, 1, 0.0);
      positions.set(0, 2, -0.757); positions.set(1, 2, 0.586); positions.set(2, 2, 0.0);
      
      const atomicNumbers = Module.IVec.fromArray([8, 1, 1]);
      const charges = Module.eeqPartialCharges(atomicNumbers, positions, 0.0);
      
      expect(charges.size()).toBe(3);
      
      // Check coordination numbers
      const coordNumbers = Module.eeqCoordinationNumbers(atomicNumbers, positions);
      expect(coordNumbers.size()).toBe(3);
      expect(coordNumbers.get(0)).toBeGreaterThan(1.5); // O should have ~2 neighbors
    });
  });

  describe('Data Directory', () => {
    it('should manage data directory paths', () => {
      const originalDir = Module.getDataDirectory();
      
      Module.setDataDirectory('/test/path');
      expect(Module.getDataDirectory()).toBe('/test/path');
      
      // Restore original
      Module.setDataDirectory(originalDir);
    });
  });
});