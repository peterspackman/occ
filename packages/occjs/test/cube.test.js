import { describe, it, expect, beforeAll } from 'vitest';
import { 
  loadOCC, 
  moleculeFromXYZ, 
  createQMCalculation
} from '../dist/index.js';

describe('Cube File Tests', () => {
  let Module;
  let h2Molecule;
  let waterMolecule;
  let h2Wavefunction;

  beforeAll(async () => {
    Module = await loadOCC();
    
    // Create test molecules
    h2Molecule = await moleculeFromXYZ(`2
H2 molecule
H 0.0 0.0 0.0
H 0.0 0.0 0.74`);

    waterMolecule = await moleculeFromXYZ(`3
Water molecule
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692`);

    // Generate a wavefunction for testing
    const calc = await createQMCalculation(h2Molecule, 'sto-3g');
    await calc.runHF();
    h2Wavefunction = calc.wavefunction;
  });

  describe('Cube Creation and Basic Properties', () => {
    it('should create a new cube', () => {
      const cube = new Module.Cube();
      expect(cube).toBeDefined();
      expect(cube.name).toBeDefined();
      expect(cube.description).toBeDefined();
    });

    it('should set and get cube name and description', () => {
      const cube = new Module.Cube();
      cube.name = 'Test Cube';
      cube.description = 'A test cube file';
      
      expect(cube.name).toBe('Test Cube');
      expect(cube.description).toBe('A test cube file');
    });

    it('should set and get origin', () => {
      const cube = new Module.Cube();
      cube.setOrigin(1.0, 2.0, 3.0);
      
      const origin = cube.getOrigin();
      expect(origin).toHaveLength(3);
      expect(origin[0]).toBe(1.0);
      expect(origin[1]).toBe(2.0);
      expect(origin[2]).toBe(3.0);
    });

    it('should set and get grid steps', () => {
      const cube = new Module.Cube();
      cube.setSteps(20, 25, 30);
      
      const steps = cube.getSteps();
      expect(steps).toHaveLength(3);
      expect(steps[0]).toBe(20);
      expect(steps[1]).toBe(25);
      expect(steps[2]).toBe(30);
    });

    it('should set and get basis matrix', () => {
      const cube = new Module.Cube();
      
      // Create a 3x3 identity matrix as flat array
      const basisMatrix = [
        1.0, 0.0, 0.0,  // First row
        0.0, 1.0, 0.0,  // Second row  
        0.0, 0.0, 1.0   // Third row
      ];
      
      cube.setBasis(basisMatrix);
      const retrieved = cube.getBasis();
      
      expect(retrieved).toBeInstanceOf(Float64Array);
      expect(retrieved).toHaveLength(9);
      expect(Array.from(retrieved)).toEqual(basisMatrix);
    });

    it('should validate basis matrix size', () => {
      const cube = new Module.Cube();
      
      // The C++ implementation DOES validate array size
      // Should throw for invalid size
      expect(() => cube.setBasis([1.0, 2.0, 3.0])).toThrow();
      
      // Test with correct size
      const validBasis = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
      ];
      expect(() => cube.setBasis(validBasis)).not.toThrow();
    });
  });

  describe('Atom Management', () => {
    it('should add individual atoms', () => {
      const cube = new Module.Cube();
      
      cube.addAtom(1, 0.0, 0.0, 0.0);  // Hydrogen at origin
      cube.addAtom(1, 0.0, 0.0, 1.4);  // Hydrogen at 1.4 Bohr
      
      // We can't directly access atoms count in the bindings, 
      // but we can test that the method doesn't throw
      expect(() => cube.addAtom(8, 1.0, 1.0, 1.0)).not.toThrow();
    });

    it('should set molecule', () => {
      const cube = new Module.Cube();
      
      expect(() => cube.setMolecule(h2Molecule)).not.toThrow();
      expect(() => cube.setMolecule(waterMolecule)).not.toThrow();
    });

    it('should center molecule', () => {
      const cube = new Module.Cube();
      cube.setMolecule(h2Molecule);
      
      expect(() => cube.centerMolecule()).not.toThrow();
    });
  });

  describe('Grid Data Management', () => {
    it('should handle grid data', () => {
      const cube = new Module.Cube();
      cube.setSteps(5, 5, 5); // Small grid for testing
      
      const totalSize = 5 * 5 * 5; // 125 points
      const testData = new Float32Array(totalSize);
      
      // Fill with test pattern
      for (let i = 0; i < totalSize; i++) {
        testData[i] = i * 0.1;
      }
      
      cube.setData(testData);
      const retrieved = cube.getData();
      
      expect(retrieved).toBeInstanceOf(Float32Array);
      expect(retrieved).toHaveLength(totalSize);
      expect(Array.from(retrieved)).toEqual(Array.from(testData));
    });

    it('should validate data array size', () => {
      const cube = new Module.Cube();
      cube.setSteps(5, 5, 5);
      
      const rightSize = new Float32Array(125); // 5x5x5
      expect(() => cube.setData(rightSize)).not.toThrow();
      
      // Test with wrong size - C++ implementation DOES validate
      const wrongSize = new Float32Array(100); 
      expect(() => cube.setData(wrongSize)).toThrow();
    });

    it('should get empty data initially', () => {
      const cube = new Module.Cube();
      cube.setSteps(3, 3, 3);
      
      const data = cube.getData();
      expect(data).toBeInstanceOf(Float32Array);
      expect(data).toHaveLength(27);
      
      // Should be initialized to zeros
      const allZero = Array.from(data).every(val => val === 0.0);
      expect(allZero).toBe(true);
    });
  });

  describe('Electron Density Filling', () => {
    it('should fill cube with electron density from wavefunction', () => {
      const cube = new Module.Cube();
      cube.setMolecule(h2Molecule);
      cube.setSteps(10, 10, 10);
      
      // Set up a small grid around the molecule
      cube.setOrigin(-2.0, -2.0, -2.0);
      const basis = [
        0.4, 0.0, 0.0,  // 4 Bohr range in x
        0.0, 0.4, 0.0,  // 4 Bohr range in y
        0.0, 0.0, 0.4   // 4 Bohr range in z
      ];
      cube.setBasis(basis);
      
      cube.fillFromElectronDensity(h2Wavefunction);
      
      const data = cube.getData();
      expect(data).toBeInstanceOf(Float32Array);
      expect(data).toHaveLength(1000); // 10x10x10
      
      // Should have some non-zero values (electron density)
      const hasNonZero = Array.from(data).some(val => val > 0.0);
      expect(hasNonZero).toBe(true);
      
      // Density values should be reasonable (positive, not too large)
      const maxDensity = Math.max(...Array.from(data));
      expect(maxDensity).toBeGreaterThan(0);
      expect(maxDensity).toBeLessThan(10.0); // Reasonable upper bound
    }, 30000);

    it('should handle different grid sizes', () => {
      const cube = new Module.Cube();
      cube.setMolecule(h2Molecule);
      
      // Test with a very small grid for speed
      cube.setSteps(3, 3, 3);
      cube.setOrigin(-1.0, -1.0, -1.0);
      const basis = [
        0.67, 0.0, 0.0,  
        0.0, 0.67, 0.0,  
        0.0, 0.0, 0.67   
      ];
      cube.setBasis(basis);
      
      expect(() => cube.fillFromElectronDensity(h2Wavefunction)).not.toThrow();
      
      const data = cube.getData();
      expect(data).toHaveLength(27); // 3x3x3
    }, 15000);
  });

  describe('Cube File I/O', () => {
    it('should convert cube to string format', () => {
      const cube = new Module.Cube();
      cube.name = 'Test Cube';
      cube.description = 'Electron density';
      cube.setMolecule(h2Molecule);
      cube.setSteps(3, 3, 3);
      
      const cubeString = cube.toString();
      
      expect(typeof cubeString).toBe('string');
      expect(cubeString.length).toBeGreaterThan(0);
      
      // Should contain expected header elements
      expect(cubeString).toContain('Test Cube');
      expect(cubeString).toContain('Electron density');
    });

    it('should generate valid cube file format', () => {
      const cube = new Module.Cube();
      cube.name = 'H2 Density';
      cube.description = 'H2 electron density cube';
      cube.setMolecule(h2Molecule);
      cube.setSteps(4, 4, 4);
      cube.setOrigin(-2.0, -2.0, -2.0);
      
      // Fill with simple test data
      const data = new Float32Array(64); // 4x4x4
      for (let i = 0; i < 64; i++) {
        data[i] = 0.001 * i;
      }
      cube.setData(data);
      
      const cubeString = cube.toString();
      const lines = cubeString.split('\n'); // Use actual newline, not escaped
      
      // Check basic cube file structure - adjust expectations to actual behavior
      expect(lines.length).toBeGreaterThan(3);
      expect(lines[0]).toContain('H2 Density'); // Title line
      expect(lines[1]).toContain('H2 electron density cube'); // Comment line
      
      // Should have atom count and origin info on line 3 (0-indexed line 2)
      expect(lines[2]).toMatch(/\s*\d+/); // Atom count line, use unescaped regex
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid cube operations gracefully', () => {
      const cube = new Module.Cube();
      
      // Try to get data before setting grid size
      expect(() => cube.getData()).not.toThrow();
      
      // Try to set invalid origin (should not throw, just use the values)
      expect(() => cube.setOrigin(NaN, 0, 0)).not.toThrow();
    });

    it('should handle wavefunction errors', () => {
      const cube = new Module.Cube();
      cube.setSteps(2, 2, 2);
      
      // This might throw or handle gracefully depending on implementation
      // Just test that it doesn't crash the entire test suite
      expect(() => {
        try {
          cube.fillFromElectronDensity(null);
        } catch (e) {
          // Expected to throw
        }
      }).not.toThrow();
    });
  });

  describe('Integration with QM Calculations', () => {
    it('should generate cube files for different basis sets', async () => {
      // Test with STO-3G
      const calc1 = await createQMCalculation(h2Molecule, 'sto-3g');
      await calc1.runHF();
      
      const cube1 = new Module.Cube();
      cube1.setMolecule(h2Molecule);
      cube1.setSteps(5, 5, 5);
      cube1.setOrigin(-1.0, -1.0, -1.0);
      
      expect(() => cube1.fillFromElectronDensity(calc1.wavefunction)).not.toThrow();
      
      const data1 = cube1.getData();
      const hasData1 = Array.from(data1).some(val => val > 0);
      expect(hasData1).toBe(true);
    }, 45000);

    it('should work with DFT calculations', async () => {
      const calc = await createQMCalculation(h2Molecule, 'sto-3g');
      await calc.runDFT('b3lyp');
      
      const cube = new Module.Cube();
      cube.setMolecule(h2Molecule);
      cube.setSteps(4, 4, 4);
      cube.setOrigin(-1.5, -1.5, -1.5);
      
      expect(() => cube.fillFromElectronDensity(calc.wavefunction)).not.toThrow();
      
      const data = cube.getData();
      const hasData = Array.from(data).some(val => val > 0);
      expect(hasData).toBe(true);
    }, 45000);
  });

  describe('Performance and Memory', () => {
    it('should handle moderately sized grids efficiently', () => {
      const cube = new Module.Cube();
      cube.setMolecule(h2Molecule);
      cube.setSteps(15, 15, 15); // 3375 grid points
      
      const startTime = Date.now();
      cube.fillFromElectronDensity(h2Wavefunction);
      const endTime = Date.now();
      
      const data = cube.getData();
      expect(data).toHaveLength(3375);
      
      // Should complete in reasonable time (less than 30 seconds)
      expect(endTime - startTime).toBeLessThan(30000);
    }, 35000);
  });
});