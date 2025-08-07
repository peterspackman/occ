import { describe, it, expect, beforeAll } from 'vitest';
import { 
  loadOCC, 
  moleculeFromXYZ, 
  createQMCalculation
} from '../dist/index.js';

describe('Volume Generation Tests', () => {
  let OCC;
  let waterMolecule;
  let wavefunction;

  beforeAll(async () => {
    OCC = await loadOCC();
    
    // Create water molecule for testing
    waterMolecule = await moleculeFromXYZ(`3
Water molecule
O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692`);

    // Run HF calculation to get wavefunction
    const calc = await createQMCalculation(waterMolecule, 'sto-3g');
    await calc.runHF();
    wavefunction = calc.wavefunction;
  });

  describe('OCC.VolumeCalculator basic functionality', () => {
    it('should create OCC.VolumeCalculator instance', () => {
      const calculator = new OCC.VolumeCalculator();
      expect(calculator).toBeDefined();
      calculator.delete();
    });

    it('should set wavefunction without error', () => {
      const calculator = new OCC.VolumeCalculator();
      expect(() => calculator.setWavefunction(wavefunction)).not.toThrow();
      calculator.delete();
    });

    it('should set molecule without error', () => {
      const calculator = new OCC.VolumeCalculator();
      expect(() => calculator.setMolecule(waterMolecule)).not.toThrow();
      calculator.delete();
    });
  });

  describe('OCC.VolumeGenerationParameters', () => {
    it('should create parameters instance', () => {
      const params = new OCC.VolumeGenerationParameters();
      expect(params).toBeDefined();
      params.delete();
    });

    it('should set property type', () => {
      const params = new OCC.VolumeGenerationParameters();
      params.property = OCC.VolumePropertyKind.ElectronDensity;
      expect(params.property).toBe(OCC.VolumePropertyKind.ElectronDensity);
      params.delete();
    });

    it('should set grid steps', () => {
      const params = new OCC.VolumeGenerationParameters();
      params.setSteps(2, 2, 2);
      // Note: Can't verify steps directly without getter
      params.delete();
    });
  });

  describe('Volume computation', () => {
    it('should compute electron density volume', () => {
      const calculator = new OCC.VolumeCalculator();
      calculator.setWavefunction(wavefunction);
      
      const params = new OCC.VolumeGenerationParameters();
      params.property = OCC.VolumePropertyKind.ElectronDensity;
      params.setSteps(2, 2, 2); // Small grid for fast test
      
      const volume = calculator.computeVolume(params);
      expect(volume).toBeDefined();
      expect(volume.nx()).toBe(2);
      expect(volume.ny()).toBe(2);
      expect(volume.nz()).toBe(2);
      expect(volume.totalPoints()).toBe(8);
      
      // Check that data is not all zeros
      const data = volume.getData();
      expect(data.length).toBe(8);
      const hasNonZero = Array.from(data).some(val => val !== 0);
      expect(hasNonZero).toBe(true);
      
      volume.delete();
      params.delete();
      calculator.delete();
    });

    it('should compute ESP volume', () => {
      const calculator = new OCC.VolumeCalculator();
      calculator.setWavefunction(wavefunction);
      
      const params = new OCC.VolumeGenerationParameters();
      params.property = OCC.VolumePropertyKind.ElectricPotential;
      params.setSteps(2, 2, 2); // Very small grid for fast test
      
      const volume = calculator.computeVolume(params);
      expect(volume).toBeDefined();
      expect(volume.totalPoints()).toBe(8);
      
      volume.delete();
      params.delete();
      calculator.delete();
    });

    it('should generate cube file string', () => {
      const calculator = new OCC.VolumeCalculator();
      calculator.setWavefunction(wavefunction);
      
      const params = new OCC.VolumeGenerationParameters();
      params.property = OCC.VolumePropertyKind.ElectronDensity;
      params.setSteps(2, 2, 2); // Very small grid
      
      const volume = calculator.computeVolume(params);
      const cubeString = calculator.volumeAsCubeString(volume);
      
      expect(cubeString).toBeDefined();
      expect(typeof cubeString).toBe('string');
      expect(cubeString).toContain('electron_density'); // Should have comment line
      expect(cubeString).toContain('3'); // Should have 3 atoms (water)
      
      // Check format has the right structure
      const lines = cubeString.split('\n');
      expect(lines.length).toBeGreaterThan(10); // Header + atoms + data
      
      volume.delete();
      params.delete();
      calculator.delete();
    });
  });

  describe('Convenience functions', () => {
    it('should generate electron density cube via convenience function', () => {
      const cubeString = OCC.generateElectronDensityCube(wavefunction, 2, 2, 2);
      expect(cubeString).toBeDefined();
      expect(typeof cubeString).toBe('string');
      expect(cubeString).toContain('electron_density');
    });

    it('should generate MO cube via convenience function', () => {
      const homoIndex = 4; // Water has 5 occupied MOs (0-4)
      const cubeString = OCC.generateMOCube(wavefunction, homoIndex, 2, 2, 2);
      expect(cubeString).toBeDefined();
      expect(typeof cubeString).toBe('string');
      // Should contain electron_density (since MO cubes use electron density property)
      expect(cubeString).toContain('electron_density');
    });

    it('should generate alpha MO cube with spin constraint', () => {
      const homoIndex = 4;
      const cubeString = OCC.generateMOCubeWithSpin(
        wavefunction, 
        homoIndex, 
        OCC.SpinConstraint.Alpha, 
        2, 2, 2
      );
      expect(cubeString).toBeDefined();
      expect(typeof cubeString).toBe('string');
    });
  });

  describe('Static methods', () => {
    it('should compute density volume via static method', () => {
      const params = new OCC.VolumeGenerationParameters();
      params.property = OCC.VolumePropertyKind.ElectronDensity;
      params.setSteps(2, 2, 2);
      
      const volume = OCC.VolumeCalculator.computeDensityVolume(wavefunction, params);
      expect(volume).toBeDefined();
      expect(volume.totalPoints()).toBe(8);
      
      volume.delete();
      params.delete();
    });

    it('should compute MO volume via static method', () => {
      const params = new OCC.VolumeGenerationParameters();
      params.property = OCC.VolumePropertyKind.ElectronDensity;
      params.setSteps(2, 2, 2);
      params.mo_number = 4; // HOMO
      
      const volume = OCC.VolumeCalculator.computeMOVolume(wavefunction, 4, params);
      expect(volume).toBeDefined();
      expect(volume.totalPoints()).toBe(8);
      
      volume.delete();
      params.delete();
    });
  });

  describe('VolumeData access', () => {
    it('should access volume properties', () => {
      const calculator = new OCC.VolumeCalculator();
      calculator.setWavefunction(wavefunction);
      
      const params = new OCC.VolumeGenerationParameters();
      params.property = OCC.VolumePropertyKind.ElectronDensity;
      params.setSteps(2, 2, 2);
      
      const volume = calculator.computeVolume(params);
      
      // Test dimension accessors
      expect(volume.nx()).toBe(2);
      expect(volume.ny()).toBe(2);
      expect(volume.nz()).toBe(2);
      expect(volume.totalPoints()).toBe(8);
      
      // Test origin accessor
      const origin = volume.getOrigin();
      expect(origin).toBeDefined();
      expect(origin.length).toBe(3);
      
      // Test basis accessor
      const basis = volume.getBasis();
      expect(basis).toBeDefined();
      expect(basis.length).toBe(9); // 3x3 matrix flattened
      
      // Test steps accessor
      const steps = volume.getSteps();
      expect(steps).toBeDefined();
      expect(steps.length).toBe(3);
      expect(steps[0]).toBe(2);
      expect(steps[1]).toBe(2);
      expect(steps[2]).toBe(2);
      
      volume.delete();
      params.delete();
      calculator.delete();
    });

    it('should have non-zero electron density near nuclei', () => {
      const calculator = new OCC.VolumeCalculator();
      calculator.setWavefunction(wavefunction);
      
      const params = new OCC.VolumeGenerationParameters();
      params.property = OCC.VolumePropertyKind.ElectronDensity;
      params.setSteps(2, 2, 2);
      
      const volume = calculator.computeVolume(params);
      const data = volume.getData();
      
      // Find maximum density value
      let maxDensity = 0;
      for (let i = 0; i < data.length; i++) {
        if (data[i] > maxDensity) {
          maxDensity = data[i];
        }
      }
      
      // Electron density should have significant values
      expect(maxDensity).toBeGreaterThan(0.1);
      
      volume.delete();
      params.delete();
      calculator.delete();
    });
  });

  describe('Error handling', () => {
    it('should handle computing volume without wavefunction', () => {
      const calculator = new OCC.VolumeCalculator();
      // Don't set wavefunction
      
      const params = new OCC.VolumeGenerationParameters();
      params.property = OCC.VolumePropertyKind.ElectronDensity;
      params.setSteps(2, 2, 2);
      
      // Should either throw or return empty volume
      expect(() => {
        const volume = calculator.computeVolume(params);
        if (volume) {
          volume.delete();
        }
      }).toThrow();
      
      params.delete();
      calculator.delete();
    });
  });
});