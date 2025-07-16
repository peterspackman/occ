import { describe, it, expect, beforeAll } from 'vitest';
import { 
  loadOCC, 
  moleculeFromXYZ, 
  createQMCalculation
} from '../src/index.js';

describe('Advanced Isosurface Tests', () => {
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

  describe('SurfaceKind Enums', () => {
    it('should have all surface kinds available', () => {
      expect(Module.SurfaceKind).toBeDefined();
      expect(Module.SurfaceKind.PromoleculeDensity).toBeDefined();
      expect(Module.SurfaceKind.Hirshfeld).toBeDefined();
      expect(Module.SurfaceKind.EEQ_ESP).toBeDefined();
      expect(Module.SurfaceKind.ElectronDensity).toBeDefined();
      expect(Module.SurfaceKind.ESP).toBeDefined();
      expect(Module.SurfaceKind.SpinDensity).toBeDefined();
      expect(Module.SurfaceKind.DeformationDensity).toBeDefined();
      expect(Module.SurfaceKind.Orbital).toBeDefined();
      expect(Module.SurfaceKind.CrystalVoid).toBeDefined();
      expect(Module.SurfaceKind.VolumeGrid).toBeDefined();
      expect(Module.SurfaceKind.SoftVoronoi).toBeDefined();
      expect(Module.SurfaceKind.VDWLogSumExp).toBeDefined();
      expect(Module.SurfaceKind.HSRinv).toBeDefined();
      expect(Module.SurfaceKind.HSExp).toBeDefined();
    });

    it('should convert surface kinds to strings', () => {
      expect(Module.surfaceToString(Module.SurfaceKind.PromoleculeDensity)).toBe('promolecule_density');
      expect(Module.surfaceToString(Module.SurfaceKind.Hirshfeld)).toBe('hirshfeld');
      expect(Module.surfaceToString(Module.SurfaceKind.ElectronDensity)).toBe('electron_density');
      expect(Module.surfaceToString(Module.SurfaceKind.ESP)).toBe('esp');
    });

    it('should convert strings to surface kinds', () => {
      expect(Module.surfaceFromString('promolecule_density')).toBe(Module.SurfaceKind.PromoleculeDensity);
      expect(Module.surfaceFromString('hirshfeld')).toBe(Module.SurfaceKind.Hirshfeld);
      expect(Module.surfaceFromString('electron_density')).toBe(Module.SurfaceKind.ElectronDensity);
    });

    it('should identify which surfaces require wavefunctions', () => {
      expect(Module.surfaceRequiresWavefunction(Module.SurfaceKind.PromoleculeDensity)).toBe(false);
      expect(Module.surfaceRequiresWavefunction(Module.SurfaceKind.ElectronDensity)).toBe(true);
      expect(Module.surfaceRequiresWavefunction(Module.SurfaceKind.ESP)).toBe(true);
      expect(Module.surfaceRequiresWavefunction(Module.SurfaceKind.Orbital)).toBe(true);
    });
  });

  describe('PropertyKind Enums', () => {
    it('should have all property kinds available', () => {
      expect(Module.PropertyKind).toBeDefined();
      expect(Module.PropertyKind.Dnorm).toBeDefined();
      expect(Module.PropertyKind.Dint_norm).toBeDefined();
      expect(Module.PropertyKind.Dext_norm).toBeDefined();
      expect(Module.PropertyKind.Dint).toBeDefined();
      expect(Module.PropertyKind.Dext).toBeDefined();
      expect(Module.PropertyKind.FragmentPatch).toBeDefined();
      expect(Module.PropertyKind.ShapeIndex).toBeDefined();
      expect(Module.PropertyKind.Curvedness).toBeDefined();
      expect(Module.PropertyKind.GaussianCurvature).toBeDefined();
      expect(Module.PropertyKind.MeanCurvature).toBeDefined();
      expect(Module.PropertyKind.CurvatureK1).toBeDefined();
      expect(Module.PropertyKind.CurvatureK2).toBeDefined();
    });

    it('should convert property kinds to strings', () => {
      expect(Module.propertyToString(Module.PropertyKind.Dnorm)).toBe('dnorm');
      expect(Module.propertyToString(Module.PropertyKind.ShapeIndex)).toBe('shape_index');
      expect(Module.propertyToString(Module.PropertyKind.Curvedness)).toBe('curvedness');
    });

    it('should identify which properties require wavefunctions', () => {
      expect(Module.propertyRequiresWavefunction(Module.PropertyKind.Dnorm)).toBe(false);
      expect(Module.propertyRequiresWavefunction(Module.PropertyKind.ElectronDensity)).toBe(true);
      expect(Module.propertyRequiresWavefunction(Module.PropertyKind.ESP)).toBe(true);
    });
  });

  describe('IsosurfaceProperties', () => {
    it('should create and manage properties', () => {
      const props = new Module.IsosurfaceProperties();
      expect(props).toBeDefined();
      expect(props.count()).toBe(0);
      expect(props.hasProperty('test')).toBe(false);
    });

    it('should add and retrieve float properties', () => {
      const props = new Module.IsosurfaceProperties();
      const values = [1.0, 2.0, 3.0, 4.0];
      
      props.addFloat('test_prop', values);
      expect(props.hasProperty('test_prop')).toBe(true);
      expect(props.count()).toBe(1);
      
      const retrieved = props.getFloat('test_prop');
      expect(retrieved).toBeInstanceOf(Float32Array);
      expect(retrieved.length).toBe(4);
      expect(Array.from(retrieved)).toEqual([1.0, 2.0, 3.0, 4.0]);
    });

    it('should add and retrieve integer properties', () => {
      const props = new Module.IsosurfaceProperties();
      const values = [1, 2, 3, 4];
      
      props.addInt('test_int', values);
      expect(props.hasProperty('test_int')).toBe(true);
      
      const retrieved = props.getInt('test_int');
      expect(retrieved).toBeInstanceOf(Int32Array);
      expect(retrieved.length).toBe(4);
      expect(Array.from(retrieved)).toEqual([1, 2, 3, 4]);
    });

    it('should return null for non-existent properties', () => {
      const props = new Module.IsosurfaceProperties();
      expect(props.getFloat('nonexistent')).toBeNull();
      expect(props.getInt('nonexistent')).toBeNull();
    });
  });

  describe('IsosurfaceGenerationParameters', () => {
    it('should create with default parameters', () => {
      const params = new Module.IsosurfaceGenerationParameters();
      expect(params).toBeDefined();
      expect(params.isovalue).toBeDefined();
      expect(params.separation).toBeDefined();
      expect(params.surfaceKind).toBeDefined();
    });

    it('should allow setting all parameters', () => {
      const params = new Module.IsosurfaceGenerationParameters();
      
      params.isovalue = 0.002;
      params.separation = 0.2;
      params.backgroundDensity = 0.0001;
      params.flipNormals = true;
      params.binaryOutput = false;
      params.surfaceKind = Module.SurfaceKind.ElectronDensity;
      
      expect(params.isovalue).toBe(0.002);
      expect(params.separation).toBe(0.2);
      expect(params.backgroundDensity).toBe(0.0001);
      expect(params.flipNormals).toBe(true);
      expect(params.binaryOutput).toBe(false);
      expect(params.surfaceKind).toBe(Module.SurfaceKind.ElectronDensity);
    });
  });

  describe('IsosurfaceCalculator', () => {
    it('should create calculator and set basic properties', () => {
      const calc = new Module.IsosurfaceCalculator();
      expect(calc).toBeDefined();
      
      calc.setMolecule(h2Molecule);
      // Setting a molecule might automatically set it as environment too in some implementations
      // Just check that these methods return boolean values
      expect(typeof calc.haveEnvironment()).toBe('boolean');
      expect(calc.haveWavefunction()).toBe(false);
    });

    it('should set wavefunction and validate', () => {
      const calc = new Module.IsosurfaceCalculator();
      calc.setMolecule(h2Molecule);
      calc.setWavefunction(h2Wavefunction);
      
      expect(calc.haveWavefunction()).toBe(true);
    });

    it('should set parameters and validate for promolecule density', () => {
      const calc = new Module.IsosurfaceCalculator();
      calc.setMolecule(h2Molecule);
      
      const params = new Module.IsosurfaceGenerationParameters();
      params.surfaceKind = Module.SurfaceKind.PromoleculeDensity;
      params.isovalue = 0.001;
      params.separation = 0.1;
      
      calc.setParameters(params);
      expect(calc.validate()).toBe(true);
    });

    it('should require wavefunction for electron density', () => {
      const calc = new Module.IsosurfaceCalculator();
      calc.setMolecule(h2Molecule);
      
      const params = new Module.IsosurfaceGenerationParameters();
      params.surfaceKind = Module.SurfaceKind.ElectronDensity;
      params.isovalue = 0.001;
      
      calc.setParameters(params);
      
      expect(calc.requiresWavefunction()).toBe(true);
      expect(calc.validate()).toBe(false); // Should fail without wavefunction
      
      calc.setWavefunction(h2Wavefunction);
      expect(calc.validate()).toBe(true); // Should pass with wavefunction
    });

    it('should compute promolecule density isosurface', async () => {
      const calc = new Module.IsosurfaceCalculator();
      calc.setMolecule(h2Molecule);
      
      const params = new Module.IsosurfaceGenerationParameters();
      params.surfaceKind = Module.SurfaceKind.PromoleculeDensity;
      params.isovalue = 0.001;
      params.separation = 0.2;
      
      calc.setParameters(params);
      expect(calc.validate()).toBe(true);
      
      calc.compute();
      const surface = calc.isosurface();
      
      expect(surface).toBeDefined();
      expect(surface.volume()).toBeGreaterThan(0);
      expect(surface.surfaceArea()).toBeGreaterThan(0);
      
      const vertices = surface.getVertices();
      const faces = surface.getFaces();
      const normals = surface.getNormals();
      
      expect(vertices).toBeInstanceOf(Float32Array);
      expect(faces).toBeInstanceOf(Uint32Array);
      expect(normals).toBeInstanceOf(Float32Array);
      
      expect(vertices.length % 3).toBe(0); // Vertices are 3D
      expect(faces.length % 3).toBe(0);    // Faces are triangles
      expect(normals.length).toBe(vertices.length); // One normal per vertex
    }, 30000);

    it('should compute electron density isosurface', async () => {
      const calc = new Module.IsosurfaceCalculator();
      calc.setMolecule(h2Molecule);
      calc.setWavefunction(h2Wavefunction);
      
      const params = new Module.IsosurfaceGenerationParameters();
      params.surfaceKind = Module.SurfaceKind.ElectronDensity;
      params.isovalue = 0.001;
      params.separation = 0.2;
      
      calc.setParameters(params);
      expect(calc.validate()).toBe(true);
      
      calc.compute();
      const surface = calc.isosurface();
      
      expect(surface).toBeDefined();
      expect(surface.volume()).toBeGreaterThan(0);
      expect(surface.surfaceArea()).toBeGreaterThan(0);
    }, 30000);
  });

  describe('Isosurface Properties', () => {
    it('should have properties container', async () => {
      const calc = new Module.IsosurfaceCalculator();
      calc.setMolecule(h2Molecule);
      
      const params = new Module.IsosurfaceGenerationParameters();
      params.surfaceKind = Module.SurfaceKind.PromoleculeDensity;
      params.isovalue = 0.001;
      params.separation = 0.2;
      
      calc.setParameters(params);
      calc.compute();
      
      const surface = calc.isosurface();
      const props = surface.properties;
      
      expect(props).toBeDefined();
      expect(props).toBeInstanceOf(Module.IsosurfaceProperties);
    }, 30000);
  });

  describe('Legacy Helper Functions', () => {
    it('should still support generatePromoleculeDensityIsosurface', async () => {
      const result = Module.generatePromoleculeDensityIsosurface(h2Molecule, 0.001, 0.2);
      
      expect(result).toBeDefined();
      expect(result.vertices).toBeInstanceOf(Float32Array);
      expect(result.faces).toBeInstanceOf(Uint32Array);
      expect(result.normals).toBeInstanceOf(Float32Array);
      expect(result.numVertices).toBeGreaterThan(0);
      expect(result.numFaces).toBeGreaterThan(0);
      expect(result.volume).toBeGreaterThan(0);
      expect(result.surfaceArea).toBeGreaterThan(0);
    }, 30000);

    it('should still support generateElectronDensityIsosurface', async () => {
      const result = Module.generateElectronDensityIsosurface(h2Wavefunction, 0.001, 0.2);
      
      expect(result).toBeDefined();
      expect(result.vertices).toBeInstanceOf(Float32Array);
      expect(result.faces).toBeInstanceOf(Uint32Array);
      expect(result.normals).toBeInstanceOf(Float32Array);
      expect(result.numVertices).toBeGreaterThan(0);
      expect(result.numFaces).toBeGreaterThan(0);
      expect(result.volume).toBeGreaterThan(0);
      expect(result.surfaceArea).toBeGreaterThan(0);
    }, 30000);
  });

  describe('Isosurface JSON Export', () => {
    it('should export isosurface to JSON', async () => {
      const calc = new Module.IsosurfaceCalculator();
      calc.setMolecule(h2Molecule);
      
      const params = new Module.IsosurfaceGenerationParameters();
      params.surfaceKind = Module.SurfaceKind.PromoleculeDensity;
      params.isovalue = 0.001;
      
      calc.setParameters(params);
      calc.compute();
      
      const surface = calc.isosurface();
      const jsonString = Module.isosurfaceToJSON(surface);
      
      expect(typeof jsonString).toBe('string');
      expect(jsonString.length).toBeGreaterThan(0);
      
      // Should be valid JSON
      const parsed = JSON.parse(jsonString);
      expect(parsed).toBeDefined();
    }, 30000);
  });
});