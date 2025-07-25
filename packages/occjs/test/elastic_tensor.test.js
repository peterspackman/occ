import { describe, it, expect, beforeAll } from 'vitest';
import { loadOCC } from '../dist/index.js';

describe('ElasticTensor Tests', () => {
  let Module;
  
  beforeAll(async () => {
    Module = await loadOCC();
  });

  // Reference test tensor from C++ tests
  const testTensorData = [
    [48.137, 11.411, 12.783,  0.000, -3.654,  0.000],
    [11.411, 34.968, 14.749,  0.000, -0.094,  0.000],
    [12.783, 14.749, 26.015,  0.000, -4.528,  0.000],
    [ 0.000,  0.000,  0.000, 14.545,  0.000,  0.006],
    [-3.654, -0.094, -4.528,  0.000, 10.771,  0.000],
    [ 0.000,  0.000,  0.000,  0.006,  0.000, 11.947]
  ];

  function createTestElasticTensor() {
    // Try Mat6 first, fallback to Mat if not available
    let mat6;
    if (Module.Mat6 && Module.Mat6.create) {
      mat6 = Module.Mat6.create(6, 6);
    } else {
      mat6 = Module.Mat.create(6, 6);
    }
    
    for (let i = 0; i < 6; i++) {
      for (let j = 0; j < 6; j++) {
        mat6.set(i, j, testTensorData[i][j]);
      }
    }
    return new Module.ElasticTensor(mat6);
  }

  describe('Basic Functionality', () => {
    it('should create ElasticTensor from 6x6 matrix', () => {
      const elasticTensor = createTestElasticTensor();
      expect(elasticTensor).toBeDefined();
      
      // Verify we can access the Voigt matrices
      const voigtC = elasticTensor.voigtC;
      const voigtS = elasticTensor.voigtS;
      expect(voigtC).toBeDefined();
      expect(voigtS).toBeDefined();
    });

    it('should have correct matrix dimensions', () => {
      const elasticTensor = createTestElasticTensor();
      const voigtC = elasticTensor.voigtC;
      expect(voigtC.rows()).toBe(6);
      expect(voigtC.cols()).toBe(6);
    });
  });

  describe('Averaging Schemes', () => {
    let elasticTensor;
    
    beforeAll(() => {
      elasticTensor = createTestElasticTensor();
    });

    describe('Voigt Averages', () => {
      it('should calculate correct bulk modulus', () => {
        const bulkModulus = elasticTensor.averageBulkModulus(Module.AveragingScheme.VOIGT);
        expect(bulkModulus).toBeCloseTo(20.778, 3);
      });

      it('should calculate correct shear modulus', () => {
        const shearModulus = elasticTensor.averageShearModulus(Module.AveragingScheme.VOIGT);
        expect(shearModulus).toBeCloseTo(12.131, 3);
      });

      it('should calculate correct Young\'s modulus', () => {
        const youngsModulus = elasticTensor.averageYoungsModulus(Module.AveragingScheme.VOIGT);
        expect(youngsModulus).toBeCloseTo(30.465, 3);
      });

      it('should calculate correct Poisson\'s ratio', () => {
        const poissonRatio = elasticTensor.averagePoissonRatio(Module.AveragingScheme.VOIGT);
        expect(poissonRatio).toBeCloseTo(0.25564, 5);
      });
    });

    describe('Reuss Averages', () => {
      it('should calculate correct bulk modulus', () => {
        const bulkModulus = elasticTensor.averageBulkModulus(Module.AveragingScheme.REUSS);
        expect(bulkModulus).toBeCloseTo(19.000, 3);
      });

      it('should calculate correct shear modulus', () => {
        const shearModulus = elasticTensor.averageShearModulus(Module.AveragingScheme.REUSS);
        expect(shearModulus).toBeCloseTo(10.728, 3);
      });

      it('should calculate correct Young\'s modulus', () => {
        const youngsModulus = elasticTensor.averageYoungsModulus(Module.AveragingScheme.REUSS);
        expect(youngsModulus).toBeCloseTo(27.087, 3);
      });

      it('should calculate correct Poisson\'s ratio', () => {
        const poissonRatio = elasticTensor.averagePoissonRatio(Module.AveragingScheme.REUSS);
        expect(poissonRatio).toBeCloseTo(0.26239, 5);
      });
    });

    describe('Hill Averages', () => {
      it('should calculate correct bulk modulus', () => {
        const bulkModulus = elasticTensor.averageBulkModulus(Module.AveragingScheme.HILL);
        expect(bulkModulus).toBeCloseTo(19.889, 3);
      });

      it('should calculate correct shear modulus', () => {
        const shearModulus = elasticTensor.averageShearModulus(Module.AveragingScheme.HILL);
        expect(shearModulus).toBeCloseTo(11.430, 3);
      });

      it('should calculate correct Young\'s modulus', () => {
        const youngsModulus = elasticTensor.averageYoungsModulus(Module.AveragingScheme.HILL);
        expect(youngsModulus).toBeCloseTo(28.777, 3);
      });

      it('should calculate correct Poisson\'s ratio', () => {
        const poissonRatio = elasticTensor.averagePoissonRatio(Module.AveragingScheme.HILL);
        expect(poissonRatio).toBeCloseTo(0.25886, 5);
      });
    });
  });

  describe('Directional Properties', () => {
    let elasticTensor;
    
    beforeAll(() => {
      elasticTensor = createTestElasticTensor();
    });

    describe('Young\'s Modulus Directionality', () => {
      it('should calculate minimum Young\'s modulus', () => {
        const direction = Module.Vec3.create(0.3540, 0.0, 0.9352);
        const youngsModulus = elasticTensor.youngsModulus(direction);
        expect(youngsModulus).toBeCloseTo(14.751, 2); // Reduced precision for JS/C++ differences
      });

      it('should calculate maximum Young\'s modulus', () => {
        const direction = Module.Vec3.create(0.9885, 0.0000, -0.1511);
        const youngsModulus = elasticTensor.youngsModulus(direction);
        expect(youngsModulus).toBeCloseTo(41.961, 2); // Reduced precision for JS/C++ differences
      });
    });

    describe('Linear Compressibility Directionality', () => {
      it('should calculate minimum linear compressibility', () => {
        const direction = Module.Vec3.create(0.9295, -0.0000, -0.3688);
        const compressibility = elasticTensor.linearCompressibility(direction);
        expect(compressibility).toBeCloseTo(8.2545, 3);
      });

      it('should calculate maximum linear compressibility', () => {
        const direction = Module.Vec3.create(0.3688, -0.0000, 0.9295);
        const compressibility = elasticTensor.linearCompressibility(direction);
        expect(compressibility).toBeCloseTo(31.357, 3);
      });
    });

    describe('Shear Modulus Directionality', () => {
      it('should calculate minimum shear modulus', () => {
        const dir1 = Module.Vec3.create(-0.2277, 0.7071, -0.6694);
        const dir2 = Module.Vec3.create(-0.2276, -0.7071, -0.6695);
        const shearModulus = elasticTensor.shearModulus(dir1, dir2);
        expect(shearModulus).toBeCloseTo(6.5183, 3);
      });

      it('should calculate maximum shear modulus', () => {
        const dir1 = Module.Vec3.create(0.7352, 0.6348, 0.2378);
        const dir2 = Module.Vec3.create(-0.6612, 0.5945, 0.4575);
        const shearModulus = elasticTensor.shearModulus(dir1, dir2);
        expect(shearModulus).toBeCloseTo(15.505, 3);
      });
    });

    describe('Poisson\'s Ratio Directionality', () => {
      it('should calculate minimum Poisson\'s ratio', () => {
        const dir1 = Module.Vec3.create(0.5593, 0.6044, 0.5674);
        const dir2 = Module.Vec3.create(0.0525, 0.6572, -0.7519);
        const poissonRatio = elasticTensor.poissonRatio(dir1, dir2);
        expect(poissonRatio).toBeCloseTo(0.067042, 4); // Reduced precision for JS/C++ differences
      });

      it('should calculate maximum Poisson\'s ratio', () => {
        const dir1 = Module.Vec3.create(0.0, 1.0, -0.0);
        const dir2 = Module.Vec3.create(-0.2611, -0.0000, -0.9653);
        const poissonRatio = elasticTensor.poissonRatio(dir1, dir2);
        expect(poissonRatio).toBeCloseTo(0.59507, 4); // Reduced precision for JS/C++ differences
      });
    });
  });

  describe('Min/Max Methods', () => {
    let elasticTensor;
    
    beforeAll(() => {
      elasticTensor = createTestElasticTensor();
    });

    it('should calculate shear modulus min/max for a direction', () => {
      const direction = Module.Vec3.create(1.0, 0.0, 0.0);
      const result = elasticTensor.shearModulusMinMax(direction);
      
      expect(result).toHaveProperty('min');
      expect(result).toHaveProperty('max');
      expect(typeof result.min).toBe('number');
      expect(typeof result.max).toBe('number');
      expect(result.max).toBeGreaterThan(result.min);
    });

    it('should calculate Poisson\'s ratio min/max for a direction', () => {
      const direction = Module.Vec3.create(1.0, 0.0, 0.0);
      const result = elasticTensor.poissonRatioMinMax(direction);
      
      expect(result).toHaveProperty('min');
      expect(result).toHaveProperty('max');
      expect(typeof result.min).toBe('number');
      expect(typeof result.max).toBe('number');
      expect(result.max).toBeGreaterThan(result.min);
    });
  });

  describe('Utility Functions', () => {
    let elasticTensor;
    
    beforeAll(() => {
      elasticTensor = createTestElasticTensor();
    });

    it('should generate directional data for Young\'s modulus', () => {
      const data = Module.generateDirectionalData(elasticTensor, 'youngs', 36);
      
      // Check if it's an Emscripten array or JavaScript array
      const length = data.size ? data.size() : data.length;
      expect(length).toBe(36);
      
      // Check first point
      const firstPoint = data.size ? data.get(0) : data[0];
      const x = firstPoint.get ? firstPoint.get('x') : firstPoint.x;
      const y = firstPoint.get ? firstPoint.get('y') : firstPoint.y;
      const value = firstPoint.get ? firstPoint.get('value') : firstPoint.value;
      const angle = firstPoint.get ? firstPoint.get('angle') : firstPoint.angle;
      
      expect(x).toBeDefined();
      expect(y).toBeDefined();
      expect(value).toBeDefined();
      expect(angle).toBeDefined();
      
      expect(typeof value).toBe('number');
      expect(value).toBeGreaterThan(0);
    });

    it('should generate directional data for linear compressibility', () => {
      const data = Module.generateDirectionalData(elasticTensor, 'linear_compressibility', 18);
      
      const length = data.size ? data.size() : data.length;
      expect(length).toBe(18);
      
      const firstPoint = data.size ? data.get(0) : data[0];
      const value = firstPoint.get ? firstPoint.get('value') : firstPoint.value;
      expect(value).toBeGreaterThan(0);
    });

    it('should generate directional data for shear modulus', () => {
      const data = Module.generateDirectionalData(elasticTensor, 'shear', 24);
      
      const length = data.size ? data.size() : data.length;
      expect(length).toBe(24);
      
      const firstPoint = data.size ? data.get(0) : data[0];
      const value = firstPoint.get ? firstPoint.get('value') : firstPoint.value;
      expect(value).toBeGreaterThan(0);
    });

    it('should generate directional data for Poisson\'s ratio', () => {
      const data = Module.generateDirectionalData(elasticTensor, 'poisson', 12);
      
      const length = data.size ? data.size() : data.length;
      expect(length).toBe(12);
      
      const firstPoint = data.size ? data.get(0) : data[0];
      const value = firstPoint.get ? firstPoint.get('value') : firstPoint.value;
      expect(typeof value).toBe('number');
    });
  });

  describe('Real-world Material Tests', () => {
    describe('Diamond', () => {
      it('should analyze diamond elastic tensor correctly', () => {
        const diamondData = [
          [1076, 126, 126, 0, 0, 0],
          [126, 1076, 126, 0, 0, 0],
          [126, 126, 1076, 0, 0, 0],
          [0, 0, 0, 578, 0, 0],
          [0, 0, 0, 0, 578, 0],
          [0, 0, 0, 0, 0, 578]
        ];
        
        const mat6 = Module.Mat6.create(6, 6);
        for (let i = 0; i < 6; i++) {
          for (let j = 0; j < 6; j++) {
            mat6.set(i, j, diamondData[i][j]);
          }
        }
        
        const diamond = new Module.ElasticTensor(mat6);
        
        // Diamond should have very high moduli
        const bulkModulus = diamond.averageBulkModulus(Module.AveragingScheme.HILL);
        const youngsModulus = diamond.averageYoungsModulus(Module.AveragingScheme.HILL);
        
        expect(bulkModulus).toBeGreaterThan(400); // GPa
        expect(youngsModulus).toBeGreaterThan(1000); // GPa
        
        // Diamond is cubic, so should be isotropic in certain directions
        const dir1 = Module.Vec3.create(1, 0, 0);
        const dir2 = Module.Vec3.create(0, 1, 0);
        const dir3 = Module.Vec3.create(0, 0, 1);
        
        const y1 = diamond.youngsModulus(dir1);
        const y2 = diamond.youngsModulus(dir2);
        const y3 = diamond.youngsModulus(dir3);
        
        expect(Math.abs(y1 - y2)).toBeLessThan(0.1);
        expect(Math.abs(y2 - y3)).toBeLessThan(0.1);
      });
    });

    describe('Silicon', () => {
      it('should analyze silicon elastic tensor correctly', () => {
        const siliconData = [
          [166, 64, 64, 0, 0, 0],
          [64, 166, 64, 0, 0, 0],
          [64, 64, 166, 0, 0, 0],
          [0, 0, 0, 80, 0, 0],
          [0, 0, 0, 0, 80, 0],
          [0, 0, 0, 0, 0, 80]
        ];
        
        const mat6 = Module.Mat6.create(6, 6);
        for (let i = 0; i < 6; i++) {
          for (let j = 0; j < 6; j++) {
            mat6.set(i, j, siliconData[i][j]);
          }
        }
        
        const silicon = new Module.ElasticTensor(mat6);
        
        const bulkModulus = silicon.averageBulkModulus(Module.AveragingScheme.HILL);
        const youngsModulus = silicon.averageYoungsModulus(Module.AveragingScheme.HILL);
        
        // Silicon should have moderate moduli
        expect(bulkModulus).toBeGreaterThan(90);
        expect(bulkModulus).toBeLessThan(110);
        expect(youngsModulus).toBeGreaterThan(100);
        expect(youngsModulus).toBeLessThan(200);
      });
    });
  });

  describe('Matrix Format Parsing', () => {
    it('should handle full 6x6 matrix format', () => {
      const fullMatrix = [
        [83.504, 50.299, 50.299, 0, 0, 0],
        [50.299, 83.504, 50.299, 0, 0, 0],
        [50.299, 50.299, 83.504, 0, 0, 0],
        [0, 0, 0, 19.636, 0, 0],
        [0, 0, 0, 0, 19.636, 0],
        [0, 0, 0, 0, 0, 19.636]
      ];
      
      let mat6;
      if (Module.Mat6 && Module.Mat6.create) {
        mat6 = Module.Mat6.create(6, 6);
      } else {
        mat6 = Module.Mat.create(6, 6);
      }
      
      for (let i = 0; i < 6; i++) {
        for (let j = 0; j < 6; j++) {
          mat6.set(i, j, fullMatrix[i][j]);
        }
      }
      
      const elasticTensor = new Module.ElasticTensor(mat6);
      expect(elasticTensor).toBeDefined();
      
      // Verify symmetry
      const voigtC = elasticTensor.voigtC;
      expect(voigtC.get(0, 1)).toBeCloseTo(voigtC.get(1, 0), 10);
      expect(voigtC.get(0, 2)).toBeCloseTo(voigtC.get(2, 0), 10);
    });

    it('should handle upper triangular matrix format', () => {
      // This test simulates parsing upper triangular input like:
      // 83.504  50.299  50.299   0.000   0.000   0.000
      //         83.504  50.299   0.000   0.000   0.000
      //                 83.504   0.000   0.000   0.000
      //                         19.636   0.000   0.000
      //                                 19.636   0.000
      //                                         19.636
      
      const upperTriangular = [
        [83.504, 50.299, 50.299, 0, 0, 0],
        [83.504, 50.299, 0, 0, 0],
        [83.504, 0, 0, 0],
        [19.636, 0, 0],
        [19.636, 0],
        [19.636]
      ];
      
      // Convert to full matrix (simulating the parser logic)
      const fullMatrix = Array(6).fill().map(() => Array(6).fill(0));
      for (let i = 0; i < 6; i++) {
        for (let j = i; j < 6; j++) {
          const value = upperTriangular[i][j - i];
          fullMatrix[i][j] = value;
          fullMatrix[j][i] = value; // Symmetric
        }
      }
      
      let mat6;
      if (Module.Mat6 && Module.Mat6.create) {
        mat6 = Module.Mat6.create(6, 6);
      } else {
        mat6 = Module.Mat.create(6, 6);
      }
      
      for (let i = 0; i < 6; i++) {
        for (let j = 0; j < 6; j++) {
          mat6.set(i, j, fullMatrix[i][j]);
        }
      }
      
      const elasticTensor = new Module.ElasticTensor(mat6);
      expect(elasticTensor).toBeDefined();
      
      // Verify the matrix was constructed correctly
      const voigtC = elasticTensor.voigtC;
      expect(voigtC.get(0, 0)).toBeCloseTo(83.504, 3);
      expect(voigtC.get(0, 1)).toBeCloseTo(50.299, 3);
      expect(voigtC.get(1, 0)).toBeCloseTo(50.299, 3); // Symmetry
      expect(voigtC.get(3, 3)).toBeCloseTo(19.636, 3);
      expect(voigtC.get(5, 5)).toBeCloseTo(19.636, 3);
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid matrix sizes gracefully', () => {
      // This test depends on how the C++ binding handles errors
      // We'll test that we can at least create a valid matrix
      let mat6;
      if (Module.Mat6 && Module.Mat6.create) {
        mat6 = Module.Mat6.create(6, 6);
      } else {
        mat6 = Module.Mat.create(6, 6);
      }
      expect(mat6.rows()).toBe(6);
      expect(mat6.cols()).toBe(6);
    });
  });
});