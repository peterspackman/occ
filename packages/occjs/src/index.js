/**
 * OCC JavaScript/WebAssembly bindings
 * Main entry point for the package
 */

const path = require('path');
const fs = require('fs');

// Cache for the loaded module
let moduleInstance = null;
let modulePromise = null;

/**
 * Load and initialize the OCC WASM module
 * @param {Object} options - Configuration options
 * @param {string} options.wasmPath - Custom path to the WASM file
 * @param {Object} options.env - Environment variables to pass to the module
 * @returns {Promise<Object>} The initialized OCC module
 */
async function loadOCC(options = {}) {
  // Return cached instance if already loaded
  if (moduleInstance) {
    return moduleInstance;
  }

  // Return existing promise if loading is in progress
  if (modulePromise) {
    return modulePromise;
  }

  modulePromise = (async () => {
    try {
      // Determine WASM file path
      const wasmPath = options.wasmPath || path.join(__dirname, 'occjs.wasm');
      
      // Load the module factory
      const createModule = require('./occjs.js');
      
      // Handle different module formats
      const moduleFactory = typeof createModule === 'function' 
        ? createModule 
        : (createModule.default || createModule);
      
      // Initialize the module with options
      const Module = await moduleFactory({
        locateFile: (filename) => {
          if (filename.endsWith('.wasm')) {
            return wasmPath;
          }
          return filename;
        },
        ...options.env
      });

      // Set default log level to reduce noise
      if (Module.LogLevel && Module.setLogLevel) {
        Module.setLogLevel(Module.LogLevel.WARN || 3);
      }

      moduleInstance = Module;
      return Module;
    } catch (error) {
      modulePromise = null; // Reset on error to allow retry
      throw new Error(`Failed to load OCC module: ${error.message}`);
    }
  })();

  return modulePromise;
}

/**
 * Helper function to create a molecule from XYZ string
 * @param {string} xyzString - XYZ format molecular structure
 * @returns {Promise<Object>} Molecule object
 */
async function moleculeFromXYZ(xyzString) {
  const Module = await loadOCC();
  return Module.Molecule.fromXyzString(xyzString);
}

/**
 * Helper function to create a molecule from atomic numbers and positions
 * @param {number[]} atomicNumbers - Array of atomic numbers
 * @param {number[][]} positions - Array of [x, y, z] coordinates
 * @returns {Promise<Object>} Molecule object
 */
async function createMolecule(atomicNumbers, positions) {
  const Module = await loadOCC();
  
  const numAtoms = atomicNumbers.length;
  const positionMatrix = Module.Mat3N.create(numAtoms);
  
  for (let i = 0; i < numAtoms; i++) {
    positionMatrix.set(0, i, positions[i][0]);
    positionMatrix.set(1, i, positions[i][1]);
    positionMatrix.set(2, i, positions[i][2]);
  }
  
  const atomicNumbersVec = Module.IVec.fromArray(atomicNumbers);
  return new Module.Molecule(atomicNumbersVec, positionMatrix);
}

/**
 * Constants for common elements
 */
const Elements = {
  H: 1, He: 2, Li: 3, Be: 4, B: 5, C: 6, N: 7, O: 8, F: 9, Ne: 10,
  Na: 11, Mg: 12, Al: 13, Si: 14, P: 15, S: 16, Cl: 17, Ar: 18,
  K: 19, Ca: 20, Sc: 21, Ti: 22, V: 23, Cr: 24, Mn: 25, Fe: 26,
  Co: 27, Ni: 28, Cu: 29, Zn: 30, Ga: 31, Ge: 32, As: 33, Se: 34,
  Br: 35, Kr: 36, Rb: 37, Sr: 38, Y: 39, Zr: 40, Nb: 41, Mo: 42,
  Tc: 43, Ru: 44, Rh: 45, Pd: 46, Ag: 47, Cd: 48, In: 49, Sn: 50,
  Sb: 51, Te: 52, I: 53, Xe: 54, Cs: 55, Ba: 56
};

/**
 * Common basis sets
 */
const BasisSets = {
  'sto-3g': 'sto-3g',
  '3-21g': '3-21g',
  '6-31g': '6-31g',
  '6-31g(d)': '6-31g(d)',
  '6-31g(d,p)': '6-31g(d,p)',
  '6-311g(d,p)': '6-311g(d,p)',
  'def2-svp': 'def2-svp',
  'def2-tzvp': 'def2-tzvp',
  'def2-qzvp': 'def2-qzvp',
  'cc-pvdz': 'cc-pvdz',
  'cc-pvtz': 'cc-pvtz',
  'cc-pvqz': 'cc-pvqz'
};

// Export the main functions and constants
module.exports = {
  loadOCC,
  moleculeFromXYZ,
  createMolecule,
  Elements,
  BasisSets,
  
  // Re-export some common classes/functions after loading
  get Module() {
    if (!moduleInstance) {
      throw new Error('OCC module not loaded. Call loadOCC() first.');
    }
    return moduleInstance;
  }
};