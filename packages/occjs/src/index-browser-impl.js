/**
 * Browser-compatible OCC JavaScript bindings implementation
 * This is the actual implementation without Node.js dependencies
 */

import { loadOCC, getModule } from './module-loader.js';

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

// Import core QM functionality 
import {
  QMCalculation,
  SCFSettings,
  createQMCalculation as createQMCalc,
  createHFCalculation,
  createDFTCalculation,
  createMP2Calculation,
  createHFMP2Workflow,
  createDFTWorkflow,
  quickCalculation
} from './qm/index.js';

// Import DMA functionality
import {
  calculateDMA,
  generatePunchFile,
  DMAConfig,
  DMAResult
} from './dma.js';

/**
 * Main QM calculation factory with module loading
 * @param {Object} molecule - Molecule object
 * @param {string} basisName - Basis set name
 * @param {Object} options - Calculation options
 * @returns {Promise<QMCalculation>} QM calculation object
 */
export async function createQMCalculation(molecule, basisName, options = {}) {
  const Module = await loadOCC();
  return createQMCalc(molecule, basisName, options, Module);
}

/**
 * Alias for loadOCC to match common usage patterns
 * @param {Object} options - Configuration options
 * @returns {Promise<Object>} The initialized OCC module
 */
export const createOCC = loadOCC;

/**
 * Load basis set using preloaded data
 * @param {Object} molecule - Molecule object
 * @param {string} basisName - Basis set name
 * @returns {Promise<Object>} AOBasis object
 */
export async function loadBasisSet(molecule, basisName) {
  const Module = await loadOCC();
  return Module.AOBasis.load(molecule.atoms(), basisName);
}

// getModule is now imported from module-loader.js

/**
 * Helper function to create a wavefunction from string content
 * @param {string} content - File content (FCHK or Molden format)
 * @param {string} format - Format type ("fchk" or "molden")
 * @returns {Promise<Object>} Wavefunction object
 */
export async function wavefunctionFromString(content, format) {
  const Module = await loadOCC();
  return Module.Wavefunction.fromString(content, format);
}


// Re-export all bindings from the WASM module
export * from './occjs.js';

// Export the main functions and constants (including our convenience methods)
export {
  // Core module loading
  loadOCC,
  getModule,
  // Molecule utilities
  moleculeFromXYZ,
  createMolecule,
  Elements,
  BasisSets,
  // Core QM functionality
  QMCalculation,
  SCFSettings,
  // Calculation factories
  createHFCalculation,
  createDFTCalculation,
  createMP2Calculation,
  createHFMP2Workflow,
  createDFTWorkflow,
  quickCalculation,
  // DMA functionality
  calculateDMA,
  generatePunchFile,
  DMAConfig,
  DMAResult
};
