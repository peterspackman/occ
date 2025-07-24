/**
 * Browser-compatible OCC JavaScript bindings
 * Re-exports from the main index.js without Node.js dependencies
 */

// Re-export everything from index.js except the things that would cause conflicts
export {
  // Core module loading
  loadOCC,
  createOCC,
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
  createQMCalculation,
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
  DMAResult,
  // Wavefunction utilities
  wavefunctionFromString,
  // Basis set loading
  loadBasisSet
} from './index-browser-impl.js';