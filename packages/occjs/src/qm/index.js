/**
 * Quantum Chemistry module for OCC JavaScript bindings
 * 
 * This module provides quantum chemistry functionality with preloaded basis sets
 * and a simplified interface for custom basis set loading.
 */

// Core QM calculation functionality
export { 
  QMCalculation, 
  SCFSettings, 
  createQMCalculation,
  loadBasisSet
} from './core.js';

// Simplified basis set loading utilities
export { 
  loadBasisFromJSON,
  addCustomBasisSet,
  listAvailableBasisSets,
  hasBasisSet
} from './SimpleBasisLoader.js';

// Calculation factory functions
export {
  createHFCalculation,
  createDFTCalculation,
  createMP2Calculation,
  createHFMP2Workflow,
  createDFTWorkflow,
  quickCalculation
} from './methods/factory.js';