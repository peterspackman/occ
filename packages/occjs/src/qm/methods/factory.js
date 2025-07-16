import { createQMCalculation, SCFSettings } from '../core.js';

/**
 * Create a Hartree-Fock calculation with optimized settings
 * @param {Object} molecule - Molecule object
 * @param {string} basisName - Basis set name  
 * @param {Object} options - Calculation options
 * @param {Object} module - OCC module
 * @returns {Promise<QMCalculation>} QM calculation configured for HF
 */
export async function createHFCalculation(molecule, basisName, options = {}, module) {
  const calc = await createQMCalculation(molecule, basisName, options, module);
  
  // Set up HF-specific default settings
  const scfSettings = new SCFSettings()
    .setMaxIterations(options.maxIterations || 100)
    .setEnergyTolerance(options.energyTolerance || 1e-8)
    .setDensityTolerance(options.densityTolerance || 1e-6)
    .setInitialGuess(options.initialGuess || 'core')
    .setDIIS(options.diis !== false, options.diisSize || 8);
  
  // Store settings for automatic use
  calc._defaultHFSettings = scfSettings;
  
  // Add convenience method
  calc.runHFWithDefaults = function() {
    return this.runHF(this._defaultHFSettings);
  };
  
  return calc;
}

/**
 * Create a DFT calculation with optimized settings
 * @param {Object} molecule - Molecule object
 * @param {string} basisName - Basis set name
 * @param {string} functional - DFT functional name
 * @param {Object} options - Calculation options
 * @param {Object} module - OCC module
 * @returns {Promise<QMCalculation>} QM calculation configured for DFT
 */
export async function createDFTCalculation(molecule, basisName, functional, options = {}, module) {
  // Let C++ handle functional validation
  const calc = await createQMCalculation(molecule, basisName, options, module);
  
  // Set up DFT-specific default settings
  const scfSettings = new SCFSettings()
    .setMaxIterations(options.maxIterations || 150) // DFT often needs more iterations
    .setEnergyTolerance(options.energyTolerance || 1e-8)
    .setDensityTolerance(options.densityTolerance || 1e-6)
    .setInitialGuess(options.initialGuess || 'core')
    .setDIIS(options.diis !== false, options.diisSize || 8);
  
  const dftOptions = {
    scfSettings: scfSettings,
    densityFittingBasis: options.densityFittingBasis,
    gridSettings: options.gridSettings,
    ...options.dftOptions
  };
  
  // Store settings for automatic use
  calc._defaultFunctional = functional;
  calc._defaultDFTOptions = dftOptions;
  
  // Add convenience method
  calc.runDFTWithDefaults = function() {
    return this.runDFT(this._defaultFunctional, this._defaultDFTOptions);
  };
  
  return calc;
}

/**
 * Create an MP2 calculation (requires a reference calculation first)
 * @param {QMCalculation} referenceCalc - Reference calculation (HF or DFT)
 * @param {Object} options - MP2 options
 * @returns {QMCalculation} Same calculation object with MP2 settings
 */
export function createMP2Calculation(referenceCalc, options = {}) {
  if (!referenceCalc.wavefunction) {
    throw new Error('MP2 calculation requires a reference wavefunction. Run HF or DFT on the reference calculation first.');
  }
  
  // Set up MP2-specific default settings
  const mp2Options = {
    frozenCore: options.frozenCore !== undefined ? options.frozenCore : true,
    useRI: options.useRI !== undefined ? options.useRI : false,
    spinComponentScaling: options.spinComponentScaling,
    ...options
  };
  
  // Store settings for automatic use
  referenceCalc._defaultMP2Options = mp2Options;
  
  // Add convenience method
  referenceCalc.runMP2WithDefaults = function() {
    return this.runMP2(this._defaultMP2Options);
  };
  
  return referenceCalc;
}

/**
 * Create a complete calculation workflow (HF -> MP2)
 * @param {Object} molecule - Molecule object
 * @param {string} basisName - Basis set name
 * @param {Object} options - Workflow options
 * @param {Object} module - OCC module
 * @returns {Promise<Object>} Object with calc and results
 */
export async function createHFMP2Workflow(molecule, basisName, options = {}, module) {
  const calc = await createHFCalculation(molecule, basisName, options, module);
  
  // Run HF
  const hfEnergy = await calc.runHFWithDefaults();
  
  // Configure and run MP2
  createMP2Calculation(calc, options.mp2Options);
  const mp2Energy = await calc.runMP2WithDefaults();
  
  return {
    calculation: calc,
    energies: {
      hf: hfEnergy,
      mp2: mp2Energy
    },
    summary: calc.getSummary()
  };
}

/**
 * Create a complete DFT calculation workflow  
 * @param {Object} molecule - Molecule object
 * @param {string} basisName - Basis set name
 * @param {string} functional - DFT functional
 * @param {Object} options - Workflow options
 * @param {Object} module - OCC module
 * @returns {Promise<Object>} Object with calc and results
 */
export async function createDFTWorkflow(molecule, basisName, functional, options = {}, module) {
  const calc = await createDFTCalculation(molecule, basisName, functional, options, module);
  
  // Run DFT
  const dftEnergy = await calc.runDFTWithDefaults();
  
  // Calculate properties if requested
  let properties = {};
  if (options.properties && Array.isArray(options.properties)) {
    properties = await calc.calculateProperties(options.properties);
  }
  
  return {
    calculation: calc,
    energy: dftEnergy,
    properties: properties,
    summary: calc.getSummary()
  };
}

/**
 * Quick calculation helper for common use cases
 * @param {Object} molecule - Molecule object
 * @param {Object} options - Calculation options
 * @param {Object} module - OCC module
 * @returns {Promise<Object>} Calculation results
 */
export async function quickCalculation(molecule, options = {}, module) {
  const {
    method = 'HF',
    basis = 'sto-3g',
    functional = 'b3lyp',
    properties = ['energy', 'mulliken', 'dipole'],
    ...otherOptions
  } = options;
  
  let calc;
  let energy;
  
  switch (method.toUpperCase()) {
    case 'HF':
      calc = await createHFCalculation(molecule, basis, otherOptions, module);
      energy = await calc.runHFWithDefaults();
      break;
      
    case 'DFT':
      calc = await createDFTCalculation(molecule, basis, functional, otherOptions, module);
      energy = await calc.runDFTWithDefaults();
      break;
      
    case 'MP2':
      calc = await createHFCalculation(molecule, basis, otherOptions, module);
      await calc.runHFWithDefaults();
      createMP2Calculation(calc, otherOptions.mp2Options);
      energy = await calc.runMP2WithDefaults();
      break;
      
    default:
      throw new Error(`Unknown method: ${method}. Use 'HF', 'DFT', or 'MP2'.`);
  }
  
  // Calculate properties
  const calculatedProperties = await calc.calculateProperties(properties);
  
  return {
    method: method,
    energy: energy,
    properties: calculatedProperties,
    calculation: calc
  };
}