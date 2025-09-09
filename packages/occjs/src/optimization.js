/**
 * High-level optimization utilities for OCC.js
 * 
 * This module provides convenient functions for molecular geometry optimization
 * and vibrational analysis, similar to the Python interface.
 */

/**
 * Perform geometry optimization using Hartree-Fock method
 * @param {Object} module - The loaded OCC.js module
 * @param {Object} molecule - Initial molecule geometry 
 * @param {string} basisName - Basis set name (e.g., "3-21G", "STO-3G")
 * @param {Object} options - Optimization options
 * @param {Object} options.criteria - Convergence criteria
 * @param {number} options.maxSteps - Maximum optimization steps
 * @param {function} options.onStep - Callback for each step
 * @returns {Promise<Object>} Optimization result
 */
export async function optimizeHF(module, molecule, basisName = "3-21G", options = {}) {
  const {
    criteria = null,
    maxSteps = 25,
    onStep = null
  } = options;

  // Set up default convergence criteria if not provided
  const optCriteria = criteria || (() => {
    const c = new module.ConvergenceCriteria();
    c.gradientMax = 1e-4;
    c.gradientRms = 1e-5;
    c.stepMax = 1e-3;
    c.stepRms = 1e-4;
    return c;
  })();

  // Create optimizer
  const optimizer = new module.BernyOptimizer(molecule, optCriteria);
  
  // Storage for optimization trajectory
  const trajectory = {
    energies: [],
    gradientNorms: [],
    geometries: [],
    converged: false,
    steps: 0,
    finalEnergy: null,
    finalMolecule: null
  };

  let converged = false;

  for (let step = 0; step < maxSteps; step++) {
    // Get current geometry
    const currentMol = optimizer.getNextGeometry();
    trajectory.geometries.push(currentMol);

    // Create calculation for current geometry
    const basis = module.AOBasis.load(currentMol.atoms(), basisName);
    const hf = new module.HartreeFock(basis);

    // Run SCF calculation
    const scf = new module.HartreeFockSCF(hf);
    scf.setChargeMultiplicity(0, 1); // Neutral singlet
    const energy = await scf.run();

    // Compute gradient
    const wfn = scf.wavefunction();
    const gradient = hf.computeGradient(wfn.molecularOrbitals);

    // Store progress
    trajectory.energies.push(energy);
    const gradNorm = Math.sqrt(gradient.squaredNorm());
    trajectory.gradientNorms.push(gradNorm);

    // Update optimizer
    optimizer.update(energy, gradient);

    // Call user callback if provided
    if (onStep) {
      onStep({
        step: step + 1,
        energy,
        gradientNorm: gradNorm,
        molecule: currentMol,
        optimizer
      });
    }

    // Check convergence
    if (optimizer.step()) {
      converged = true;
      trajectory.converged = true;
      trajectory.steps = step + 1;
      break;
    }
  }

  // Get final results
  trajectory.finalMolecule = optimizer.getNextGeometry();
  trajectory.finalEnergy = optimizer.currentEnergy();
  trajectory.converged = converged;
  trajectory.steps = converged ? trajectory.steps : maxSteps;

  return trajectory;
}

/**
 * Perform geometry optimization using DFT method
 * @param {Object} module - The loaded OCC.js module
 * @param {Object} molecule - Initial molecule geometry 
 * @param {string} functional - DFT functional (e.g., "b3lyp", "pbe")
 * @param {string} basisName - Basis set name
 * @param {Object} options - Optimization options
 * @returns {Promise<Object>} Optimization result
 */
export async function optimizeDFT(module, molecule, functional = "b3lyp", basisName = "3-21G", options = {}) {
  const {
    criteria = null,
    maxSteps = 25,
    onStep = null
  } = options;

  // Set up default convergence criteria if not provided
  const optCriteria = criteria || (() => {
    const c = new module.ConvergenceCriteria();
    c.gradientMax = 1e-4;
    c.gradientRms = 1e-5;
    c.stepMax = 1e-3;
    c.stepRms = 1e-4;
    return c;
  })();

  // Create optimizer
  const optimizer = new module.BernyOptimizer(molecule, optCriteria);
  
  // Storage for optimization trajectory
  const trajectory = {
    energies: [],
    gradientNorms: [],
    geometries: [],
    converged: false,
    steps: 0,
    finalEnergy: null,
    finalMolecule: null
  };

  let converged = false;

  for (let step = 0; step < maxSteps; step++) {
    // Get current geometry
    const currentMol = optimizer.getNextGeometry();
    trajectory.geometries.push(currentMol);

    // Create calculation for current geometry
    const basis = module.AOBasis.load(currentMol.atoms(), basisName);
    const dft = new module.DFT(functional, basis);

    // Run SCF calculation
    const scf = new module.KohnShamSCF(dft);
    scf.setChargeMultiplicity(0, 1); // Neutral singlet
    const energy = await scf.run();

    // Compute gradient
    const wfn = scf.wavefunction();
    const gradient = dft.computeGradient(wfn.molecularOrbitals);

    // Store progress
    trajectory.energies.push(energy);
    const gradNorm = Math.sqrt(gradient.squaredNorm());
    trajectory.gradientNorms.push(gradNorm);

    // Update optimizer
    optimizer.update(energy, gradient);

    // Call user callback if provided
    if (onStep) {
      onStep({
        step: step + 1,
        energy,
        gradientNorm: gradNorm,
        molecule: currentMol,
        optimizer
      });
    }

    // Check convergence
    if (optimizer.step()) {
      converged = true;
      trajectory.converged = true;
      trajectory.steps = step + 1;
      break;
    }
  }

  // Get final results
  trajectory.finalMolecule = optimizer.getNextGeometry();
  trajectory.finalEnergy = optimizer.currentEnergy();
  trajectory.converged = converged;
  trajectory.steps = converged ? trajectory.steps : maxSteps;

  return trajectory;
}

/**
 * Compute vibrational frequencies at optimized geometry
 * @param {Object} module - The loaded OCC.js module
 * @param {Object} molecule - Optimized molecule
 * @param {string} method - Method type ("HF" or "DFT")
 * @param {string} functional - DFT functional (ignored for HF)
 * @param {string} basisName - Basis set name
 * @param {Object} options - Frequency calculation options
 * @returns {Promise<Object>} Vibrational analysis result
 */
export async function computeFrequencies(module, molecule, method = "HF", functional = "b3lyp", basisName = "3-21G", options = {}) {
  const {
    stepSize = 0.005,
    useAcousticSumRule = true,
    projectTransRot = true
  } = options;

  // Set up calculation at optimized geometry
  const basis = module.AOBasis.load(molecule.atoms(), basisName);
  
  let calculator, scf, hessEvaluator;
  
  let scfEnergy;
  if (method.toUpperCase() === "HF") {
    calculator = new module.HartreeFock(basis);
    scf = new module.HartreeFockSCF(calculator);
    scf.setChargeMultiplicity(0, 1);
    scfEnergy = await scf.run();
    
    hessEvaluator = calculator.hessianEvaluator();
  } else if (method.toUpperCase() === "DFT") {
    calculator = new module.DFT(functional, basis);
    scf = new module.KohnShamSCF(calculator);
    scf.setChargeMultiplicity(0, 1);
    scfEnergy = await scf.run();
    
    hessEvaluator = calculator.hessianEvaluator();
  } else {
    throw new Error(`Unknown method: ${method}. Use "HF" or "DFT"`);
  }

  // Configure Hessian evaluator
  hessEvaluator.setStepSize(stepSize);
  hessEvaluator.setUseAcousticSumRule(useAcousticSumRule);

  // Compute Hessian
  const wfn = scf.wavefunction();
  const hessian = hessEvaluator.compute(wfn.molecularOrbitals);

  // Compute vibrational modes
  const vibrationalModes = module.computeVibrationalModesFromMolecule(hessian, molecule, projectTransRot);

  // Extract frequency data
  const frequencies = vibrationalModes.getAllFrequencies();
  const freqArray = [];
  for (let i = 0; i < frequencies.size(); i++) {
    freqArray.push(frequencies.get(i));
  }

  return {
    modes: vibrationalModes,
    frequencies: freqArray,
    nModes: vibrationalModes.nModes(),
    nAtoms: vibrationalModes.nAtoms(),
    summary: vibrationalModes.summaryString(),
    frequenciesString: vibrationalModes.frequenciesString(),
    scfEnergy: scfEnergy
  };
}

/**
 * Complete workflow: optimize geometry and compute frequencies
 * @param {Object} module - The loaded OCC.js module
 * @param {Object} molecule - Initial molecule geometry
 * @param {Object} options - Combined optimization and frequency options
 * @returns {Promise<Object>} Complete result with optimization and frequencies
 */
export async function optimizeAndAnalyze(module, molecule, options = {}) {
  const {
    method = "HF",
    functional = "b3lyp",
    basisName = "3-21G",
    optimization = {},
    frequencies = {}
  } = options;

  // First perform optimization
  let optimizationResult;
  if (method.toUpperCase() === "HF") {
    optimizationResult = await optimizeHF(module, molecule, basisName, optimization);
  } else if (method.toUpperCase() === "DFT") {
    optimizationResult = await optimizeDFT(module, molecule, functional, basisName, optimization);
  } else {
    throw new Error(`Unknown method: ${method}. Use "HF" or "DFT"`);
  }

  if (!optimizationResult.converged) {
    console.warn("Optimization did not converge - frequencies may not be meaningful");
  }

  // Then compute frequencies at optimized geometry
  const frequencyResult = await computeFrequencies(
    module, 
    optimizationResult.finalMolecule, 
    method, 
    functional, 
    basisName, 
    frequencies
  );

  return {
    optimization: optimizationResult,
    frequencies: frequencyResult,
    finalMolecule: optimizationResult.finalMolecule,
    finalEnergy: optimizationResult.finalEnergy,
    converged: optimizationResult.converged
  };
}

/**
 * Export molecule to XYZ format with optional comment
 * @param {Object} module - The loaded OCC.js module
 * @param {Object} molecule - Molecule to export
 * @param {string} comment - Optional comment line
 * @returns {string} XYZ format string
 */
export function moleculeToXYZ(module, molecule, comment = "") {
  if (comment) {
    return module.moleculeToXYZWithComment(molecule, comment);
  } else {
    return module.moleculeToXYZ(molecule);
  }
}

export default {
  optimizeHF,
  optimizeDFT,
  computeFrequencies,
  optimizeAndAnalyze,
  moleculeToXYZ
};