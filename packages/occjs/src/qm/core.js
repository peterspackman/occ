/**
 * Core quantum chemistry classes and utilities
 */

/**
 * SCF convergence settings
 */
export class SCFSettings {
  constructor() {
    this.maxIterations = 100;
    this.energyTolerance = 1e-8;
    this.densityTolerance = 1e-6;
    this.initialGuess = 'core';
    this.diis = true;
    this.diisSize = 8;
  }

  setMaxIterations(max) {
    this.maxIterations = max;
    return this;
  }

  setEnergyTolerance(tol) {
    this.energyTolerance = tol;
    return this;
  }

  setDensityTolerance(tol) {
    this.densityTolerance = tol;
    return this;
  }

  setInitialGuess(guess) {
    this.initialGuess = guess;
    return this;
  }

  setDIIS(enabled, size = 8) {
    this.diis = enabled;
    this.diisSize = size;
    return this;
  }
}


/**
 * Quantum chemistry calculation wrapper
 */
export class QMCalculation {
  constructor(molecule, basis, module) {
    this.molecule = molecule;
    this.basis = basis;
    this.module = module;
    this.wavefunction = null;
    this.energy = null;
    this.method = null;
    this.properties = new Map();
    // Store C++ objects to prevent premature garbage collection
    this._cppProcedure = null;  // HartreeFock or DFT object
    this._cppScf = null;  // SCF object
  }

  /**
   * Run Hartree-Fock SCF calculation
   * @param {SCFSettings|Object} settings - SCF settings
   * @returns {Promise<number>} SCF energy
   */
  async runHF(settings = {}) {
    const scfSettings = settings instanceof SCFSettings ? settings : new SCFSettings();
    if (!(settings instanceof SCFSettings)) {
      Object.assign(scfSettings, settings);
    }

    const hf = new this.module.HartreeFock(this.basis);
    
    // Configure precision if specified
    if (settings.precision) {
      hf.setPrecision(settings.precision);
    }

    // Create SCF procedure
    const spinKind = settings.unrestricted ? 
      this.module.SpinorbitalKind.Unrestricted : 
      this.module.SpinorbitalKind.Restricted;
    
    const scf = new this.module.HartreeFockSCF(hf, spinKind);
    
    // Set charge and multiplicity
    scf.setChargeMultiplicity(this.molecule.charge(), this.molecule.multiplicity());
    
    // Configure SCF convergence settings
    const convergenceSettings = scf.convergenceSettings;
    if (scfSettings.energyTolerance) {
      convergenceSettings.energyThreshold = scfSettings.energyTolerance;
    }
    if (scfSettings.commutatorTolerance) {
      convergenceSettings.commutatorThreshold = scfSettings.commutatorTolerance;
    }
    
    this.energy = scf.run();
    this.wavefunction = scf.wavefunction();
    this.method = 'HF';
    
    return this.energy;
  }

  /**
   * Run DFT calculation
   * @param {string} functional - DFT functional name
   * @param {Object} options - DFT options including SCF settings
   * @returns {Promise<number>} DFT energy
   */
  async runDFT(functional, options = {}) {
    // Let C++ handle functional validation - just pass the name directly
    const dft = new this.module.DFT(functional, this.basis);
    
    // Configure precision if specified
    if (options.precision) {
      dft.setPrecision(options.precision);
    }

    // Create SCF procedure
    const spinKind = options.unrestricted ? 
      this.module.SpinorbitalKind.Unrestricted : 
      this.module.SpinorbitalKind.Restricted;
    
    const scf = new this.module.KohnShamSCF(dft, spinKind);
    
    // Set charge and multiplicity
    scf.setChargeMultiplicity(this.molecule.charge(), this.molecule.multiplicity());
    
    // Configure SCF convergence settings
    if (options.scfSettings) {
      const convergenceSettings = scf.convergenceSettings;
      if (options.scfSettings.energyTolerance) {
        convergenceSettings.energyThreshold = options.scfSettings.energyTolerance;
      }
    }
    
    this.energy = scf.run();
    this.wavefunction = scf.wavefunction();
    this.method = `DFT/${functional}`;  // Use the functional name as provided
    
    return this.energy;
  }

  /**
   * Run a post-HF correlation calculation (MP2 / CCSD / CCSD(T)) on the
   * converged SCF wavefunction. Backend and auxiliary basis are resolved
   * exactly like the occ CLI.
   * @param {string} method - e.g. 'mp2', 'ri-mp2', 'ccsd', 'ccsd(t)'
   * @param {Object} options - optional overrides: backend, auxBasis,
   *   spinScaling, nFrozen, maxMemoryGb, maxCycle, tol
   * @returns {Promise<Object>} correlation result (totalEnergy,
   *   correlationEnergy, triplesCorrection, ...)
   */
  async runCorrelation(method = 'mp2', options = {}) {
    if (!this.wavefunction) {
      throw new Error('Correlation methods require a reference wavefunction. Run HF or DFT first.');
    }

    const opts = new this.module.CorrelationOptions();
    opts.method = method;
    if (options.backend !== undefined) {
      opts.backend = options.backend;
    }
    if (options.auxBasis !== undefined) {
      opts.auxBasis = options.auxBasis;
    }
    if (options.spinScaling !== undefined) {
      opts.spinScaling = options.spinScaling;
    }
    if (options.nFrozen !== undefined) {
      opts.nFrozen = options.nFrozen;
    }
    if (options.maxMemoryGb !== undefined) {
      opts.maxMemoryGb = options.maxMemoryGb;
    }
    if (options.maxCycle !== undefined) {
      opts.maxCycle = options.maxCycle;
    }
    if (options.tol !== undefined) {
      opts.tol = options.tol;
    }

    const result = this.module.runCorrelationWithOptions(this.wavefunction, opts);
    this.energy = result.totalEnergy;
    this.method = result.method;

    return {
      method: result.method,
      scfEnergy: result.scfEnergy,
      correlationEnergy: result.correlationEnergy,
      totalEnergy: result.totalEnergy,
      sameSpin: result.sameSpin,
      oppositeSpin: result.oppositeSpin,
      scaledCorrelation: result.scaledCorrelation,
      ccsdCorrelation: result.ccsdCorrelation,
      triplesCorrection: result.triplesCorrection,
      iterations: result.iterations,
      converged: result.converged,
      nFrozen: result.nFrozen,
    };
  }

  /**
   * Run MP2 calculation
   * @param {Object} options - MP2 options (backend, auxBasis, spinScaling, nFrozen, ...)
   * @returns {Promise<number>} MP2 total energy
   */
  async runMP2(options = {}) {
    const result = await this.runCorrelation(options.method || 'mp2', options);
    return result.totalEnergy;
  }

  /**
   * Run CCSD or CCSD(T)
   * @param {Object} options - options (triples: true for CCSD(T), backend, nFrozen, ...)
   * @returns {Promise<Object>} correlation result
   */
  async runCCSD(options = {}) {
    const method = options.triples ? 'ccsd(t)' : 'ccsd';
    return this.runCorrelation(method, options);
  }

  /**
   * Calculate molecular properties
   * @param {Array<string>} properties - List of properties to calculate
   * @returns {Promise<Object>} Calculated properties
   */
  async calculateProperties(properties) {
    if (!this.wavefunction) {
      throw new Error('Properties calculation requires a wavefunction. Run SCF first.');
    }

    const results = {};
    
    for (const prop of properties) {
      switch (prop.toLowerCase()) {
        case 'mulliken':
          results.mulliken = this.wavefunction.mullikenCharges();
          this.properties.set('mulliken', results.mulliken);
          break;
        case 'energy':
          results.energy = this.energy;
          break;
        case 'orbitals':
          results.orbitals = {
            coefficients: this.wavefunction.coefficients(),
            energies: this.wavefunction.orbitalEnergies(),
            occupations: this.wavefunction.occupations()
          };
          this.properties.set('orbitals', results.orbitals);
          break;
        case 'homo':
          results.homo = this.wavefunction.homoEnergy();
          break;
        case 'lumo':
          results.lumo = this.wavefunction.lumoEnergy();
          break;
        case 'gap':
          results.gap = this.wavefunction.lumoEnergy() - this.wavefunction.homoEnergy();
          break;
        default:
          console.warn(`Unknown property: ${prop}`);
      }
    }
    
    return results;
  }

  /**
   * Export wavefunction to various formats
   * @param {string} format - Export format ('json', 'molden')
   * @returns {string} Exported wavefunction data
   */
  exportWavefunction(format = 'molden') {
    if (!this.wavefunction) {
      throw new Error('No wavefunction to export. Run SCF calculation first.');
    }
    
    // Use the unified exportToString method that takes format as parameter
    return this.wavefunction.exportToString(format.toLowerCase());
  }

  /**
   * Get calculation summary
   * @returns {Object} Summary of calculation results
   */
  getSummary() {
    return {
      method: this.method,
      energy: this.energy,
      molecule: {
        formula: this.molecule.name(),
        natoms: this.molecule.size(),
        charge: this.molecule.charge(),
        multiplicity: this.molecule.multiplicity()
      },
      basis: this.basis ? this.basis.name() : 'Unknown',
      converged: this.energy !== null,
      properties: Object.fromEntries(this.properties)
    };
  }
}

/**
 * Load basis set for a molecule
 * @param {Object} module - OCC module
 * @param {Object} molecule - Molecule object
 * @param {string} basisName - Basis set name
 * @param {Object} options - Loading options
 * @returns {Object} Loaded basis set
 */
export function loadBasisSet(module, molecule, basisName, options = {}) {
  // If JSON basis data is provided, use fromJson method
  if (options.json) {
    const jsonString = typeof options.json === 'string' ? options.json : JSON.stringify(options.json);
    return module.AOBasis.fromJson(molecule.atoms(), jsonString);
  }
  
  // Otherwise, load by name
  return module.AOBasis.load(molecule.atoms(), basisName);
}

/**
 * Create a QM calculation object
 * @param {Object} molecule - Molecule object
 * @param {string} basisName - Basis set name
 * @param {Object} options - Creation options
 * @param {Object} module - OCC module
 * @returns {Promise<QMCalculation>} QM calculation object
 */
export function createQMCalculation(molecule, basisName, options = {}, module) {
  const basis = loadBasisSet(module, molecule, basisName, options);
  return new QMCalculation(molecule, basis, module);
}