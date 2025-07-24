/**
 * DMA (Distributed Multipole Analysis) JavaScript interface
 * Provides high-level access to DMA calculations similar to the CLI
 */

// DMA module now receives the module instance to avoid multiple instances

/**
 * Configuration object for DMA calculations
 */
export class DMAConfig {
  constructor() {
    this.maxRank = 4;
    this.bigExponent = 4.0;
    this.includeNuclei = true;
    this.atomRadii = new Map(); // element -> radius (Angstrom)
    this.atomLimits = new Map(); // element -> max rank
    this.writePunch = false;
    this.punchFilename = "dma.punch";
  }

  /**
   * Set radius for a specific element
   * @param {string} element - Element symbol (e.g., "H", "C")
   * @param {number} radius - Radius in Angstrom
   */
  setAtomRadius(element, radius) {
    this.atomRadii.set(element, radius);
  }

  /**
   * Set maximum rank for a specific element
   * @param {string} element - Element symbol (e.g., "H", "C")
   * @param {number} maxRank - Maximum multipole rank
   */
  setAtomLimit(element, maxRank) {
    this.atomLimits.set(element, maxRank);
  }
}

/**
 * High-level DMA calculation function
 * @param {Wavefunction} wavefunction - OCC Wavefunction object
 * @param {Object} options - DMA options
 * @param {number} options.maxRank - Maximum multipole rank (default: 4)
 * @param {number} options.bigExponent - Switch parameter (default: 4.0)
 * @param {boolean} options.includeNuclei - Include nuclear contributions (default: true)
 * @param {Object} options.atomRadii - Element-specific radii {element: radius}
 * @param {Object} options.atomLimits - Element-specific max ranks {element: rank}
 * @returns {Promise<DMAResult>} DMA calculation results
 */
export async function calculateDMA(wavefunction, options = {}) {
  // Import loadOCC dynamically to avoid circular dependencies
  // Use module-loader directly to avoid importing Node.js-specific code
  const { loadOCC } = await import('./module-loader.js');
  const Module = await loadOCC();
  
  console.log('DMA: Starting calculation with options:', options);
  
  // Create DMA config
  console.log('DMA: Creating config...');
  const config = new Module.DMAConfig();
  console.log('DMA: Config created successfully');
  
  console.log('DMA: Setting basic settings...');
  config.setMaxRank(options.maxRank ?? 4);
  config.setBigExponent(options.bigExponent ?? 4.0);
  config.setIncludeNuclei(options.includeNuclei ?? true);
  console.log(`DMA: Basic settings set - max_rank=${config.settings.max_rank}, big_exponent=${config.settings.big_exponent}`);
  
  // Set atom-specific parameters using element symbols
  if (options.atomRadii) {
    console.log('DMA: Setting atom radii...');
    for (const [element, radius] of Object.entries(options.atomRadii)) {
      console.log(`DMA: Setting radius for ${element} to ${radius}`);
      config.setAtomRadius(element, radius);
    }
    console.log('DMA: Atom radii set');
  }
  
  if (options.atomLimits) {
    console.log('DMA: Setting atom limits...');
    for (const [element, limit] of Object.entries(options.atomLimits)) {
      console.log(`DMA: Setting limit for ${element} to ${limit}`);
      config.setAtomLimit(element, limit);
    }
    console.log('DMA: Atom limits set');
  }
  
  // Set default H settings if not specified
  if (!options.atomRadii?.H) {
    console.log('DMA: Setting default H radius...');
    config.setAtomRadius("H", 0.35);
  }
  if (!options.atomLimits?.H) {
    console.log('DMA: Setting default H limit...');
    config.setAtomLimit("H", 1);
  }
  
  // Create DMA driver and compute
  console.log('DMA: Creating driver...');
  const driver = new Module.DMADriver();
  console.log('DMA: Driver created');
  
  console.log('DMA: Setting config on driver...');
  driver.set_config(config);
  console.log('DMA: Config set on driver');
  
  console.log('DMA: About to call runWithWavefunction...');
  const output = driver.runWithWavefunction(wavefunction);
  console.log('DMA: runWithWavefunction completed successfully');
  
  console.log('DMA: Creating result wrapper...');
  const result = new DMAResult(output);
  console.log('DMA: Result wrapper created');
  
  return result;
}

/**
 * Generate GDMX-compatible punch file content
 * @param {DMAResult} dmaResult - DMA calculation results
 * @returns {string} Punch file content
 */
export async function generatePunchFile(dmaResult) {
  // Import getModule dynamically to avoid circular dependencies
  // Use module-loader directly to avoid importing Node.js-specific code
  const { getModule } = await import('./module-loader.js');
  const Module = getModule();
  const { result, sites } = dmaResult;
  
  // Use the native C++ function which handles the formatting correctly
  return Module.generate_punch_file(result, sites);
}

/**
 * DMA Result wrapper class
 */
export class DMAResult {
  constructor(output) {
    this.result = output.result;
    this.sites = output.sites;
  }
  
  /**
   * Get multipole moments for a specific site
   * @param {number} siteIndex - Site index
   * @returns {Object} Multipole moment object
   */
  getSiteMultipoles(siteIndex) {
    return this.result.multipoles.get(siteIndex);
  }
  
  /**
   * Get total multipole moments
   * @returns {Object} Total multipole moments
   */
  getTotalMultipoles() {
    // Total multipoles would need to be computed separately
    // For now, return null - this can be added later if needed
    return null;
  }
  
  /**
   * Get site information
   * @returns {Object} Sites information
   */
  getSites() {
    return this.sites;
  }
  
  /**
   * Generate punch file content
   * @returns {Promise<string>} Punch file content
   */
  async toPunchFile() {
    // Import getModule dynamically to avoid circular dependencies
    // Use module-loader directly to avoid importing Node.js-specific code
    const { getModule } = await import('./module-loader.js');
    const Module = getModule();
    return Module.generate_punch_file(this.result, this.sites);
  }
  
  /**
   * Get multipole component by name for a site
   * @param {number} siteIndex - Site index
   * @param {string} component - Component name (e.g., "Q00", "Q10", "Q11c")
   * @returns {number} Multipole component value
   */
  getMultipoleComponent(siteIndex, component) {
    const mult = this.getSiteMultipoles(siteIndex);
    return mult.getComponent(component);
  }
  
  /**
   * Get all multipole components for a site as an object
   * @param {number} siteIndex - Site index
   * @returns {Object} Object with all multipole components
   */
  getAllComponents(siteIndex) {
    const mult = this.getSiteMultipoles(siteIndex);
    const components = {
      Q00: mult.getComponent('Q00'),
      charge: mult.getComponent('Q00'),
      maxRank: mult.max_rank
    };
    
    // Add components up to the multipole's max rank
    const componentNames = ['Q10', 'Q11c', 'Q11s', 'Q20', 'Q21c', 'Q21s', 'Q22c', 'Q22s',
                           'Q30', 'Q31c', 'Q31s', 'Q32c', 'Q32s', 'Q33c', 'Q33s',
                           'Q40', 'Q41c', 'Q41s', 'Q42c', 'Q42s', 'Q43c', 'Q43s', 'Q44c', 'Q44s'];
    
    for (const name of componentNames) {
      try {
        components[name] = mult.getComponent(name);
      } catch (e) {
        // Component not available for this multipole rank
        break;
      }
    }
    
    return components;
  }
}


// Export the main API
export default {
  DMAConfig,
  DMAResult,
  calculateDMA,
  generatePunchFile
};