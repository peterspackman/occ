/**
 * SimpleBasisLoader - Minimal utility for loading custom basis sets
 * Now that we have preloaded data, this just provides a simple interface
 * for adding custom JSON basis sets to the virtual filesystem if needed
 */

/**
 * Load a basis set for a molecule using the built-in preloaded data
 * @param {Object} module - OCC module
 * @param {Object} molecule - Molecule object  
 * @param {string} basisName - Name of basis set
 * @returns {Object} AOBasis object
 */
export function loadBasisSet(module, molecule, basisName) {
  return module.AOBasis.load(molecule.atoms(), basisName);
}

/**
 * Create AOBasis from custom JSON data
 * @param {Object} module - OCC module
 * @param {Object} molecule - Molecule object
 * @param {Object|string} basisData - JSON basis data
 * @returns {Object} AOBasis object
 */
export function loadBasisFromJSON(module, molecule, basisData) {
  const jsonString = typeof basisData === 'string' ? basisData : JSON.stringify(basisData);
  return module.AOBasis.fromJson(molecule.atoms(), jsonString);
}

/**
 * Add a custom basis set to the virtual filesystem
 * This allows the built-in AOBasis.load() to find it later
 * @param {Object} module - OCC module
 * @param {string} basisName - Name for the basis set
 * @param {Object} basisData - JSON basis data
 */
export function addCustomBasisSet(module, basisName, basisData) {
  if (!module.FS) {
    console.warn('Virtual filesystem not available - cannot add custom basis set');
    return false;
  }
  
  try {
    // Ensure basis directory exists
    try {
      module.FS.stat('/basis');
    } catch (e) {
      module.FS.mkdir('/basis');
    }
    
    // Write the basis data as a JSON file
    const jsonString = typeof basisData === 'string' ? basisData : JSON.stringify(basisData, null, 2);
    const filename = `/basis/${basisName.toLowerCase()}.json`;
    
    module.FS.writeFile(filename, jsonString);
    console.log(`Added custom basis set: ${basisName} -> ${filename}`);
    return true;
    
  } catch (error) {
    console.error(`Failed to add custom basis set ${basisName}:`, error.message);
    return false;
  }
}

/**
 * List available basis sets from the virtual filesystem
 * @param {Object} module - OCC module
 * @returns {string[]} Array of basis set names
 */
export function listAvailableBasisSets(module) {
  if (!module.FS) {
    console.warn('Virtual filesystem not available');
    return [];
  }
  
  try {
    const files = module.FS.readdir('/basis');
    return files
      .filter(name => name.endsWith('.json'))
      .map(name => name.replace('.json', ''))
      .sort();
  } catch (error) {
    console.warn('Failed to list basis sets:', error.message);
    return [];
  }
}

/**
 * Check if a basis set is available
 * @param {Object} module - OCC module
 * @param {string} basisName - Name of basis set
 * @returns {boolean} True if available
 */
export function hasBasisSet(module, basisName) {
  if (!module.FS) {
    return false;
  }
  
  try {
    const filename = `/basis/${basisName.toLowerCase()}.json`;
    module.FS.stat(filename);
    return true;
  } catch (error) {
    return false;
  }
}