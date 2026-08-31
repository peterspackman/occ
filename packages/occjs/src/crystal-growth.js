/**
 * Crystal growth (occ cg) and COSMO-RS solvation.
 *
 * The native config objects are embind handles that must be released, so
 * these wrappers take a plain options object, own the handle's lifetime, and
 * hand back ordinary JS values.
 */

const CG_KEYS = new Set([
  'crystalFilename', 'modelName', 'maxRadius', 'solvent', 'solvationModel',
  'temperature', 'chargeString', 'cgRadius', 'computeMorphology',
  'numSurfaceEnergies'
]);

const COSMORS_KEYS = new Set([
  'method', 'basis', 'pureSpherical', 'probeRadius', 'angularPoints',
  'constrainCharge', 'temperature', 'liquidVolume', 'numRings'
]);

function applyOptions(target, options, allowed, what) {
  for (const [key, value] of Object.entries(options)) {
    if (!allowed.has(key)) {
      throw new Error(
        `${what}: unknown option '${key}'. Valid options: ` +
        `${[...allowed].sort().join(', ')}`);
    }
    target[key] = value;
  }
}

/**
 * Run a crystal-growth calculation.
 *
 * The native driver reads its structure from a path, so pass either `cif`
 * (the text of a CIF, staged into the module filesystem here and removed
 * afterwards) or `crystalFilename` for a file already there.
 *
 * @param {Object} options - `cif`, plus any CrystalGrowthConfig property.
 * @returns {Promise<Object>} {moleculeResults, morphology?}
 */
export async function calculateCrystalGrowth(options = {}) {
  const { loadOCC } = await import('./module-loader.js');
  const Module = await loadOCC();

  const { cif, ...rest } = options;
  if (cif && rest.crystalFilename) {
    throw new Error(
      'calculateCrystalGrowth: pass either cif or crystalFilename, not both');
  }
  if (!cif && !rest.crystalFilename) {
    throw new Error(
      'calculateCrystalGrowth: a structure is required; pass cif with the ' +
      'text of a CIF, or crystalFilename for one already in the module ' +
      'filesystem');
  }

  let staged = null;
  if (cif) {
    if (!Module.FS) {
      throw new Error(
        'calculateCrystalGrowth: this build does not expose FS, so cif text ' +
        'cannot be staged; write the file yourself and pass crystalFilename');
    }
    staged = `/occ_cg_${Date.now()}_${Math.random().toString(36).slice(2)}.cif`;
    Module.FS.writeFile(staged, cif);
    rest.crystalFilename = staged;
  }

  const config = new Module.CrystalGrowthConfig();
  try {
    applyOptions(config, rest, CG_KEYS, 'calculateCrystalGrowth');
    return Module.calculateCrystalGrowthEnergies(config);
  } finally {
    config.delete();
    if (staged) {
      try { Module.FS.unlink(staged); } catch { /* already gone */ }
    }
  }
}

/**
 * openCOSMO-RS solvation free energy of a molecule.
 *
 * @param {Object} solute - Molecule object.
 * @param {string|Object} solvent - A solvent name with a cached segment
 *   ensemble (see availableCosmoRsSolvents), or a Molecule whose conductor
 *   cavity is computed instead.
 * @param {Object} options - Any of the CosmoRSSettings properties.
 * @returns {Promise<Object>} {energy: {...}, cavityArea, cavityVolume,
 *   numRings, total}, energies in kJ/mol.
 */
export async function cosmoRsSolvation(solute, solvent, options = {}) {
  const { loadOCC } = await import('./module-loader.js');
  const Module = await loadOCC();

  const settings = new Module.CosmoRSSettings();
  try {
    applyOptions(settings, options, COSMORS_KEYS, 'cosmoRsSolvation');
    if (typeof solvent === 'string') {
      return Module.cosmoRsSolvationFreeEnergy(solute, solvent, settings);
    }
    return Module.cosmoRsSolvationFreeEnergyInSolventGeometry(
      solute, solvent, settings);
  } finally {
    settings.delete();
  }
}

/**
 * Solvent names with a cached segment ensemble, sorted.
 * @returns {Promise<string[]>}
 */
export async function availableCosmoRsSolvents() {
  const { loadOCC } = await import('./module-loader.js');
  const Module = await loadOCC();

  const names = Module.availableCosmoRsSolvents();
  try {
    const out = [];
    for (let i = 0; i < names.size(); i++) out.push(names.get(i));
    return out;
  } finally {
    names.delete();
  }
}
