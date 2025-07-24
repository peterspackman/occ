import { describe, it, expect, beforeAll } from 'vitest';
import { loadOCC, moleculeFromXYZ, BasisSets } from '../dist/index.js';

describe('Basis Set Availability Check', () => {
  let Module;
  let h2Molecule;

  beforeAll(async () => {
    Module = await loadOCC();
    h2Molecule = await moleculeFromXYZ(`2
H2 molecule
H 0.0 0.0 0.0
H 0.0 0.0 0.74`);
  });

  it('should list available basis sets', () => {
    console.log('Available basis sets in BasisSets:', Object.keys(BasisSets));
    expect(Object.keys(BasisSets).length).toBeGreaterThan(0);
  });

  it('should test specific basis sets', async () => {
    const testBases = ['sto-3g', 'STO-3G', '3-21g', '6-31g'];
    
    for (const basisName of testBases) {
      try {
        console.log(`Testing basis: ${basisName}`);
        const basis = Module.AOBasis.load(h2Molecule.atoms(), basisName);
        console.log(`✓ ${basisName}: nbf=${basis.nbf()}, nsh=${basis.nsh()}`);
      } catch (e) {
        console.log(`✗ ${basisName}: ${e.message}`);
      }
    }
  });

  it('should test JSON basis loading and find correct method names', async () => {
    const basisJson = JSON.stringify({
      "elements": {
        "H": {
          "electron_shells": [
            {
              "function_type": "gto",
              "region": "",
              "angular_momentum": [0],
              "exponents": ["3.42525091", "0.62391373", "0.16885540"],
              "coefficients": [["0.15432897", "0.53532814", "0.44463454"]]
            }
          ]
        }
      }
    });
    
    try {
      const basis = Module.AOBasis.fromJson(h2Molecule.atoms(), basisJson);
      console.log(`JSON basis loaded successfully!`);
      console.log(`nbf: ${basis.nbf()}`);
      
      // Try different method names to find the correct one for number of shells
      const methodsToTry = ['nsh', 'nShell', 'nShells', 'numShells', 'size', 'shellCount'];
      for (const method of methodsToTry) {
        try {
          if (typeof basis[method] === 'function') {
            console.log(`${method}: ${basis[method]()}`);
          }
        } catch (e) {
          console.log(`${method}: not available`);
        }
      }
      
      // List all available methods
      console.log('Available methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(basis)).filter(name => typeof basis[name] === 'function'));
      
      expect(basis.nbf()).toBe(2);
    } catch (e) {
      console.log(`JSON basis failed: ${e.message}`);
      throw e;
    }
  });
});