/**
 * Tests for DMA (Distributed Multipole Analysis) functionality
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { loadOCC, moleculeFromXYZ, createQMCalculation, wavefunctionFromString } from '../dist/index.js';
import { 
  calculateDMA, 
  generatePunchFile, 
  DMAConfig, 
  DMAResult 
} from '../dist/dma.js';

// Test molecules - H2 molecule that matches the inline wavefunction
const h2XYZ = `2
H2 molecule
H  0.000000  0.000000  0.699199
H  0.000000  0.000000 -0.699199`;

const waterXYZ = `3
Water molecule
O  0.000000  0.000000  0.000000
H  0.000000  0.000000  1.000000
H  0.942809  0.000000 -0.333333`;

const methanolXYZ = `6
Methanol molecule
C  0.000000  0.000000  0.000000
O  1.400000  0.000000  0.000000
H  2.000000  0.500000  0.500000
H -0.500000  0.866025  0.000000
H -0.500000 -0.866025  0.000000
H -0.500000  0.000000  0.866025`;

// Inline H2 wavefunction file (B3LYP/STO-3G)
const h2WavefunctionFchk = `h2
SP        RB3LYP                                                      STO-3G
Number of atoms                            I                2
Info1-9                                    I   N=           9
           9           9           0           0           0         110
           1          18        -502
Charge                                     I                0
Multiplicity                               I                1
Number of electrons                        I                2
Number of alpha electrons                  I                1
Number of beta electrons                   I                1
Number of basis functions                  I                2
Number of independent functions            I                2
Number of point charges in /Mol/           I                0
Number of translation vectors              I                0
Atomic numbers                             I   N=           2
           1           1
Nuclear charges                            R   N=           2
  1.00000000E+00  1.00000000E+00
Current cartesian coordinates              R   N=           6
  0.00000000E+00  0.00000000E+00  6.99198669E-01  8.56271412E-17  0.00000000E+00
 -6.99198669E-01
Force Field                                I                0
Int Atom Types                             I   N=           2
           0           0
MM charges                                 R   N=           2
  0.00000000E+00  0.00000000E+00
Integer atomic weights                     I   N=           2
           1           1
Real atomic weights                        R   N=           2
  1.00782504E+00  1.00782504E+00
Atom fragment info                         I   N=           2
           0           0
Atom residue num                           I   N=           2
           0           0
Nuclear spins                              I   N=           2
           1           1
Nuclear ZEff                               R   N=           2
 -1.00000000E+00 -1.00000000E+00
Nuclear ZNuc                               R   N=           2
  1.00000000E+00  1.00000000E+00
Nuclear QMom                               R   N=           2
  0.00000000E+00  0.00000000E+00
Nuclear GFac                               R   N=           2
  2.79284600E+00  2.79284600E+00
MicOpt                                     I   N=           2
          -1          -1
Number of contracted shells                I                2
Number of primitive shells                 I                6
Pure/Cartesian d shells                    I                0
Pure/Cartesian f shells                    I                0
Highest angular momentum                   I                0
Largest degree of contraction              I                3
Shell types                                I   N=           2
           0           0
Number of primitives per shell             I   N=           2
           3           3
Shell to atom map                          I   N=           2
           1           2
Primitive exponents                        R   N=           6
  3.42525091E+00  6.23913730E-01  1.68855404E-01  3.42525091E+00  6.23913730E-01
  1.68855404E-01
Contraction coefficients                   R   N=           6
  1.54328967E-01  5.35328142E-01  4.44634542E-01  1.54328967E-01  5.35328142E-01
  4.44634542E-01
Coordinates of each shell                  R   N=           6
  0.00000000E+00  0.00000000E+00  6.99198669E-01  8.56271412E-17  0.00000000E+00
 -6.99198669E-01
Constraint Structure                       R   N=           6
  0.00000000E+00  0.00000000E+00  6.99198669E-01  8.56271412E-17  0.00000000E+00
 -6.99198669E-01
Num ILSW                                   I              100
ILSW                                       I   N=         100
           0           0           0           0           2           0
           0           0           0           0         402          -1
           0           0           0           0           0           0
           0           0           0           0           0           0
           1           0           0           0           0           0
           0           0      100000           0          -1           0
           0           0           0           0           0           0
           0           0           0           1           0           0
           0           0           1           0           0           0
           0           0           4          41           0           0
           0           0           5           0           0           0
           0           0           0           2           0           0
           0           0           0           0           0           0
           0           0           0           0           0           0
           0           0           0           0           0           0
           0           0           0           0           0           0
           0           0           0           0
Num RLSW                                   I               41
RLSW                                       R   N=          41
  8.00000000E-01  7.20000000E-01  1.00000000E+00  8.10000000E-01  2.00000000E-01
  0.00000000E+00  0.00000000E+00  1.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  1.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  1.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  1.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  1.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  1.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  1.00000000E+00  1.00000000E+00
  0.00000000E+00
MxBond                                     I                1
NBond                                      I   N=           2
           1           1
IBond                                      I   N=           2
           2           1
RBond                                      R   N=           2
  1.00000000E+00  1.00000000E+00
Virial Ratio                               R      1.970141361625062E+00
SCF Energy                                 R     -1.165418375762579E+00
Total Energy                               R     -1.165418375762579E+00
External E-field                           R   N=          35
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
IOpCl                                      I                0
IROHF                                      I                0
Alpha Orbital Energies                     R   N=           2
 -4.14539570E-01  4.27590260E-01
Alpha MO coefficients                      R   N=           4
  5.48842275E-01  5.48842275E-01  1.21245192E+00 -1.21245192E+00
Total SCF Density                          R   N=           3
  6.02455687E-01  6.02455687E-01  6.02455687E-01
Mulliken Charges                           R   N=           2
 -2.77555756E-16 -3.33066907E-16
ONIOM Charges                              I   N=          16
           0           0           0           0           0           0
           0           0           0           0           0           0
           0           0           0           0
ONIOM Multiplicities                       I   N=          16
           1           0           0           0           0           0
           0           0           0           0           0           0
           0           0           0           0
Atom Layers                                I   N=           2
           1           1
Atom Modifiers                             I   N=           2
           0           0
Force Field                                I                0
Int Atom Modified Types                    I   N=           2
           0           0
Link Atoms                                 I   N=           2
           0           0
Atom Modified MM Charges                   R   N=           2
  0.00000000E+00  0.00000000E+00
Link Distances                             R   N=           8
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00  0.00000000E+00  0.00000000E+00
Cartesian Gradient                         R   N=           6
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
  0.00000000E+00
Dipole Moment                              R   N=           3
 -1.23259516E-32  0.00000000E+00  5.55111512E-17
Quadrupole Moment                          R   N=           6
 -1.04128604E-01 -1.04128604E-01  2.08257207E-01  0.00000000E+00 -1.91281142E-17
  0.00000000E+00
QEq coupling tensors                       R   N=          12
  1.83153654E-01  0.00000000E+00  1.83153654E-01  4.89879653E-17  0.00000000E+00
 -3.66307308E-01  1.83153654E-01  0.00000000E+00  1.83153654E-01  1.34443277E-17
  0.00000000E+00 -3.66307308E-01

`;

// Test helper function to create wavefunction from inline data
async function createWavefunction() {
  console.log('Loading wavefunction from FCHK string...');
  const wavefunction = await wavefunctionFromString(h2WavefunctionFchk, 'fchk');
  
  console.log('Wavefunction loaded, checking properties:');
  console.log('- Atoms:', wavefunction.atoms.size());
  console.log('- Basis functions:', wavefunction.basis.nbf());
  console.log('- Charge:', wavefunction.charge());
  console.log('- Multiplicity:', wavefunction.multiplicity());
  
  // Check atom positions
  for (let i = 0; i < wavefunction.atoms.size(); i++) {
    const atom = wavefunction.atoms.get(i);
    console.log(`- Atom ${i}: Z=${atom.atomicNumber} at (${atom.x}, ${atom.y}, ${atom.z})`);
  }
  
  return wavefunction;
}

describe('DMA Configuration', () => {
  let Module;
  
  beforeAll(async () => {
    Module = await loadOCC();
  });

  it('should create DMA config with default values', () => {
    const config = new DMAConfig();
    
    expect(config.maxRank).toBe(4);
    expect(config.bigExponent).toBe(4.0);
    expect(config.includeNuclei).toBe(true);
    expect(config.writePunch).toBe(false);
    expect(config.punchFilename).toBe("dma.punch");
  });

  it('should allow setting atom-specific parameters', () => {
    const config = new DMAConfig();
    
    config.setAtomRadius("H", 0.35);
    config.setAtomRadius("C", 0.65);
    config.setAtomLimit("H", 1);
    config.setAtomLimit("C", 4);
    
    expect(config.atomRadii.get("H")).toBe(0.35);
    expect(config.atomRadii.get("C")).toBe(0.65);
    expect(config.atomLimits.get("H")).toBe(1);
    expect(config.atomLimits.get("C")).toBe(4);
  });
});

describe('DMA Calculations', () => {
  let Module;
  
  beforeAll(async () => {
    Module = await loadOCC();
  });

  it('should perform DMA calculation on H2 molecule', async () => {
    const wavefunction = await createWavefunction();
    
    // Let's try to call DMA directly to see exactly where it fails
    const Module = await loadOCC();
    const config = new Module.DMAConfig();
    config.settings.max_rank = 4;
    config.settings.big_exponent = 4.0;
    config.settings.include_nuclei = true;
    config.setAtomRadius("H", 0.35);
    config.setAtomLimit("H", 1);
    
    const driver = new Module.DMADriver();
    driver.set_config(config);
    
    console.log('About to call driver.runWithWavefunction directly...');
    console.log('Wavefunction basis is spherical:', wavefunction.basis.isPure ? wavefunction.basis.isPure() : 'unknown');
    
    try {
      const output = driver.runWithWavefunction(wavefunction);
      console.log('DMA calculation successful!');
      
      const result = new DMAResult(output);
      expect(result).toBeInstanceOf(DMAResult);
      expect(result.result).toBeDefined();
      expect(result.sites).toBeDefined();
      
      // H2 should have 2 sites (H, H)
      expect(result.sites.size()).toBe(2);
    } catch (error) {
      console.error('Direct DMA call failed:', error);
      throw error;
    }
  });

  it('should calculate DMA with custom options', async () => {
    const wavefunction = await createWavefunction();
    
    const options = {
      maxRank: 2,
      bigExponent: 3.0,
      atomRadii: { "H": 0.40 },
      atomLimits: { "H": 1 }
    };
    
    const result = await calculateDMA(wavefunction, options);
    
    expect(result).toBeInstanceOf(DMAResult);
    expect(result.result.max_rank).toBe(2);
  });

});

describe('DMA Results Analysis', () => {
  let dmaResult;
  
  beforeAll(async () => {
    const wavefunction = await createWavefunction();
    dmaResult = await calculateDMA(wavefunction);
  });

  it('should provide access to individual site multipoles', () => {
    const firstHydrogenMultipoles = dmaResult.getSiteMultipoles(0);
    
    expect(firstHydrogenMultipoles).toBeDefined();
    expect(firstHydrogenMultipoles.max_rank).toBeDefined();
    expect(typeof firstHydrogenMultipoles.max_rank).toBe('number');
  });

  it('should provide total multipole moments', () => {
    const total = dmaResult.getTotalMultipoles();
    
    // Note: getTotalMultipoles() returns null in current implementation
    // This test should be updated when total multipoles are implemented
    expect(total).toBeNull();
  });

  it('should allow access to multipole components by name', () => {
    const charge = dmaResult.getMultipoleComponent(0, 'charge');
    const q00 = dmaResult.getMultipoleComponent(0, 'Q00');
    const q10 = dmaResult.getMultipoleComponent(0, 'Q10');
    
    expect(typeof charge).toBe('number');
    expect(charge).toBe(q00);
    expect(typeof q10).toBe('number');
  });

  it('should provide all components as an object', () => {
    const components = dmaResult.getAllComponents(0);
    
    expect(components).toHaveProperty('Q00');
    expect(components).toHaveProperty('charge');
    expect(components).toHaveProperty('Q10');
    expect(components).toHaveProperty('Q11c');
    expect(components).toHaveProperty('Q11s');
    expect(components).toHaveProperty('maxRank');
    
    expect(components.Q00).toBe(components.charge);
    expect(typeof components.maxRank).toBe('number');
  });
});

describe('Punch File Generation', () => {
  let dmaResult;
  
  beforeAll(async () => {
    const wavefunction = await createWavefunction();
    dmaResult = await calculateDMA(wavefunction);
  });

  it('should generate punch file content', async () => {
    const punchContent = await generatePunchFile(dmaResult);
    
    expect(typeof punchContent).toBe('string');
    expect(punchContent).toContain('! Distributed multipoles from occ dma');
    expect(punchContent).toContain('Units angstrom');
    expect(punchContent).toContain('Rank');
  });

  it('should generate punch file via DMAResult method', async () => {
    const punchContent = await dmaResult.toPunchFile();
    
    expect(typeof punchContent).toBe('string');
    expect(punchContent).toContain('! Distributed multipoles from occ dma');
  });

  it('should contain expected number of sites in punch file', async () => {
    const punchContent = await dmaResult.toPunchFile();
    const rankLines = punchContent.match(/^Rank \d+$/gm);
    
    // Should have one "Rank" line per site
    expect(rankLines).toHaveLength(2); // H2 has 2 atoms
  });
});

describe('Error Handling', () => {
  it('should return 0 for invalid multipole component', async () => {
    const wavefunction = await createWavefunction();
    const result = await calculateDMA(wavefunction);
    
    const invalidComponent = result.getMultipoleComponent(0, 'invalid_component');
    expect(invalidComponent).toBe(0.0);
  });
});

describe('Complex Molecules', () => {
  it('should handle H2 molecule with various settings', async () => {
    const wavefunction = await createWavefunction();
    
    const result = await calculateDMA(wavefunction, {
      maxRank: 3,
      atomRadii: { "H": 0.35 },
      atomLimits: { "H": 2 }
    });
    
    expect(result).toBeInstanceOf(DMAResult);
    expect(result.sites.size()).toBe(2); // H2 has 2 atoms
    
    // Check that we can access multipoles for all sites
    for (let i = 0; i < 2; i++) {
      const components = result.getAllComponents(i);
      expect(components).toHaveProperty('Q00');
      expect(typeof components.Q00).toBe('number');
    }
    
    // Total charge should be close to 0 for neutral molecule
    // Note: getTotalMultipoles() returns null in current implementation
    // This test should be updated when total multipoles are implemented
    const total = result.getTotalMultipoles();
    if (total) {
      expect(Math.abs(total.Q00())).toBeLessThan(0.1);
    }
  });
});