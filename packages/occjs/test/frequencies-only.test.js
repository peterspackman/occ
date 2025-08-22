import { describe, it, expect, beforeAll } from 'vitest';
import { loadOCC, moleculeFromXYZ } from '../dist/index.js';

describe('Frequency Calculation Test', () => {
  let Module;
  let waterMolecule;

  beforeAll(async () => {
    Module = await loadOCC();

    // Reduce logging verbosity for tests
    if (Module.setLogLevel) {
      Module.setLogLevel('WARN');
    }

    // Use the optimized water geometry from HF/3-21G optimization
    waterMolecule = await moleculeFromXYZ(`3
Optimized geometry - Final energy: -75.585959756562 Ha
O    -0.019728    -0.027022     0.000000
H     0.946851    -0.012467     0.000000
H    -0.327123     0.889489     0.000000`);
  });

  it('should compute frequencies at equilibrium geometry', async () => {
    console.log('Setting up calculation at equilibrium geometry...');

    // Set up calculation with 3-21G basis (same as optimization)
    const basis = Module.AOBasis.load(waterMolecule.atoms(), "3-21G");
    const hf = new Module.HartreeFock(basis);

    // Run SCF
    const scf = new Module.HartreeFockSCF(hf);
    scf.setChargeMultiplicity(0, 1);
    const energy = await scf.run();
    const wfn = scf.wavefunction();

    console.log(`SCF energy: ${energy.toFixed(8)} Ha`);

    // Compute Hessian using convenience method
    console.log('Creating Hessian evaluator...');
    const hessEvaluator = hf.hessianEvaluator();
    hessEvaluator.setStepSize(0.01); // 0.01 Bohr step size
    hessEvaluator.setUseAcousticSumRule(true);

    console.log(`Hessian settings: step_size=${hessEvaluator.stepSize().toFixed(3)} Bohr, acoustic_sum_rule=${hessEvaluator.useAcousticSumRule()}`);

    // Compute Hessian
    console.log('Computing Hessian matrix...');
    const hessian = hessEvaluator.compute(wfn);
    expect(hessian).toBeDefined();
    console.log(`Hessian computed: ${hessian.rows()}x${hessian.cols()} matrix`);

    // Compute vibrational modes
    console.log('Computing vibrational modes...');
    const vibModes = Module.computeVibrationalModesFromMolecule(hessian, waterMolecule, true);
    expect(vibModes).toBeDefined();
    expect(vibModes.nModes()).toBeGreaterThan(0);
    expect(vibModes.nAtoms()).toBe(3);

    console.log(`Found ${vibModes.nModes()} modes for ${vibModes.nAtoms()} atoms`);

    // Get and display frequencies
    const frequencies = vibModes.getAllFrequencies();
    console.log('\n=== VIBRATIONAL FREQUENCIES (cm⁻¹) ===');

    const freqArray = [];
    for (let i = 0; i < frequencies.size(); i++) {
      const freq = frequencies.get(i);
      freqArray.push(freq);
      if (freq < 0) {
        console.log(`  Mode ${i + 1}: ${Math.abs(freq).toFixed(2)}i cm⁻¹ (imaginary)`);
      } else {
        console.log(`  Mode ${i + 1}: ${freq.toFixed(2)} cm⁻¹`);
      }
    }

    // Check that we have realistic frequencies for water
    const realFreqs = freqArray.filter(f => f > 100); // Filter out rotational/translational modes
    expect(realFreqs.length).toBeGreaterThan(0); // Should have vibrational modes

    console.log('\n=== VIBRATIONAL ANALYSIS SUMMARY ===');
    const summary = vibModes.summaryString();
    console.log(summary);

    // Test some expected properties
    expect(vibModes.nModes()).toBe(9); // 3N modes for 3 atoms
    expect(frequencies.size()).toBe(9);

    // For water, we expect 3 real vibrational modes (after translational/rotational projection)
    const vibFreqs = freqArray.filter(f => f > 500); // Real vibrational modes
    console.log(`\nFound ${vibFreqs.length} high-frequency vibrational modes`);
    expect(vibFreqs.length).toBe(3); // Water should have exactly 3 vibrational modes

    // Display the frequency analysis
    console.log('\n=== FREQUENCY ANALYSIS ===');
    console.log(`Total modes: ${freqArray.length}`);
    console.log(`Imaginary frequencies: ${freqArray.filter(f => f < 0).length}`);
    console.log(`Near-zero frequencies: ${freqArray.filter(f => Math.abs(f) < 100).length}`);
    console.log(`Vibrational frequencies: ${freqArray.filter(f => f > 100).length}`);

    if (vibFreqs.length > 0) {
      console.log('\nVibrational modes:');
      const sortedFreqs = vibFreqs.sort((a, b) => a - b);
      sortedFreqs.forEach((freq, i) => {
        const mode = i === 0 ? 'bend' : i === 1 ? 'sym stretch' : 'asym stretch';
        console.log(`  ${freq.toFixed(2)} cm⁻¹ (${mode})`);
      });

      // Check that frequencies match expected values for HF/3-21G water at this geometry:
      // Expected exact values:
      // Bend: 1799.23 cm⁻¹
      // Symmetric stretch: 3812.17 cm⁻¹  
      // Asymmetric stretch: 3945.58 cm⁻¹
      const [bend, symStretch, asymStretch] = sortedFreqs;

      console.log('\n=== VALIDATION ===');
      console.log(`Bend frequency: ${bend.toFixed(2)} cm⁻¹ (expected 1799.23)`);
      console.log(`Sym stretch: ${symStretch.toFixed(2)} cm⁻¹ (expected 3812.17)`);
      console.log(`Asym stretch: ${asymStretch.toFixed(2)} cm⁻¹ (expected 3945.58)`);

      // Check how close we are to expected values (within ~1% tolerance)
      const bendError = Math.abs(bend - 1799.23);
      const symError = Math.abs(symStretch - 3812.17);
      const asymError = Math.abs(asymStretch - 3945.58);

      console.log('\n=== ACCURACY CHECK ===');
      console.log(`Bend error: ${bendError.toFixed(2)} cm⁻¹ (${(100 * bendError / 1799.23).toFixed(2)}%)`);
      console.log(`Sym stretch error: ${symError.toFixed(2)} cm⁻¹ (${(100 * symError / 3812.17).toFixed(2)}%)`);
      console.log(`Asym stretch error: ${asymError.toFixed(2)} cm⁻¹ (${(100 * asymError / 3945.58).toFixed(2)}%)`);

      // Validate frequencies are close to expected (within ~30 cm⁻¹ tolerance)
      expect(bendError).toBeLessThan(30); // Allow for numerical differences
      expect(symError).toBeLessThan(30); // Allow for numerical differences  
      expect(asymError).toBeLessThan(30); // Allow for numerical differences

      console.log('✅ All frequencies match expected values within tolerance!');
    }

  }, 120000); // 2-minute timeout for Hessian calculation
});
