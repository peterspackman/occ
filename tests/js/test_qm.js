/**
 * JavaScript tests for OCC QM module
 * 
 * Tests quantum mechanical functionality including basis sets, SCF, and wavefunctions
 */

const test = require('./test_framework.js');

async function runQMTests(Module) {
    const suite = test.createSuite('QM Module Tests');

    // Set data directory for basis sets - this is required for WASM
    // In a browser/WASM environment, you'll need to set this to the path
    // where your basis set files are located (typically in a share/basis directory)
    Module.setDataDirectory('./share');

    // Create a simple H2 molecule for testing
    function createH2Molecule() {
        const h2xyz = "2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 1.4\n"
        const h2 = Module.Molecule.fromXyzString(h2xyz);
        h2.setName("H2");
        return h2;
    }

    // Helper function to create STO-3G basis for H, C, N, O
    function createSTO3GBasis(atoms) {
        const sto3gJson = JSON.stringify({
            "molssi_bse_schema": {
                "schema_type": "complete",
                "schema_version": "0.1"
            },
            "elements": {
                "1": {
                    "electron_shells": [
                        {
                            "function_type": "gto",
                            "angular_momentum": [0],
                            "exponents": ["3.42525091", "0.62391373", "0.16885540"],
                            "coefficients": [
                                ["0.15432897", "0.53532814", "0.44463454"]
                            ]
                        }
                    ]
                },
                "6": {
                    "electron_shells": [
                        {
                            "function_type": "gto",
                            "angular_momentum": [0],
                            "exponents": ["71.6168370", "13.0450960", "3.5305122"],
                            "coefficients": [
                                ["0.15432897", "0.53532814", "0.44463454"]
                            ]
                        },
                        {
                            "function_type": "gto",
                            "angular_momentum": [0],
                            "exponents": ["2.9412494", "0.6834831", "0.2222899"],
                            "coefficients": [
                                ["-0.09996723", "0.39951283", "0.70011547"]
                            ]
                        },
                        {
                            "function_type": "gto",
                            "angular_momentum": [1],
                            "exponents": ["2.9412494", "0.6834831", "0.2222899"],
                            "coefficients": [
                                ["0.15591627", "0.60768372", "0.39195739"]
                            ]
                        }
                    ]
                },
                "7": {
                    "electron_shells": [
                        {
                            "function_type": "gto",
                            "angular_momentum": [0],
                            "exponents": ["99.1061690", "18.0523120", "4.8856602"],
                            "coefficients": [
                                ["0.15432897", "0.53532814", "0.44463454"]
                            ]
                        },
                        {
                            "function_type": "gto",
                            "angular_momentum": [0],
                            "exponents": ["3.7804559", "0.8784966", "0.2857144"],
                            "coefficients": [
                                ["-0.09996723", "0.39951283", "0.70011547"]
                            ]
                        },
                        {
                            "function_type": "gto",
                            "angular_momentum": [1],
                            "exponents": ["3.7804559", "0.8784966", "0.2857144"],
                            "coefficients": [
                                ["0.15591627", "0.60768372", "0.39195739"]
                            ]
                        }
                    ]
                },
                "8": {
                    "electron_shells": [
                        {
                            "function_type": "gto",
                            "angular_momentum": [0],
                            "exponents": ["130.7093200", "23.8088610", "6.4436083"],
                            "coefficients": [
                                ["0.15432897", "0.53532814", "0.44463454"]
                            ]
                        },
                        {
                            "function_type": "gto",
                            "angular_momentum": [0],
                            "exponents": ["5.0331513", "1.1695961", "0.3803890"],
                            "coefficients": [
                                ["-0.09996723", "0.39951283", "0.70011547"]
                            ]
                        },
                        {
                            "function_type": "gto",
                            "angular_momentum": [1],
                            "exponents": ["5.0331513", "1.1695961", "0.3803890"],
                            "coefficients": [
                                ["0.15591627", "0.60768372", "0.39195739"]
                            ]
                        }
                    ]
                }
            }
        });

        return Module.AOBasis.fromJson(atoms, sto3gJson);
    }

    // Basis set tests using JSON
    suite.test('AOBasis loading and properties from JSON', () => {
        const h2 = createH2Molecule();
        const atoms = h2.atoms();
        const basis = createSTO3GBasis(atoms);

        test.assertEqual(basis.size(), 2, `${basis} Number of shells for H2/STO-3G`);
        test.assertEqual(basis.nbf(), 2, 'Number of basis functions for H2/STO-3G');
        test.assertTrue(basis.lMax() >= 0, 'Maximum angular momentum >= 0');

        //const shells = basis.shells();
        //test.assertEqual(shells.size(), 2, 'Shells vector size');

        // Test first shell properties
        //const shell0 = shells.get(0);
        //test.assertTrue(shell0.numPrimitives() > 0, 'Shell has primitives');
        //test.assertTrue(shell0.numContractions() > 0, 'Shell has contractions');
        //test.assertTrue(shell0.norm() > 0, 'Shell norm > 0');
    });

    // MolecularOrbitals tests
    suite.test('MolecularOrbitals creation and properties', () => {
        const mo = new Module.MolecularOrbitals();

        // Set basic properties
        mo.kind = Module.SpinorbitalKind.Restricted;
        mo.numAlpha = 1;
        mo.numBeta = 1;
        mo.numAo = 2;

        test.assertEqual(mo.kind, Module.SpinorbitalKind.Restricted, 'MO kind');
        test.assertEqual(mo.numAlpha, 1, 'Number of alpha electrons');
        test.assertEqual(mo.numBeta, 1, 'Number of beta electrons');
        test.assertEqual(mo.numAo, 2, 'Number of AOs');
    });

    // HartreeFock basic tests
    suite.test('HartreeFock object creation and basic integrals', () => {
        const h2 = createH2Molecule();
        const atoms = h2.atoms();
        test.assertTrue(atoms.size() == 2, `Length of H2 atoms == 2 (${atoms.size()})`);
        const basis = createSTO3GBasis(atoms);
        const hf = new Module.HartreeFock(basis);

        // Test nuclear repulsion
        const nuclearRepulsion = hf.nuclearRepulsion();
        test.assertTrue(nuclearRepulsion > 0, 'Nuclear repulsion > 0');
        // H2 bond length is 1.4 Angstrom = 2.646 Bohr, so nuclear repulsion = 1*1/2.646 = 0.3779
        test.assertAlmostEqual(nuclearRepulsion, 1.0 / 2.6458, 1e-4, 'Nuclear repulsion for H2');

        // Test one-electron integrals
        const overlap = hf.overlapMatrix();
        const kinetic = hf.kineticMatrix();
        const nuclear = hf.nuclearAttractionMatrix();

        test.assertEqual(overlap.rows(), 2, 'Overlap matrix rows');
        test.assertEqual(overlap.cols(), 2, 'Overlap matrix cols');
        test.assertEqual(kinetic.rows(), 2, 'Kinetic matrix rows');
        test.assertEqual(kinetic.cols(), 2, 'Kinetic matrix cols');
        test.assertEqual(nuclear.rows(), 2, 'Nuclear matrix rows');
        test.assertEqual(nuclear.cols(), 2, 'Nuclear matrix cols');

        // Overlap matrix should be positive definite on diagonal
        test.assertTrue(overlap.get(0, 0) > 0, 'Overlap diagonal element positive');
        test.assertTrue(overlap.get(1, 1) > 0, 'Overlap diagonal element positive');

        // Kinetic energy should be positive on diagonal
        test.assertTrue(kinetic.get(0, 0) > 0, 'Kinetic diagonal element positive');
        test.assertTrue(kinetic.get(1, 1) > 0, 'Kinetic diagonal element positive');

        // Nuclear attraction should be negative
        test.assertTrue(nuclear.get(0, 0) < 0, 'Nuclear diagonal element negative');
        test.assertTrue(nuclear.get(1, 1) < 0, 'Nuclear diagonal element negative');
    });

    // SCF convergence settings
    suite.test('SCFConvergenceSettings', () => {
        const settings = new Module.SCFConvergenceSettings();

        // Test default values are reasonable
        test.assertTrue(settings.energyThreshold > 0, 'Energy threshold > 0');
        test.assertTrue(settings.commutatorThreshold > 0, 'Commutator threshold > 0');
        test.assertTrue(settings.incrementalFockThreshold > 0, 'Incremental Fock threshold > 0');

        // Test setting values
        settings.energyThreshold = 1e-8;
        settings.commutatorThreshold = 1e-6;

        test.assertAlmostEqual(settings.energyThreshold, 1e-8, 1e-12, 'Energy threshold set');
        test.assertAlmostEqual(settings.commutatorThreshold, 1e-6, 1e-12, 'Commutator threshold set');

        // Test convergence checking (with dummy values)
        test.assertFalse(settings.energyConverged(1.0), 'Energy not converged with large delta');
        test.assertTrue(settings.energyConverged(1e-10), 'Energy converged with small delta');
    });

    // SCF object construction test
    suite.test('HartreeFockSCF construction and setup', () => {
        const h2 = createH2Molecule();
        const atoms = h2.atoms();
        const basis = createSTO3GBasis(atoms);
        const hf = new Module.HartreeFock(basis);

        // Test SCF object creation
        const scf = new Module.HartreeFockSCF(hf, Module.SpinorbitalKind.Restricted);
        test.assertTrue(scf !== undefined, 'SCF object created successfully');

        // Test convergence settings access
        const convergence = scf.convergenceSettings;
        test.assertTrue(convergence !== undefined, 'Convergence settings accessible');

        // Test setting convergence parameters
        convergence.energyThreshold = 1e-8;
        convergence.commutatorThreshold = 1e-6;
        test.assertAlmostEqual(convergence.energyThreshold, 1e-8, 1e-12, 'Energy threshold set');
        test.assertAlmostEqual(convergence.commutatorThreshold, 1e-6, 1e-12, 'Commutator threshold set');

        // Test charge and multiplicity setting
        scf.setChargeMultiplicity(0, 1);
        test.assertTrue(true, 'Charge and multiplicity set successfully');
    });

    // SCF calculation test
    suite.test('H2 SCF energy calculation', () => {
        const h2 = createH2Molecule();
        const atoms = h2.atoms();
        const basis = createSTO3GBasis(atoms);
        const hf = new Module.HartreeFock(basis);

        // Create and setup SCF object
        const scf = new Module.HartreeFockSCF(hf, Module.SpinorbitalKind.Restricted);
        const convergence = scf.convergenceSettings;
        convergence.energyThreshold = 1e-6;  // Looser convergence for stability
        convergence.commutatorThreshold = 1e-5;
        scf.setChargeMultiplicity(0, 1);

        // Run SCF calculation
        const energy = scf.run();

        // Check that SCF didn't fail
        test.assertTrue(typeof energy === 'number', 'SCF energy is a number');
        test.assertTrue(!isNaN(energy), 'SCF energy is not NaN (no error occurred)');

        // Check energy is reasonable for H2/STO-3G
        test.assertTrue(energy < 0, 'SCF energy is negative');
        test.assertTrue(energy > -2.0, 'SCF energy not too negative');

        // Known approximate value for H2/STO-3G at 1.4 Angstrom
        test.assertAlmostEqual(energy, -0.941480655488, 1e-8, 'H2/STO-3G energy');
    });

    // SCF wavefunction test
    suite.test('H2 SCF wavefunction properties', () => {
        const h2 = createH2Molecule();
        const atoms = h2.atoms();
        const basis = createSTO3GBasis(atoms);
        const hf = new Module.HartreeFock(basis);

        // Create and run SCF
        const scf = new Module.HartreeFockSCF(hf, Module.SpinorbitalKind.Restricted);
        const convergence = scf.convergenceSettings;
        convergence.energyThreshold = 1e-6;
        convergence.commutatorThreshold = 1e-5;
        scf.setChargeMultiplicity(0, 1);
        const energy = scf.run();

        // Skip wavefunction tests if SCF failed
        if (isNaN(energy)) {
            test.assertTrue(false, 'SCF calculation failed, skipping wavefunction tests');
            return;
        }

        // Get wavefunction and test properties
        const wfn = scf.wavefunction();
        test.assertEqual(wfn.charge(), 0, 'Neutral molecule charge');
        test.assertEqual(wfn.multiplicity(), 1, 'Singlet multiplicity');

        const mo = wfn.molecularOrbitals;
        test.assertEqual(mo.kind, Module.SpinorbitalKind.Restricted, 'Restricted calculation');
        test.assertEqual(mo.numAlpha, 1, 'One alpha electron');
        test.assertEqual(mo.numBeta, 1, 'One beta electron');
        test.assertEqual(mo.numAo, 2, 'Two basis functions');

        // Test orbital energies
        const orbitalEnergies = mo.orbitalEnergies;
        test.assertEqual(orbitalEnergies.size(), 2, 'Two orbital energies');

        // HOMO should be negative, LUMO positive for H2
        const homo = orbitalEnergies.get(0);
        const lumo = orbitalEnergies.get(1);
        test.assertTrue(homo < 0, 'HOMO energy negative');
        test.assertTrue(lumo > 0, 'LUMO energy positive');
        test.assertTrue(lumo > homo, 'LUMO > HOMO');

        // Test Mulliken charges
        const mullikenCharges = wfn.mullikenCharges();
        test.assertEqual(mullikenCharges.size(), 2, 'Two Mulliken charges');

        // For symmetric H2, charges should be nearly equal
        const charge1 = mullikenCharges.get(0);
        const charge2 = mullikenCharges.get(1);
        test.assertAlmostEqual(charge1, charge2, 1e-10, 'Symmetric H2 charges equal');
        test.assertAlmostEqual(charge1 + charge2, 0.0, 1e-10, 'Charge conservation');
    });

    // IntegralEngine tests
    suite.test('IntegralEngine functionality', () => {
        const h2 = createH2Molecule();
        const atoms = h2.atoms();
        const basis = createSTO3GBasis(atoms);
        const engine = new Module.IntegralEngine(basis);

        test.assertEqual(engine.nbf(), 2, 'IntegralEngine nbf');
        test.assertEqual(engine.nsh(), 2, 'IntegralEngine nsh');
        test.assertTrue(engine.isSpherical() !== undefined, 'IntegralEngine spherical flag defined');

        // Test one-electron operators
        const overlap = engine.oneElectronOperator(Module.Operator.Overlap, true);
        const kinetic = engine.oneElectronOperator(Module.Operator.Kinetic, true);
        const nuclear = engine.oneElectronOperator(Module.Operator.Nuclear, true);

        test.assertEqual(overlap.rows(), 2, 'Engine overlap matrix size');
        test.assertEqual(kinetic.rows(), 2, 'Engine kinetic matrix size');
        test.assertEqual(nuclear.rows(), 2, 'Engine nuclear matrix size');

        // Test precision setting
        engine.setPrecision(1e-12);
        test.assertTrue(true, 'Precision setting successful');

        // Test Schwarz screening
        const schwarz = engine.schwarz();
        test.assertEqual(schwarz.rows(), 2, 'Schwarz matrix size');
        test.assertEqual(schwarz.cols(), 2, 'Schwarz matrix size');
    });

    // Test orbital enumeration
    suite.test('SpinorbitalKind enumeration', () => {
        test.assertTrue(Module.SpinorbitalKind.Restricted !== undefined, 'Restricted defined');
        test.assertTrue(Module.SpinorbitalKind.Unrestricted !== undefined, 'Unrestricted defined');
        test.assertTrue(Module.SpinorbitalKind.General !== undefined, 'General defined');

        // Should be different values
        test.assertNotEqual(Module.SpinorbitalKind.Restricted, Module.SpinorbitalKind.Unrestricted, 'R != U');
        test.assertNotEqual(Module.SpinorbitalKind.Restricted, Module.SpinorbitalKind.General, 'R != G');
        test.assertNotEqual(Module.SpinorbitalKind.Unrestricted, Module.SpinorbitalKind.General, 'U != G');
    });

    // Test operator enumeration
    suite.test('Operator enumeration', () => {
        const operators = [
            'Overlap', 'Nuclear', 'Kinetic', 'Coulomb', 'Dipole',
            'Quadrupole', 'Octapole', 'Hexadecapole', 'Rinv'
        ];

        for (const op of operators) {
            test.assertTrue(Module.Operator[op] !== undefined, `Operator ${op} defined`);
        }

        // Should be different values
        test.assertNotEqual(Module.Operator.Overlap, Module.Operator.Kinetic, 'Different operators');
        test.assertNotEqual(Module.Operator.Nuclear, Module.Operator.Coulomb, 'Different operators');
    });

    return suite.run();
}

module.exports = { runQMTests };
