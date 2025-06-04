/**
 * JavaScript tests for OCC QM module
 * 
 * Tests quantum mechanical functionality including basis sets, SCF, and wavefunctions
 */

const test = require('./test_framework.js');

async function runQMTests(Module) {
    const suite = test.createSuite('QM Module Tests');
    
    // Create a simple H2 molecule for testing
    function createH2Molecule() {
        const positions = Module.Mat3N.create(2, 3);
        positions.set(0, 0, 0.0);
        positions.set(1, 0, 0.0);
        positions.set(2, 0, 0.0);
        positions.set(0, 1, 0.0);
        positions.set(1, 1, 0.0);
        positions.set(2, 1, 1.4);
        
        const atomicNumbers = Module.IVec.fromArray([1, 1]);
        const h2 = new Module.Molecule(atomicNumbers, positions);
        h2.setName("H2");
        return h2;
    }
    
    // Basis set tests
    suite.test('AOBasis loading and properties', () => {
        const h2 = createH2Molecule();
        const atoms = h2.atoms();
        const basis = Module.AOBasis.load(atoms, "sto-3g");
        
        test.assertEqual(basis.name(), "sto-3g", 'Basis set name');
        test.assertEqual(basis.size(), 2, 'Number of shells for H2/STO-3G');
        test.assertEqual(basis.nbf(), 2, 'Number of basis functions for H2/STO-3G');
        test.assertTrue(basis.lMax() >= 0, 'Maximum angular momentum >= 0');
        
        const shells = basis.shells();
        test.assertEqual(shells.size(), 2, 'Shells vector size');
        
        // Test first shell properties
        const shell0 = shells.get(0);
        test.assertTrue(shell0.numPrimitives() > 0, 'Shell has primitives');
        test.assertTrue(shell0.numContractions() > 0, 'Shell has contractions');
        test.assertTrue(shell0.norm() > 0, 'Shell norm > 0');
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
        const basis = Module.AOBasis.load(atoms, "sto-3g");
        const hf = new Module.HartreeFock(basis);
        
        // Test nuclear repulsion
        const nuclearRepulsion = hf.nuclearRepulsion();
        test.assertTrue(nuclearRepulsion > 0, 'Nuclear repulsion > 0');
        test.assertAlmostEqual(nuclearRepulsion, 1.0/1.4, 1e-6, 'Nuclear repulsion for H2');
        
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
    
    // Complete SCF calculation test
    suite.test('Complete H2 SCF calculation', () => {
        const h2 = createH2Molecule();
        const atoms = h2.atoms();
        const basis = Module.AOBasis.load(atoms, "sto-3g");
        const hf = new Module.HartreeFock(basis);
        
        // Create SCF object
        const scf = new Module.HF(hf, Module.SpinorbitalKind.Restricted);
        
        // Set tight convergence
        const convergence = scf.convergenceSettings;
        convergence.energyThreshold = 1e-8;
        convergence.commutatorThreshold = 1e-6;
        
        // Set charge and multiplicity
        scf.setChargeMultiplicity(0, 1);
        
        // Run SCF
        const energy = scf.run();
        
        // Check energy is reasonable for H2/STO-3G
        test.assertTrue(energy < 0, 'SCF energy is negative');
        test.assertTrue(energy > -2.0, 'SCF energy not too negative');
        
        // Known approximate value for H2/STO-3G at 1.4 Bohr
        test.assertAlmostEqual(energy, -1.11675930, 1e-6, 'H2/STO-3G energy');
        
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
        const basis = Module.AOBasis.load(atoms, "sto-3g");
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