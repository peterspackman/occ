/**
 * JavaScript tests for OCC Core module
 * 
 * Tests basic functionality of the core molecular chemistry classes
 */

const test = require('./test_framework.js');

async function runCoreTests(Module) {
    const suite = test.createSuite('Core Module Tests');
    
    // Element tests
    suite.test('Element creation and properties', () => {
        const h = new Module.Element('H');
        test.assertEqual(h.symbol, 'H', 'Hydrogen symbol');
        test.assertEqual(h.atomicNumber, 1, 'Hydrogen atomic number');
        test.assertTrue(h.mass > 1.0 && h.mass < 1.1, 'Hydrogen mass in range');
        
        const c = Module.Element.fromAtomicNumber(6);
        test.assertEqual(c.symbol, 'C', 'Carbon symbol from atomic number');
        test.assertEqual(c.atomicNumber, 6, 'Carbon atomic number');
    });
    
    // Atom tests
    suite.test('Atom creation and manipulation', () => {
        const atom = new Module.Atom(1, 0.0, 0.0, 1.4);
        test.assertEqual(atom.atomicNumber, 1, 'Atom atomic number');
        test.assertAlmostEqual(atom.x, 0.0, 1e-10, 'Atom x coordinate');
        test.assertAlmostEqual(atom.y, 0.0, 1e-10, 'Atom y coordinate');
        test.assertAlmostEqual(atom.z, 1.4, 1e-10, 'Atom z coordinate');
        
        const pos = atom.getPosition();
        test.assertAlmostEqual(pos.z(), 1.4, 1e-10, 'Position vector z component');
    });
    
    // PointCharge tests
    suite.test('PointCharge functionality', () => {
        const pc = new Module.PointCharge(1.5, 1.0, 2.0, 3.0);
        test.assertAlmostEqual(pc.charge, 1.5, 1e-10, 'Point charge value');
        
        const pos = pc.getPosition();
        test.assertAlmostEqual(pos.x(), 1.0, 1e-10, 'Point charge x position');
        test.assertAlmostEqual(pos.y(), 2.0, 1e-10, 'Point charge y position');
        test.assertAlmostEqual(pos.z(), 3.0, 1e-10, 'Point charge z position');
    });
    
    // Molecule tests
    suite.test('H2 molecule creation and properties', () => {
        // Create H2 molecule
        const positions = Module.Mat3N.create(3, 2);
        positions.set(0, 0, 0.0);  // H1: (0, 0, 0)
        positions.set(1, 0, 0.0);
        positions.set(2, 0, 0.0);
        positions.set(0, 1, 0.0);  // H2: (0, 0, 1.4)
        positions.set(1, 1, 0.0);
        positions.set(2, 1, 1.4);
        
        const atomicNumbers = Module.IVec.fromArray([1, 1]);
        const h2 = new Module.Molecule(atomicNumbers, positions);
        
        test.assertEqual(h2.size(), 2, 'H2 has 2 atoms');
        test.assertAlmostEqual(h2.molarMass(), 0.0020159, 1e-6, 'H2 molar mass');
        
        const com = h2.centerOfMass();
        test.assertAlmostEqual(com.x(), 0.0, 1e-10, 'H2 center of mass x');
        test.assertAlmostEqual(com.y(), 0.0, 1e-10, 'H2 center of mass y');
        test.assertAlmostEqual(com.z(), 0.7, 1e-10, 'H2 center of mass z');
        
        const centroid = h2.centroid();
        test.assertAlmostEqual(centroid.z(), 0.7, 1e-10, 'H2 centroid z');
    });
    
    // Water molecule tests
    suite.test('Water molecule creation and properties', () => {
        const positions = Module.Mat3N.create(3, 3);
        
        // O at origin
        positions.set(0, 0, 0.0);
        positions.set(1, 0, 0.0);
        positions.set(2, 0, 0.0);
        
        // H atoms (approximately tetrahedral)
        positions.set(0, 1, 1.43);
        positions.set(1, 1, 1.1);
        positions.set(2, 1, 0.0);
        
        positions.set(0, 2, -1.43);
        positions.set(1, 2, 1.1);
        positions.set(2, 2, 0.0);
        
        const atomicNumbers = Module.IVec.fromArray([8, 1, 1]);
        const water = new Module.Molecule(atomicNumbers, positions);
        water.setName("Water");
        
        test.assertEqual(water.size(), 3, 'Water has 3 atoms');
        test.assertEqual(water.name, "Water", 'Water molecule name');
        test.assertAlmostEqual(water.molarMass(), 0.018015, 1e-5, 'Water molar mass');
        
        const masses = water.atomicMasses();
        test.assertTrue(masses.get(0) > 15.0, 'Oxygen mass > 15');
        test.assertTrue(masses.get(1) < 2.0, 'Hydrogen mass < 2');
        test.assertTrue(masses.get(2) < 2.0, 'Hydrogen mass < 2');
    });
    
    // Molecular transformations
    suite.test('Molecular transformations', () => {
        // Create simple molecule
        const positions = Module.Mat3N.create(2, 3);
        positions.set(0, 0, 0.0);
        positions.set(1, 0, 0.0);
        positions.set(2, 0, 0.0);
        positions.set(0, 1, 1.0);
        positions.set(1, 1, 0.0);
        positions.set(2, 1, 0.0);
        
        const atomicNumbers = Module.IVec.fromArray([1, 1]);
        const mol = new Module.Molecule(atomicNumbers, positions);
        
        // Test translation
        const translation = Module.Vec3.create(1.0, 2.0, 3.0);
        const translated = mol.translated(translation);
        
        const newCom = translated.centerOfMass();
        const originalCom = mol.centerOfMass();
        
        test.assertAlmostEqual(newCom.x(), originalCom.x() + 1.0, 1e-10, 'Translation x');
        test.assertAlmostEqual(newCom.y(), originalCom.y() + 2.0, 1e-10, 'Translation y');
        test.assertAlmostEqual(newCom.z(), originalCom.z() + 3.0, 1e-10, 'Translation z');
        
        // Test centering
        const centered = mol.centered(Module.Origin.CENTEROFMASS);
        const centeredCom = centered.centerOfMass();
        test.assertAlmostEqual(centeredCom.x(), 0.0, 1e-10, 'Centered COM x');
        test.assertAlmostEqual(centeredCom.y(), 0.0, 1e-10, 'Centered COM y');
        test.assertAlmostEqual(centeredCom.z(), 0.0, 1e-10, 'Centered COM z');
    });
    
    // Dimer tests
    suite.test('Dimer creation and analysis', () => {
        // Create two simple molecules
        const pos1 = Module.Mat3N.create(1, 3);
        pos1.set(0, 0, 0.0);
        pos1.set(1, 0, 0.0);
        pos1.set(2, 0, 0.0);
        
        const pos2 = Module.Mat3N.create(1, 3);
        pos2.set(0, 0, 5.0);
        pos2.set(1, 0, 0.0);
        pos2.set(2, 0, 0.0);
        
        const an1 = Module.IVec.fromArray([1]);
        const an2 = Module.IVec.fromArray([1]);
        
        const mol1 = new Module.Molecule(an1, pos1);
        const mol2 = new Module.Molecule(an2, pos2);
        
        const dimer = new Module.Dimer(mol1, mol2);
        dimer.setName("H-H dimer");
        
        test.assertEqual(dimer.name, "H-H dimer", 'Dimer name');
        test.assertAlmostEqual(dimer.nearestDistance, 5.0, 1e-10, 'Dimer nearest distance');
        test.assertAlmostEqual(dimer.centerOfMassDistance, 5.0, 1e-10, 'Dimer COM distance');
        test.assertAlmostEqual(dimer.centroidDistance, 5.0, 1e-10, 'Dimer centroid distance');
    });
    
    // Point group analysis
    suite.test('Point group analysis', () => {
        // Create linear H2 molecule (D∞h symmetry)
        const positions = Module.Mat3N.create(2, 3);
        positions.set(0, 0, 0.0);
        positions.set(1, 0, 0.0);
        positions.set(2, 0, -0.7);
        positions.set(0, 1, 0.0);
        positions.set(1, 1, 0.0);
        positions.set(2, 1, 0.7);
        
        const atomicNumbers = Module.IVec.fromArray([1, 1]);
        const h2 = new Module.Molecule(atomicNumbers, positions);
        
        const pointGroup = new Module.MolecularPointGroup(h2);
        
        test.assertTrue(pointGroup.getPointGroupString().length > 0, 'Point group string not empty');
        test.assertTrue(pointGroup.getDescription().length > 0, 'Point group description not empty');
        test.assertTrue(pointGroup.symmetryNumber >= 1, 'Symmetry number >= 1');
        
        // For H2, should have high symmetry
        test.assertTrue(pointGroup.symmetryNumber >= 2, 'H2 symmetry number >= 2');
    });
    
    // Partial charges
    suite.test('Partial charge calculations', () => {
        // Create water molecule
        const positions = Module.Mat3N.create(3, 3);
        positions.set(0, 0, 0.0);
        positions.set(1, 0, 0.0);
        positions.set(2, 0, 0.0);
        positions.set(0, 1, 1.43);
        positions.set(1, 1, 1.1);
        positions.set(2, 1, 0.0);
        positions.set(0, 2, -1.43);
        positions.set(1, 2, 1.1);
        positions.set(2, 2, 0.0);
        
        const atomicNumbers = Module.IVec.fromArray([8, 1, 1]);
        const water = new Module.Molecule(atomicNumbers, positions);
        
        const eemCharges = Module.eemPartialCharges(water.atomicNumbers(), water.positions(), 0.0);
        const eeqCharges = Module.eeqPartialCharges(water.atomicNumbers(), water.positions(), 0.0);
        
        test.assertEqual(eemCharges.size(), 3, 'EEM charges size');
        test.assertEqual(eeqCharges.size(), 3, 'EEQ charges size');
        
        // Check charge conservation
        let eemTotal = 0.0;
        let eeqTotal = 0.0;
        for (let i = 0; i < 3; i++) {
            eemTotal += eemCharges.get(i);
            eeqTotal += eeqCharges.get(i);
        }
        
        test.assertAlmostEqual(eemTotal, 0.0, 1e-6, 'EEM total charge conservation');
        test.assertAlmostEqual(eeqTotal, 0.0, 1e-6, 'EEQ total charge conservation');
        
        // Oxygen should be more negative than hydrogen
        test.assertTrue(eemCharges.get(0) < eemCharges.get(1), 'EEM oxygen more negative');
        test.assertTrue(eeqCharges.get(0) < eeqCharges.get(1), 'EEQ oxygen more negative');
    });
    
    return suite.run();
}

module.exports = { runCoreTests };