const tap = require('tap');
const Module = require('../src/occ_wasm.js');

async function runTests() {
  const occ = await Module();

  tap.test('Element class', async (t) => {
    const element = new occ.Element(1);
    t.equal(element.symbol, "H", "Element symbol matches");
    t.equal(element.atomic_number, 1, "Atomic number matches");
    t.end();
  });

  tap.test('Vec3', async (t) => {
    const vec = new occ.Vec3();
    t.ok(vec, "Vector3d created successfully");
    t.end();
  });

  tap.test('Molecule', async (t) => {
    // Create simple water molecule
    const positions = new occ.Mat3N();
    positions.resize(3, 3);
    positions.data([
      0.0, 0.757, 0.757,  // x coordinates
      0.0, 0.586, -0.586, // y coordinates
      0.0, 0.0, 0.0       // z coordinates
    ]);

    const atomicNumbers = new occ.IVec();
    atomicNumbers.resize(3);
    atomicNumbers.data([ 8, 1, 1 ]); // O, H, H
    const mol = new occ.Molecule(atomicNumbers, positions);
    t.equal(mol.size(), 3, "Molecule has correct number of atoms");
    t.ok(mol.atoms(), "Molecule.atoms() works");
    t.end();
  });

  tap.test('Point Charge', async (t) => {
    const pc = new occ.PointCharge(1.0, 0.0, 0.0, 0.0);
    t.equal(pc.charge, 1.0, "Point charge has correct value");
    t.end();
  });
}

runTests().catch(console.error);
