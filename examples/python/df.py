from occpy import Molecule, AOBasis, DFT
import occpy
import numpy as np
import time

np.set_printoptions(precision=2)

WATER_XYZ = """3

O   -0.7021961  -0.0560603   0.0099423
H   -1.0221932   0.8467758  -0.0114887
H    0.2575211   0.0421215   0.0052190"""

occpy.set_num_threads(6)


def run(df=False):
    if df:
        print("*With* density fitting")
    else:
        print("*Without* density fitting")
    t1 = time.time()
    mol = Molecule.from_xyz_string(WATER_XYZ)
    print(mol)

    basis = AOBasis.load(mol.atoms(), "def2-qzvp")
    print(basis)

    blyp = DFT("blyp", basis)
    if df:
        blyp.set_density_fitting_basis("def2-universal-jkfit")
    print(blyp)

    scf = blyp.scf()
    print(scf)

    energy = scf.run()
    print(energy)

    wfn = scf.wavefunction()
    print(wfn)

    print("MO coefficients")
    print(wfn.molecular_orbitals.orbital_coeffs)
    print("MO energies")
    print(wfn.molecular_orbitals.orbital_energies)
    t2 = time.time()
    print(f"Took {t2 - t1:.3f} s")
    return energy, t2 - t1

e1, t1 = run()
e2, t2 = run(df=True)

print(f"Difference: {e2 - e1:.12f} au ({2625.5 * (e2 - e1):.1f} kJ/mol)")
print(f"Speed up: {t1 / t2:.6f}x")
