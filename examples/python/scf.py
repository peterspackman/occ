from occpy import Molecule, AOBasis, HartreeFock
import occpy
import numpy as np
np.set_printoptions(precision=2)

WATER_XYZ = """3

O   -0.7021961  -0.0560603   0.0099423
H   -1.0221932   0.8467758  -0.0114887
H    0.2575211   0.0421215   0.0052190"""

occpy.set_num_threads(6)

mol = Molecule.from_xyz_string(WATER_XYZ)
print(mol)

basis = AOBasis.load(mol.atoms(), "sto-3g")
print(basis)

hf = HartreeFock(basis)
print(hf)

scf = hf.scf()
print(scf)

energy = scf.run()
print(energy)

wfn = scf.wavefunction()
print(wfn)

print("MO coefficients")
print(wfn.molecular_orbitals.orbital_coeffs)
print("MO energies")
print(wfn.molecular_orbitals.orbital_energies)
