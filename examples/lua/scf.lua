-- examples/lua/scf.lua
-- Parallel to examples/python/scf.py: Hartree–Fock single point on water.
--
-- Run:
--     occ lua examples/lua/scf.lua

local WATER_XYZ = [[3

O   -0.7021961  -0.0560603   0.0099423
H   -1.0221932   0.8467758  -0.0114887
H    0.2575211   0.0421215   0.0052190]]

occ.set_num_threads(6)

local mol = occ.molecule_from_xyz_string(WATER_XYZ)
print(mol)

local basis = occ.AOBasis_load(mol:atoms(), "sto-3g")
print(basis)

local hf = occ.HartreeFock(basis)
print(hf)

local scf = hf:scf()
print(scf)

local energy = scf:run()
print(energy)

local wfn = scf:wavefunction()
print(wfn)

print("MO coefficients")
pp(wfn.molecular_orbitals.orbital_coeffs)
print("MO energies")
pp(wfn.molecular_orbitals.orbital_energies)
