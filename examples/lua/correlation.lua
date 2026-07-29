-- examples/lua/correlation.lua
-- Post-HF correlation methods (MP2 / CCSD / CCSD(T)) on water.
--
-- Run:
--     occ lua examples/lua/correlation.lua

local WATER_XYZ = [[3

O   -0.7021961  -0.0560603   0.0099423
H   -1.0221932   0.8467758  -0.0114887
H    0.2575211   0.0421215   0.0052190]]

local mol = occ.molecule_from_xyz_string(WATER_XYZ)
local basis = occ.AOBasis.load(mol:atoms(), "6-31g")

local hf = occ.HartreeFock(basis)
local scf = hf:scf()
scf:run()
local wfn = scf:wavefunction()
print(string.format("SCF total energy:      %18.10f", wfn.total_energy))

-- High-level API: method string (or a table / occ.CorrelationOptions).
-- Backend and auxiliary basis are resolved exactly like the CLI.
local mp2 = occ.run_correlation(wfn, "mp2")
print(string.format("MP2 correlation:       %18.10f", mp2.correlation_energy))

local rimp2 = occ.run_correlation(wfn, { method = "mp2", backend = "df" })
print(string.format("RI-MP2 correlation:    %18.10f", rimp2.correlation_energy))

local ccsdt = occ.run_correlation(wfn, "ccsd(t)")
print(string.format("CCSD correlation:      %18.10f", ccsdt.ccsd_correlation))
print(string.format("(T) correction:        %18.10f", ccsdt.triples_correction))
print(string.format("CCSD(T) total energy:  %18.10f", ccsdt.total_energy))

-- Low-level API: build MO integrals and drive the CCSD solver directly.
local nfz = occ.num_frozen_core(wfn.basis)
local eris = occ.exact_eris(wfn.basis, wfn.molecular_orbitals, nfz)
print(eris)
local res = occ.ccsd(eris)
print(string.format("CCSD (low level):      %18.10f after %d iterations",
                    res.e_corr, res.iterations))
local et = occ.ccsd_t(res, eris)
print(string.format("(T)  (low level):      %18.10f", et))
