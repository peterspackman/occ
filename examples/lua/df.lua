-- examples/lua/df.lua
-- Parallel to examples/python/df.py: DFT (BLYP/def2-qzvp) on water, with
-- and without density fitting, comparing energies and timing.
--
-- Run:
--     occ lua examples/lua/df.lua

local WATER_XYZ = [[3

O   -0.7021961  -0.0560603   0.0099423
H   -1.0221932   0.8467758  -0.0114887
H    0.2575211   0.0421215   0.0052190]]

occ.set_num_threads(6)

local function run(use_df)
    if use_df then
        print("*With* density fitting")
    else
        print("*Without* density fitting")
    end
    local t1 = os.clock()

    local mol = occ.molecule_from_xyz_string(WATER_XYZ)
    print(mol)

    local basis = occ.AOBasis_load(mol:atoms(), "def2-qzvp")
    print(basis)

    local blyp = occ.DFT("blyp", basis)
    if use_df then
        blyp:set_density_fitting_basis("def2-universal-jkfit")
    end
    print(blyp)

    local scf = blyp:scf()
    print(scf)

    local energy = scf:run()
    print(energy)

    local wfn = scf:wavefunction()
    print(wfn)

    print("MO energies")
    pp(wfn.molecular_orbitals.orbital_energies)

    local t2 = os.clock()
    print(string.format("Took %.3f s", t2 - t1))
    return energy, t2 - t1
end

local e1, t1 = run(false)
local e2, t2 = run(true)

print(string.format("Difference: %.12f au (%.1f kJ/mol)",
    e2 - e1, 2625.5 * (e2 - e1)))
print(string.format("Speed up: %.6fx", t1 / t2))
