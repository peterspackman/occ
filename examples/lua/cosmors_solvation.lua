-- examples/lua/cosmors_solvation.lua
-- Parallel to examples/python/cosmors_solvation.py: an openCOSMO-RS 24a
-- solvation free energy, term by term.
--
-- occ ships no solvent ensembles, so this computes the solvent's too, from
-- its geometry. Pass the name of a cached ensemble instead (found on
-- $OCC_DATA_PATH/solvent/cosmors or the working directory) to skip that.
--
-- Run:
--     occ lua examples/lua/cosmors_solvation.lua [solute.xyz] [solvent]
--
-- where [solvent] is either a .xyz geometry or a cached ensemble name.

local path = arg[1] or "examples/scf/water.xyz"
local solvent = arg[2] or "examples/scf/water.xyz"

-- Hartree to kJ/mol.
local AU_TO_KJ = 2625.499639479

local mol = occ.load_molecule(path)
print(string.format("Loaded %s: %s", path, tostring(mol)))

-- A cached ensemble is used if the name resolves; otherwise the solvent is
-- treated as a geometry and its cavity computed alongside the solute's.
local use_cached = false
for _, name in ipairs(occ.available_cosmo_rs_solvents()) do
    if name == solvent then use_cached = true end
end

local solvent_mol
if use_cached then
    print(string.format("Using the cached ensemble for '%s'", solvent))
else
    solvent_mol = occ.load_molecule(solvent)
    print(string.format("Computing the solvent cavity from %s: %s",
        solvent, tostring(solvent_mol)))
end

local settings = occ.CosmoRSSettings()
settings.basis = "6-31g**"
settings.temperature = 298.15
-- Liquid-phase volume per solute molecule, Angstrom^3. Leaving it at zero
-- drops the reference-state term, so the total is no longer on an absolute
-- scale.
settings.liquid_volume = 30.01

local result
if use_cached then
    result = occ.cosmo_rs_solvation_free_energy(mol, solvent, settings)
else
    result = occ.cosmo_rs_solvation_free_energy_with_solvent_geometry(
        mol, solvent_mol, settings)
end
local e = result.energy

print(string.format("\ncavity: %.2f A^2, %.2f A^3",
    result.cavity_area, result.cavity_volume))
print(string.format("rings:  %d", result.num_rings))
local label = use_cached and solvent
    or string.format("the geometry in %s", solvent)
print(string.format("\nSolvation free energy in %s (kJ/mol):", label))

local terms = {
    { "dielectric", e.dielectric, "gas -> ideal conductor" },
    { "residual", e.residual, "RT ln(gamma_res)" },
    { "combinatorial", e.combinatorial, "RT ln(gamma_comb)" },
    { "van der Waals", e.vdw, "-sum_a tau_a A_a" },
    { "ring", e.ring, "-omega_ring n_ring" },
    { "reference state", e.reference_state, "-RT ln(v_gas/v_liquid)" },
    { "eta", e.eta, "fitted intercept" },
}
for _, term in ipairs(terms) do
    print(string.format("  %-16s %9.3f   %s",
        term[1], term[2] * AU_TO_KJ, term[3]))
end
print(string.format("  %-16s %9.3f", "total", e:total() * AU_TO_KJ))
