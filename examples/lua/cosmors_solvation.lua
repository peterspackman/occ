-- examples/lua/cosmors_solvation.lua
-- Parallel to examples/python/cosmors_solvation.py: an openCOSMO-RS 24a
-- solvation free energy, term by term.
--
-- Run:
--     occ lua examples/lua/cosmors_solvation.lua [water.xyz] [solvent]

local path = arg[1] or "examples/scf/water.xyz"
local solvent = arg[2] or "water"

-- Hartree to kJ/mol.
local AU_TO_KJ = 2625.499639479

local available = occ.available_cosmo_rs_solvents()
local found = false
for _, name in ipairs(available) do
    if name == solvent then found = true end
end
if not found then
    print(string.format("no segment ensemble for '%s'; have: %s",
        solvent, table.concat(available, ", ")))
    os.exit(1)
end

local mol = occ.load_molecule(path)
print(string.format("Loaded %s: %s", path, tostring(mol)))

local settings = occ.CosmoRSSettings()
settings.basis = "6-31g**"
settings.temperature = 298.15
-- Liquid-phase volume per solute molecule, Angstrom^3. Leaving it at zero
-- drops the reference-state term, so the total is no longer on an absolute
-- scale.
settings.liquid_volume = 30.01

local result = occ.cosmo_rs_solvation_free_energy(mol, solvent, settings)
local e = result.energy

print(string.format("\ncavity: %.2f A^2, %.2f A^3",
    result.cavity_area, result.cavity_volume))
print(string.format("rings:  %d", result.num_rings))
print(string.format("\nSolvation free energy in %s (kJ/mol):", solvent))

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
