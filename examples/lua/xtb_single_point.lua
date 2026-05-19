-- examples/lua/xtb_single_point.lua
-- Parallel to examples/python/xtb_single_point.py: GFN2-xTB single-point
-- energy + gradient on a molecule.
--
-- Run:
--     occ lua examples/lua/xtb_single_point.lua [water.xyz]

local path = arg[1] or "examples/scf/water.xyz"
local mol = occ.load_molecule(path)
print(string.format("Loaded %s: %s", path, tostring(mol)))

local calc = occ.XtbCalculator(mol)
calc.charge = 0
calc.include_multipoles = true

local result = calc:single_point()
if not result.converged then
    print("SCC did not converge!")
    os.exit(1)
end

print(string.format("\nConverged in %d iterations.", result.n_iterations))
print(string.format("  Total      = %.10f Ha", result.total_energy))
print(string.format("  SCC        = %.10f Ha", result.scc_energy))
print(string.format("  Repulsion  = %.10f Ha", result.repulsion_energy))
print(string.format("  Dispersion = %.10f Ha", result.dispersion_energy))

print("\nAtomic charges (Mulliken):")
for i, q in ipairs(result.atomic_charges) do
    print(string.format("  atom %d: %+.4f", i, q))
end

-- Analytical gradient (Hartree/Bohr). energy_and_gradient returns
-- a {energy, gradient} table; the gradient is a 3×N nested table.
local eg = calc:energy_and_gradient(false, 1e-3)
print(string.format("\nGradient (Hartree/Bohr) at E = %.10f Ha:", eg.energy))
local g = eg.gradient
for j = 1, #g[1] do
    print(string.format("  atom %d: %+ .6f %+ .6f %+ .6f",
        j, g[1][j], g[2][j], g[3][j]))
end
