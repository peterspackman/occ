-- examples/lua/water_xtb.lua
-- Stage-1 smoke test: load a molecule, run a GFN2-xTB single point,
-- inspect the result. Run with:
--     occ lua examples/lua/water_xtb.lua examples/scf/water.xyz

local input = arg[1] or "examples/scf/water.xyz"

local mol = occ.load_molecule(input)
print(string.format("Loaded %s (%d atoms)", input, mol:size()))

local calc = occ.XtbCalculator(mol)
calc.charge = 0
calc.include_multipoles = true

local result = calc:single_point()

print(string.format("Converged    : %s in %d iterations",
                    tostring(result.converged), result.n_iterations))
print(string.format("Total energy : %.10f Ha", result.total_energy))
print(string.format("  SCC        : %.10f Ha", result.scc_energy))
print(string.format("  Repulsion  : %.10f Ha", result.repulsion_energy))
print(string.format("  Dispersion : %.10f Ha", result.dispersion_energy))

io.write("Atomic charges :")
for i, q in ipairs(result.atomic_charges) do
    io.write(string.format("  %+.4f", q))
end
io.write("\n")

local eg = calc:energy_and_gradient(false, 1e-3)
print(string.format("Energy (eg)  : %.10f Ha", eg.energy))
print("Gradient (Hartree/Bohr):")
-- Mat3N convention: g[1], g[2], g[3] are the x, y, z rows;
-- g[1][j] is x of atom j (Eigen column-major, matches numpy).
local g = eg.gradient
for j = 1, #g[1] do
    print(string.format("  atom %d: % .6f % .6f % .6f",
        j, g[1][j], g[2][j], g[3][j]))
end
