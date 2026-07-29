-- examples/lua/xtb_open_shell.lua
-- Spin-unrestricted GFN2-xTB on a radical. Parallel to
-- examples/python/xtb_open_shell.py.
--
-- Run:
--     occ lua examples/lua/xtb_open_shell.lua

local BOHR = 1.8897261246257702
local r = 1.078
local c = 0.8660254037844386 * r

-- Planar CH3 radical, positions in Angstrom as a 3×N table (rows are x/y/z).
local mol = occ.Molecule({6, 1, 1, 1}, {
    {0.0, r, -0.5 * r, -0.5 * r},
    {0.0, 0.0, c, -c},
    {0.0, 0.0, 0.0, 0.0},
})

local calc = occ.XtbCalculator(mol)
calc.num_unpaired_electrons = 1 -- doublet; equivalently mol:set_multiplicity(2)

local result = calc:single_point()
if not result.converged then
    print("SCC did not converge!")
    os.exit(1)
end

print(string.format("Total energy       = %.10f Ha", result.total_energy))
print(string.format("  spin polarization = %.10f Ha", result.spin_energy))

print("\n  atom      charge        spin")
local q, s = result.atomic_charges, result.atomic_magnetization
for i = 1, calc.num_atoms do
    print(string.format("  %4d  %+10.6f  %+10.6f", i, q[i], s[i]))
end

-- Zero scale drops the spin coupling: alpha and beta share one Hamiltonian,
-- which is what plain `xtb --uhf` computes.
local unpolarized = occ.XtbCalculator(mol)
unpolarized.num_unpaired_electrons = 1
unpolarized.spin_polarization = 0.0
print(string.format("\nwithout spin polarization: %.10f Ha",
    unpolarized:single_point_energy()))
