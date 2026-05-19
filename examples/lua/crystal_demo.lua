-- examples/lua/crystal_demo.lua
-- Stage-2 crystal binding smoke test.
-- Run: occ lua examples/lua/crystal_demo.lua examples/external_energy/acenaphthene.cif

local cif_path = arg[1] or "examples/external_energy/acenaphthene.cif"

local crystal = occ.crystal_from_cif_file(cif_path)
print("Crystal:", tostring(crystal))

local cell = crystal.unit_cell
print(string.format("Cell  : %s   %.4f × %.4f × %.4f Å",
    cell.cell_type, cell.a, cell.b, cell.c))
print(string.format("Angles: α=%.2f° β=%.2f° γ=%.2f°",
    math.deg(cell.alpha), math.deg(cell.beta), math.deg(cell.gamma)))
print(string.format("Volume: %.3f Å³,  Density: %.3f g/cm³",
    cell.volume, crystal.density))

local sg = crystal.space_group
print(string.format("Space group : %s  (No. %d, %d symmetry ops)",
    sg.symbol, sg.number, #sg:symmetry_operations()))

local asym = crystal.asymmetric_unit
print(string.format("Asym unit   : %d atoms  (%s)", #asym, tostring(asym)))

local uc_mols = crystal:unit_cell_molecules()
print(string.format("Unit-cell molecules: %d", #uc_mols))

local uniq_mols = crystal:symmetry_unique_molecules()
print(string.format("Symmetry-unique mols: %d", #uniq_mols))
for i, mol in ipairs(uniq_mols) do
    print(string.format("  mol %d: %d atoms, molar mass %.3f g/mol",
        i, #mol, mol.molar_mass))
end

-- A few free-function checks
local hkl = occ.HKL(1, 1, 0)
print(string.format("HKL %s  d-spacing at this cell = %.4f Å", tostring(hkl),
    hkl:d(cell.reciprocal)))

local cubic = occ.cubic_cell(5.0)
print(string.format("Cubic cell test: a=%.3f b=%.3f c=%.3f V=%.3f",
    cubic.a, cubic.b, cubic.c, cubic.volume))
