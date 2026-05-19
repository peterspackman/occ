-- examples/lua/isosurface.lua
-- Parallel to examples/python/isosurface.py: generate promolecule-density
-- isosurfaces for water at several grid spacings, saving each as PLY.
--
-- Run:
--     occ lua examples/lua/isosurface.lua [molecule.xyz]
-- (defaults to examples/scf/water.xyz)

local WATER_XYZ = [[3

O   -0.7021961  -0.0560603   0.0099423
H   -1.0221932   0.8467758  -0.0114887
H    0.2575211   0.0421215   0.0052190]]

-- Load from file if given, else fall back to the inline water geometry.
local mol
if arg[1] then
    mol = occ.load_molecule(arg[1])
else
    mol = occ.molecule_from_xyz_string(WATER_XYZ)
end
print("Loaded:", mol)

for _, sep in ipairs({1.0, 0.5, 0.25, 0.1, 0.05}) do
    local t1 = os.clock()

    local params = occ.IsosurfaceGenerationParameters()
    params.isovalue = 0.02
    params.surface_kind = occ.SurfaceKind.PromoleculeDensity
    params.separation = sep

    local calc = occ.IsosurfaceCalculator()
    calc:set_parameters(params)
    calc:set_molecule(mol)
    calc:compute()

    local surf = calc.isosurface
    surf:save(string.format("example.%s.ply", tostring(sep)))

    local t2 = os.clock()
    print(string.format("Surface with separation = %.3f took %.3fs",
        sep, t2 - t1))
end
