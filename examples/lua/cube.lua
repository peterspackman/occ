-- examples/lua/cube.lua
-- Generate Gaussian-cube files (electron density + electrostatic
-- potential) from a Hartree–Fock wavefunction. Pair these with the
-- isosurface example to render the surfaces colored by ESP in your
-- favorite viewer.
--
-- Run:
--     occ lua examples/lua/cube.lua [molecule.xyz]

local xyz_path = arg[1] or "examples/scf/water.xyz"
local n = 60   -- grid points per axis

print("Loading:", xyz_path)
local mol = occ.load_molecule(xyz_path)

print("Running HF/STO-3G ...")
local basis = occ.AOBasis_load(mol:atoms(), "sto-3g")
local hf = occ.HartreeFock(basis)
local scf = hf:scf()
scf:set_charge_multiplicity(0, 1)
local energy = scf:run()
print(string.format("SCF energy = %.10f Ha", energy))

local wfn = scf:wavefunction()

print(string.format("\nGenerating %dx%dx%d cubes ...", n, n, n))

-- Both `generate_*_cube` helpers return the cube file contents as a
-- single string — write it to disk yourself.
local density_cube = occ.generate_electron_density_cube(wfn, n, n, n)
do
    local f = io.open("density.cube", "w")
    f:write(density_cube)
    f:close()
    print(string.format("  density.cube       (%d bytes)", #density_cube))
end

local esp_cube = occ.generate_esp_cube(wfn, n, n, n)
do
    local f = io.open("esp.cube", "w")
    f:write(esp_cube)
    f:close()
    print(string.format("  esp.cube           (%d bytes)", #esp_cube))
end

-- For full control over the volume (different property, grid bounds,
-- spin component, etc.), use VolumeCalculator + VolumeGenerationParameters
-- directly:
local calc = occ.VolumeCalculator()
calc:set_wavefunction(wfn)

local params = occ.VolumeGenerationParameters()
params.property = occ.VolumePropertyKind.ElectronDensity
params.adaptive_bounds = true   -- auto-size around atoms

local vol = calc:compute_volume(params)
print(string.format("\nVolumeData: %dx%dx%d = %d total points",
    vol.nx, vol.ny, vol.nz, vol.total_points))

do
    local cube = calc:volume_as_cube_string(vol)
    local f = io.open("density_adaptive.cube", "w")
    f:write(cube)
    f:close()
    print(string.format("  density_adaptive.cube (%d bytes)", #cube))
end
