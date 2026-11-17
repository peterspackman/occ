-- examples/lua/neb.lua
-- Parallel to examples/python/neb.py: a nudged elastic band from HCN to HNC
-- with GFN2-xTB, then a saddle point search from the climbing image along
-- the band's tangent.
--
-- The command line equivalent:
--     occ scf hcn.xyz gfn2 --driver neb --product hnc.xyz --neb-ts
--
-- Run:
--     occ lua examples/lua/neb.lua

local function gfn2(molecule)
    local result = occ.XtbCalculator(molecule):energy_and_gradient(false, 1e-3)
    return result.energy, result.gradient
end

-- GFN2-xTB minima, the same atoms in the same order (Angstrom, rows x, y, z).
-- Not exactly linear: on a straight line the hydrogen could not leave the axis.
local hcn = occ.Molecule({6, 7, 1}, {
    {0.0036, 1.1412, -1.0548}, {0.0069, -0.0032, 0.0162}, {0.0, 0.0, 0.0}})
local hnc = occ.Molecule({6, 7, 1}, {
    {0.0086, 1.1670, 2.1645}, {-0.0033, 0.0072, 0.0160}, {0.0, 0.0, 0.0}})

local settings = occ.opt.NebSettings()
settings.images = 8
-- A rough band is enough when the saddle point search refines it (as --neb-ts)
settings.force_max = 2e-2
settings.climbing_force_max = 2e-3
local band = occ.opt.NudgedElasticBand(hcn, hnc, settings)
band:set_endpoint_energies((gfn2(hcn)), (gfn2(hnc)))

local iterations = 0
for iteration = 0, 199 do
    local energies, gradients = {}, {}
    for k, image in ipairs(band:images()) do
        energies[k], gradients[k] = gfn2(image)
    end
    band:update(energies, gradients)
    iterations = iteration
    if band:step() then break end
end

local energies = band:energies()
local top = band.highest_image  -- counted from 0, endpoints included
print(string.format("Band converged after %d iterations; profile (kJ/mol):", iterations))
for k = 1, energies:size() do
    print(string.format("  %2d %9.2f%s", k - 1, (energies[k] - energies[1]) * 2625.4996,
                        k - 1 == top and "  <- climbing" or ""))
end

-- Refine the climbing image to the saddle point, starting along the band
local guess = band:path()[top + 1]
local optimizer = occ.opt.GeometryOptimizer.new_with_settings(
    guess, occ.opt.ConvergenceCriteria(), occ.opt.OptimizerSettings.saddle_point(0))
optimizer:set_initial_hessian(occ.XtbCalculator(guess):hessian(0.005))
optimizer:set_followed_mode(band:tangent(top))
for _ = 1, 50 do
    optimizer:update(gfn2(optimizer:get_next_geometry()))
    if optimizer:step() then break end
end
print(string.format("\nTransition state: E = %.8f Ha", optimizer.current_energy))
print(optimizer:get_next_geometry())
