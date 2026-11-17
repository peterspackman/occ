-- examples/lua/transition_state.lua
-- Parallel to examples/python/transition_state.py: the HCN -> HNC transition
-- state with GFN2-xTB, from a rough guess and a computed Hessian.
--
-- The command line equivalent: occ scf guess.xyz gfn2 --driver ts
--
-- Run:
--     occ lua examples/lua/transition_state.lua

-- A rough guess, H bridging C and N (Angstrom, rows x, y, z; columns atoms)
local guess = occ.Molecule({6, 7, 1}, {
    {0.0, 1.18, 0.45},
    {0.0, 0.0, 1.12},
    {0.0, 0.0, 0.0}})

local optimizer = occ.opt.GeometryOptimizer.new_with_settings(
    guess, occ.opt.ConvergenceCriteria(), occ.opt.OptimizerSettings.saddle_point(0))
-- A saddle point search needs the negative curvature a model cannot give
optimizer:set_initial_hessian(occ.XtbCalculator(guess):hessian(0.005))

for step = 0, 49 do
    local calc = occ.XtbCalculator(optimizer:get_next_geometry())
    local result = calc:energy_and_gradient(false, 1e-3)
    print(string.format("step %2d: E = %.8f Ha", step, result.energy))
    optimizer:update(result.energy, result.gradient)
    if optimizer:step() then break end
end

local ts = optimizer:get_next_geometry()
print(string.format("\nConverged: %s, E = %.8f Ha", tostring(optimizer.is_converged),
                    optimizer.current_energy))
print(ts)

local modes = occ.XtbCalculator(ts):vibrational_modes(0.005, true)
local frequencies = modes.get_all_frequencies
for i = 1, frequencies:size() do
    if frequencies[i] < -50 then  -- one, about -1426
        print(string.format("Imaginary mode: %.1fi cm-1", -frequencies[i]))
    end
end
