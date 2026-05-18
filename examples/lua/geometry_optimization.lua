-- examples/lua/geometry_optimization.lua
-- Parallel to examples/python/geometry_optimization.py: HF/3-21G + Berny
-- optimizer + numerical Hessian / vibrational analysis on water.
--
-- Run:
--     occ lua examples/lua/geometry_optimization.lua

print("OCC Lua Optimization Example")
print("================================")

-- 1. Create a water molecule with distorted geometry.
local molecule = occ.Molecule(
    {8, 1, 1},   -- O, H, H
    {{0.0,  0.0, 0.0},      -- Oxygen
     {0.0,  0.8, 0.6},      -- H (elongated)
     {0.0, -0.8, 0.6}})     -- H
print("Initial geometry:")
print(molecule)

-- 2. Helper: energy + Cartesian gradient at a given geometry.
local function compute_energy_gradient(mol)
    local basis = occ.AOBasis_load(mol:atoms(), "3-21G")
    local hf = occ.HartreeFock(basis)

    local scf = occ.HF(hf)
    scf:set_charge_multiplicity(0, 1)  -- neutral singlet
    local energy = scf:run()

    local wfn = scf:wavefunction()
    local gradient = hf:compute_gradient(wfn.molecular_orbitals)
    return energy, gradient, hf, wfn
end

-- 3. Optimizer settings.
local criteria = occ.opt.ConvergenceCriteria()
criteria.gradient_max = 1e-4
criteria.gradient_rms = 1e-5

print("\nRunning geometry optimization...")
local optimizer = occ.opt.BernyOptimizer(molecule, criteria)

local converged = false
local max_steps = 25
local final_hf, final_wfn = nil, nil

for step = 1, max_steps do
    local current_mol = optimizer:get_next_geometry()
    local energy, gradient, hf, wfn = compute_energy_gradient(current_mol)

    optimizer:update(energy, gradient)

    -- |gradient| as a Frobenius-like norm over the 3xN table.
    local sum = 0
    for i = 1, #gradient do
        for j = 1, #gradient[i] do
            sum = sum + gradient[i][j] * gradient[i][j]
        end
    end
    print(string.format("Step %d: E = %.8f Ha, |grad| = %.6f",
        step, energy, math.sqrt(sum)))

    final_hf, final_wfn = hf, wfn

    -- The C++ Berny optimizer can throw `"trust radius got too small"` if
    -- the starting geometry is too far from the minimum; catch that so we
    -- still print the partial trajectory cleanly.
    local ok, stepped_converged = pcall(function() return optimizer:step() end)
    if not ok then
        print(string.format("Optimizer aborted: %s", tostring(stepped_converged)))
        break
    end
    if stepped_converged then
        print(string.format("Optimization converged in %d steps!", step))
        converged = true
        break
    end
end

if not converged then
    print(string.format("Optimization did not converge in %d steps", max_steps))
end

local final_mol = optimizer:get_next_geometry()
local final_energy = optimizer:current_energy()
print(string.format("\nFinal energy: %.8f Ha", final_energy))
print("Optimized geometry:")
print(final_mol)

-- 4. Vibrational analysis at the converged geometry.
if converged then
    print("\nCalculating vibrational frequencies...")
    -- Rebuild HF/wfn at the final geometry (the optimizer's `final_hf`
    -- holds the second-to-last point — easier to redo than to track).
    local basis = occ.AOBasis_load(final_mol:atoms(), "3-21G")
    local hf = occ.HartreeFock(basis)
    local scf = occ.HF(hf)
    scf:set_charge_multiplicity(0, 1)
    scf:run()
    local wfn = scf:wavefunction()

    local hess_eval = hf:hessian_evaluator()
    hess_eval:set_step_size(0.005)
    local hessian = hess_eval:compute(wfn)

    local vib = occ.compute_vibrational_modes(hessian, final_mol)

    print(string.format("Computed %d vibrational modes", vib:n_modes()))
    print("\nFrequencies (cm⁻¹):")
    local freqs = vib:get_all_frequencies()
    for i, f in ipairs(freqs) do
        if f < 0 then
            print(string.format("  Mode %2d: %8.2fi cm⁻¹ (imaginary)", i, math.abs(f)))
        else
            print(string.format("  Mode %2d: %8.2f cm⁻¹", i, f))
        end
    end

    print("\nVibrational analysis summary:")
    print(vib:summary_string())
end
