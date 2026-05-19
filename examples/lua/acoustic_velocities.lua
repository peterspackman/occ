-- examples/lua/acoustic_velocities.lua
-- Parallel to examples/python/acoustic_velocities.py: build an elastic
-- tensor (6×6 Voigt-form stiffness) from literature constants for NaCl,
-- compute the Voigt/Reuss/Hill averages and acoustic-wave velocities.
--
-- Run:
--     occ lua examples/lua/acoustic_velocities.lua

-- NaCl elastic constants (GPa) — example values; see Phys. Rev. 86 (1952) 651.
local C = {
    {49.5,  12.7,  12.7,  0.0,  0.0,  0.0},
    {12.7,  49.5,  12.7,  0.0,  0.0,  0.0},
    {12.7,  12.7,  49.5,  0.0,  0.0,  0.0},
    { 0.0,   0.0,   0.0, 12.8,  0.0,  0.0},
    { 0.0,   0.0,   0.0,  0.0, 12.8,  0.0},
    { 0.0,   0.0,   0.0,  0.0,  0.0, 12.8},
}

local et = occ.ElasticTensor(C)

print("=== NaCl elastic tensor ===")
-- `voigt_c` is a read-only property returning the 6×6 stiffness matrix.
print("Voigt stiffness (GPa):")
pp(et.voigt_c)

local Hill = occ.AveragingScheme.HILL

print(string.format("\nAverage bulk modulus    (Hill): K = %.2f GPa",
    et:average_bulk_modulus(Hill)))
print(string.format("Average shear modulus   (Hill): G = %.2f GPa",
    et:average_shear_modulus(Hill)))
print(string.format("Average Young's modulus (Hill): E = %.2f GPa",
    et:average_youngs_modulus(Hill)))
print(string.format("Average Poisson's ratio (Hill): ν = %.4f",
    et:average_poisson_ratio(Hill)))

-- Acoustic velocities at NaCl density 2.165 g/cm³.
local density_g_cm3 = 2.165
local K = et:average_bulk_modulus(Hill)
local G = et:average_shear_modulus(Hill)

local v_s = et:transverse_acoustic_velocity(K, G, density_g_cm3)
local v_p = et:longitudinal_acoustic_velocity(K, G, density_g_cm3)

print(string.format("\nDensity = %.3f g/cm³", density_g_cm3))
print(string.format("Transverse acoustic velocity   V_s = %.0f m/s", v_s))
print(string.format("Longitudinal acoustic velocity V_p = %.0f m/s", v_p))
