-- examples/lua/core_demo.lua
-- Exercise the stage-2 core bindings: Element / Atom / Molecule /
-- PointCharge / MolecularPointGroup, plus a few free helpers.
--
-- Run:
--     occ lua examples/lua/core_demo.lua

-- Elements --------------------------------------------------------------
local c = occ.Element("C")
local h = occ.Element(1)
print(string.format("Element  : %s (Z=%d, mass=%.3f, vdW=%.3f Bohr)",
    c.symbol, c.atomic_number, c.mass, c.van_der_waals_radius))
print(string.format("           %s (Z=%d, name=%s)", h.symbol, h.atomic_number, h.name))

-- Atoms (Bohr) ----------------------------------------------------------
local o  = occ.Atom(8, -1.32695761, -0.10593856, 0.01878821)
local h1 = occ.Atom(1, -1.93166418,  1.60017351, -0.02171049)
local h2 = occ.Atom(1,  0.48664409,  0.07959806, 0.00986248)
print("Atom     :", tostring(o))

-- Build a Molecule from atomic numbers + 3×N positions (Eigen
-- column-major; rows = x, y, z; columns = atoms).
local water = occ.Molecule(
    {8, 1, 1},
    {{-1.32695761, -1.93166418,  0.48664409},   -- x: O, H, H
     {-0.10593856,  1.60017351,  0.07959806},   -- y
     { 0.01878821, -0.02171049,  0.00986248}})  -- z
water.name = "water"
print(string.format("Molecule : %s, %d atoms, molar mass %.3f g/mol",
    water.name, #water, water.molar_mass))

local com = water.center_of_mass
print(string.format("COM      : [% .4f, % .4f, % .4f]", com[1], com[2], com[3]))

-- Atomic numbers as a Vec userdata; ipairs walks via __index for 1..n.
io.write("Z list   :")
for _, z in ipairs(water.atomic_numbers) do io.write(string.format(" %d", z)) end
io.write("\n")

-- Translate a copy
local shifted = water:translated({1.0, 0.0, 0.0})
local shifted_com = shifted.center_of_mass
print(string.format("Shifted  : [% .4f, % .4f, % .4f]",
    shifted_com[1], shifted_com[2], shifted_com[3]))

-- Symmetry --------------------------------------------------------------
local pg = occ.MolecularPointGroup(water)
print(string.format("Point group : %s  (description: %s, symnum=%d)",
    pg.point_group_string, pg.description, pg.symmetry_number))

-- Free helpers ----------------------------------------------------------
local qeem = occ.eem_partial_charges({8, 1, 1},
    {{-1.32695761, -1.93166418,  0.48664409},   -- x: O, H, H
     {-0.10593856,  1.60017351,  0.07959806},   -- y
     { 0.01878821, -0.02171049,  0.00986248}})  -- z
io.write("EEM q    :")
for _, q in ipairs(qeem) do io.write(string.format("  %+.4f", q)) end
io.write("\n")

-- Logging ---------------------------------------------------------------
occ.log_info("hello from lua")

-- PointCharge -----------------------------------------------------------
local pc = occ.PointCharge(-0.5, 0.0, 0.0, 0.0)
print("PointCharge:", tostring(pc))
