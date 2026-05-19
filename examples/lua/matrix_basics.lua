-- examples/lua/matrix_basics.lua
-- A tour of the Eigen matrix / vector userdata exposed to Lua. occ
-- methods that return matrices (positions, gradients, Mulliken charges,
-- elastic tensors, ...) hand back these userdata directly — no copy at
-- the boundary, in-place mutation works, the same `pos[i][j]` syntax
-- you'd write for a nested table.
--
-- Run:
--     occ lua examples/lua/matrix_basics.lua [molecule.xyz]
-- (defaults to examples/scf/water.xyz)

-- 1. Constructing matrices and vectors
print("---- construction ----")
-- LuaBridge3 doesn't auto-overload constructors; nested-table /
-- explicit-size construction goes through `.from_table` / `.from_size`
-- static factories. Bare `occ.Mat3()` still gives a default-constructed
-- instance.
local m = occ.Mat3.from_table({{1, 2, 3},
                               {4, 5, 6},
                               {7, 8, 9}})    -- fixed 3×3
print(m)                                      -- pretty-prints with rows
print("rows:", m:rows(), "cols:", m:cols(), "size:", m:size())

local dynamic = occ.Matrix.from_size(2, 4)    -- empty 2×4 MatrixXd
dynamic:fill(0)
dynamic[1][3] = 7.5
print("\ndynamic matrix after [1][3] = 7.5:")
print(dynamic)

local v = occ.Vector.from_table({1.5, 2.5, 3.5, 4.5})   -- 1-D
print("\nvector:", tostring(v))
print("v:sum() =", v:sum(), "   v:norm() =", v:norm())

-- 2. Index access is 1-based (Lua convention), maps to Eigen's
-- 0-based storage internally. Row-major access: m[row][col].
print("\n---- indexing ----")
print("m[1][1] =", m[1][1], "  (top-left)")
print("m[3][3] =", m[3][3], "  (bottom-right)")

-- In-place mutation through the row proxy writes straight into the
-- Eigen matrix — no copy, no temporary table.
m[2][2] = -99
print("after m[2][2] = -99:")
print(m)

-- Whole-row assignment with a Lua table (same number of cols):
m[1] = {100, 200, 300}
print("after m[1] = {100, 200, 300}:")
print(m)

-- 3. Use one in a real occ call. Mat3N goes by reference — anything
-- you mutate is reflected in the calculator.
print("\n---- in a real binding ----")
local xyz_path = arg[1] or "examples/scf/water.xyz"
print("loading:", xyz_path)
local mol = occ.load_molecule(xyz_path)
local calc = occ.XtbCalculator(mol)

local pos = calc.positions       -- Mat3N userdata: 3 rows × N atoms
print("positions:", pos)
print("number of atoms:", pos:cols())
print("oxygen x =", pos[1][1])

-- 4. Eigen matrices returned from methods also become userdata. The
-- result fields, the gradient, charges — all native types.
local result = calc:single_point()
print("\natomic charges (Vector):", result.atomic_charges)
print("sum (should be ~0):", result.atomic_charges:sum())

local g = calc:gradient()
print("\ngradient (Mat3N):")
print(g)

-- 5. Convert back to a nested Lua table when you really need one
-- (e.g. for JSON serialization). The reverse is `occ.Mat3N.from_table(t)`.
local t = pos:to_table()
print("\nas Lua table:", type(t), "row 1 =", t[1][1], t[1][2], t[1][3])

local pos2 = occ.Mat3N.from_table(t)
print("round-trip pos2[1][1]:", pos2[1][1])
