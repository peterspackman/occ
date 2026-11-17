-- Saddle point search and nudged elastic band through the Lua bindings, on
-- a model surface: a double-well bond 0-1 (minima at 1.6 and 2.4 Bohr, a
-- barrier of 0.0256 Hartree at 2.0) and springs on 0-2 and 1-2.
-- Run by ctest: occ lua tests/lua/opt_tests.lua

local function check(condition, message)
  if not condition then error(message, 2) end
end

local BOHR = 1.8897261246257702 -- per Angstrom

local terms = {
  {1, 2, "well", 1.0, 1.6, 2.4},
  {1, 3, "spring", 0.5, 1.81},
  {2, 3, "spring", 0.1, 2.86},
}

local function coordinates(molecule)
  local p = molecule.positions
  local x = {}
  for a = 1, #p[1] do
    x[a] = {p[1][a] * BOHR, p[2][a] * BOHR, p[3][a] * BOHR}
  end
  return x
end

-- energy (Hartree) and gradient (3 x N nested table, Hartree/Bohr)
local function surface(molecule)
  local x = coordinates(molecule)
  local energy, g = 0.0, {{}, {}, {}}
  for k = 1, 3 do for a = 1, #x do g[k][a] = 0.0 end end
  for _, t in ipairs(terms) do
    local i, j, kind, a, b, c = table.unpack(t)
    local d = {x[i][1] - x[j][1], x[i][2] - x[j][2], x[i][3] - x[j][3]}
    local r = math.sqrt(d[1] ^ 2 + d[2] ^ 2 + d[3] ^ 2)
    local de
    if kind == "well" then
      local u, v = r - b, r - c
      energy = energy + a * u * u * v * v
      de = 2 * a * u * v * (u + v)
    else
      energy = energy + 0.5 * a * (r - b) ^ 2
      de = a * (r - b)
    end
    for k = 1, 3 do
      g[k][i] = g[k][i] + de * d[k] / r
      g[k][j] = g[k][j] - de * d[k] / r
    end
  end
  return energy, g
end

local function triatomic(r01, r02)
  local half = math.rad(104.5 / 2)
  local s, c = math.sin(half), math.cos(half)
  return occ.Molecule({8, 1, 1}, {
    {0.0, r01 * s / BOHR, -r02 * s / BOHR},
    {0.0, 0.0, 0.0},
    {0.0, r01 * c / BOHR, r02 * c / BOHR}})
end

local function bond(molecule, i, j)
  local x = coordinates(molecule)
  return math.sqrt((x[i][1] - x[j][1]) ^ 2 + (x[i][2] - x[j][2]) ^ 2 +
                   (x[i][3] - x[j][3]) ^ 2)
end

-- central differences of the gradient
local function hessian(molecule)
  local p = molecule.positions
  local n = #p[1]
  local h = 1e-4 -- Bohr
  local H = {}
  for c = 1, 3 * n do H[c] = {} end
  for c = 1, 3 * n do
    local k, a = (c - 1) % 3 + 1, (c - 1) // 3 + 1
    local columns = {}
    for s, sign in ipairs({1, -1}) do
      local q = {{}, {}, {}}
      for kk = 1, 3 do for aa = 1, n do q[kk][aa] = p[kk][aa] end end
      q[k][a] = q[k][a] + sign * h / BOHR
      local _, g = surface(occ.Molecule({8, 1, 1}, q))
      columns[s] = g
    end
    for r = 1, 3 * n do
      local kk, aa = (r - 1) % 3 + 1, (r - 1) // 3 + 1
      H[r][c] = (columns[1][kk][aa] - columns[2][kk][aa]) / (2 * h)
    end
  end
  return H
end

local function run(optimizer, steps)
  for _ = 1, steps do
    optimizer:update(surface(optimizer:get_next_geometry()))
    if optimizer:step() then return true end
  end
  return false
end

-- saddle point search from a computed Hessian
local start = triatomic(1.85, 1.75)
local ts = occ.opt.GeometryOptimizer.new_with_settings(
  start, occ.opt.ConvergenceCriteria(), occ.opt.OptimizerSettings.saddle_point(0))
ts:set_initial_hessian(hessian(start))
check(run(ts, 50), "saddle point search did not converge")
check(ts.negative_modes == 1, "not a first-order saddle")
check(math.abs(bond(ts:get_next_geometry(), 1, 2) - 2.0) < 2e-3,
      "saddle not at the barrier")

-- climbing-image band between relaxed endpoints
local function relaxed(r01)
  local optimizer = occ.opt.GeometryOptimizer.new_default(triatomic(r01, 1.81))
  check(run(optimizer, 50), "endpoint did not relax")
  return optimizer:get_next_geometry()
end
local reactant, product = relaxed(1.6), relaxed(2.4)
local settings = occ.opt.NebSettings()
settings.images = 5
local band = occ.opt.NudgedElasticBand(reactant, product, settings)
band:set_endpoint_energies((surface(reactant)), (surface(product)))
local converged = false
for _ = 1, 200 do
  local energies, gradients = {}, {}
  for k, image in ipairs(band:images()) do
    energies[k], gradients[k] = surface(image)
  end
  band:update(energies, gradients)
  if band:step() then converged = true; break end
end
check(converged and band.climbing, "band did not converge with a climbing image")
local top = band.highest_image -- 0-based, as in C++
check(math.abs(bond(band:path()[top + 1], 1, 2) - 2.0) < 5e-3,
      "climbing image not at the barrier")
print("opt_tests.lua: all checks passed")
