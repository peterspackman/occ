-- RINSE through the Lua bindings, against the C++ reference data.
-- Run by ctest: occ lua tests/lua/rinse_tests.lua <tests/data/rinse> <tests/rinse_reference_data.h>
-- Any failed check raises an error, so occ lua exits non-zero.

local data_dir, reference_path = arg[1], arg[2]
assert(data_dir and reference_path, "usage: rinse_tests.lua <data dir> <reference header>")

local function check(condition, message)
  if not condition then error(message, 2) end
end

local reference_text = assert(io.open(reference_path)):read("a")

-- Hashes and descriptor of one fixture from the reference header
local function reference(name)
  local start = assert(reference_text:find('{"' .. name .. '",', 1, true), name)
  local finish = reference_text:find('\n    {"', start + 1, true) or #reference_text
  local block = reference_text:sub(start, finish)
  local hashes = {}
  for hash in block:gmatch('"([a-z][a-z%-]*)"') do hashes[#hashes + 1] = hash end
  local numbers = {}
  for number in block:gmatch("[-]?%d+%.%d+[eE]?[-+]?%d*") do numbers[#numbers + 1] = tonumber(number) end
  local descriptor = {}
  for i = #numbers - 127, #numbers do descriptor[#descriptor + 1] = numbers[i] end
  return hashes[1], hashes[2], descriptor
end

-- Defaults of rinse-descriptor 3.2.0
local params = occ.RinseParams()
check(params.n_max == 16 and params.l_min == 4 and params.l_max == 20, "default n/l")
check(params.size == 128 and params.num_l_levels == 8, "default size")
check(math.abs(params.qmax_factor - 1.2) < 1e-15, "default qmax_factor")
check(math.abs(params.q_cutoff - 1.2 * params.q_max) < 1e-15, "q_cutoff")
local l_values = params:l_values()
check(#l_values == 8 and l_values[1] == 4 and l_values[8] == 18, "l values")

local rinse = occ.Rinse()
for _, name in ipairs({ "nacl.cif", "QEHWEG01_P21.cif" }) do
  local one_word, five_words, expected = reference(name)
  local crystal = occ.crystal_from_cif_file(data_dir .. "/" .. name)
  local values = rinse:compute(crystal)
  check(#values == 128, name .. ": descriptor length")
  local worst = 0
  for i = 1, 128 do worst = math.max(worst, math.abs(values[i] - expected[i])) end
  check(worst < 1e-13, string.format("%s: descriptor differs by %g", name, worst))
  check(rinse:hash(crystal, 1) == one_word, name .. ": one-word hash")
  check(rinse:hash(crystal, 5) == five_words, name .. ": five-word hash")
  local spectrum = rinse:power_spectrum(crystal)
  check(#spectrum == 16 and #spectrum[1] == 8, name .. ": power spectrum shape")
end

-- A fixed U replaces the ADPs
local fixed = occ.RinseParams()
fixed:fixed_uiso(0.0)
local crystal = occ.crystal_from_cif_file(data_dir .. "/QEHWEG01_P21.cif")
local a = rinse:compute(crystal)
local b = occ.Rinse.new_with_params(fixed):compute(crystal)
local differs = false
for i = 1, 128 do differs = differs or a[i] ~= b[i] end
check(differs, "fixed_uiso changes the descriptor")

print("RINSE Lua tests passed")
