#include "occ_module.h"
#include "cg_bindings.h"
#include "core_bindings.h"
#include "crystal_bindings.h"
#include "descriptors_bindings.h"
#include "dft_bindings.h"
#include "dma_bindings.h"
#include "eigen_matrix.h"
#include "interaction_bindings.h"
#include "isosurface_bindings.h"
#include "mults_bindings.h"
#include "opt_bindings.h"
#include "qm_bindings.h"
#include "xtb_bindings.h"
#include <LuaBridge/LuaBridge.h>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/crystal/crystal.h>
#include <occ/io/load_geometry.h>

namespace occ::lua_bindings {

namespace {

void register_module_level_helpers(lua_State *L) {
  luabridge::getGlobalNamespace(L)
      .beginNamespace("occ")
      .addFunction(
          "set_log_level",
          +[](const luabridge::LuaRef &level) {
            if (level.isString()) {
              occ::log::set_log_level(level.unsafe_cast<std::string>());
            } else if (level.isNumber()) {
              occ::log::set_log_level(level.unsafe_cast<int>());
            } else {
              throw std::runtime_error(
                  "set_log_level: expected string or integer");
            }
          })
      .addFunction("set_log_file", &occ::log::set_log_file)
      .addFunction(
          "log_info", +[](const std::string &msg) { occ::log::info(msg); })
      .addFunction(
          "log_warn", +[](const std::string &msg) { occ::log::warn(msg); })
      .addFunction(
          "log_error", +[](const std::string &msg) { occ::log::error(msg); })
      .addFunction(
          "log_debug", +[](const std::string &msg) { occ::log::debug(msg); })
      .addFunction(
          "set_num_threads", +[](int n) { occ::parallel::set_num_threads(n); })
      .addFunction(
          "set_data_directory",
          +[](const std::string &s) { occ::set_data_directory(s); })
      .addFunction(
          "load_molecule",
          +[](const std::string &path) { return occ::io::load_molecule(path); })
      .addFunction(
          "load_crystal",
          +[](const std::string &path) { return occ::io::load_crystal(path); })
      .endNamespace();
}

// Lua-side help() / pp() helpers. Need a tweak vs the old sol2 version:
// LuaBridge3 stores bound methods on `mt.__index` (a table), NOT on the
// metatable itself the way sol2 did. We iterate __index when present.
constexpr const char *kHelpSnippet = R"LUA(
local function _next_strkey(t, k)
  while true do
    k = next(t, k)
    if k == nil then return nil end
    if type(k) == "string" then return k end
  end
end

local function _sorted_keys(t, filter)
  local out = {}
  local k = nil
  while true do
    k = _next_strkey(t, k)
    if k == nil then break end
    if filter == nil or filter(k) then table.insert(out, k) end
  end
  table.sort(out)
  return out
end

local function _not_private(k)
  return k:sub(1, 1) ~= "_" and k ~= "class_cast"
end

-- LuaBridge3 puts instance methods on mt.__index (a table). For sol2-
-- backed objects (none should exist post-migration, but be defensive) it
-- put them on mt directly. Falling back to mt if __index is not a table
-- means this works either way.
local function _method_source(mt)
  local idx = rawget(mt, "__index")
  if type(idx) == "table" then return idx end
  return mt
end

-- LuaBridge3's namespace tables reject writes, so we install help / pp
-- only as Lua globals (the sol2-era `occ.help` / `occ.pp` shortcut is
-- gone). Callers should use the bare names from any script.

local function _help(obj)
  if obj == nil then
    print("occ Lua API")
    print("  help(occ)        list members of the occ namespace")
    print("  help(obj)        list methods on a userdata or table")
    print("  Tab              complete identifiers / methods")
    print("  Examples:")
    print("    mol = occ.load_molecule('water.xyz')")
    print("    help(mol)")
    print("    calc = occ.XtbCalculator(mol)")
    print("    e = calc:single_point_energy()")
    return
  end
  local t = type(obj)
  if t == "userdata" then
    local mt = debug.getmetatable(obj)
    if mt == nil then
      print("userdata (no metatable):", tostring(obj))
      return
    end
    print(tostring(obj))
    local tname = rawget(mt, "__type") or rawget(mt, "__name") or "?"
    print(string.format("type     : %s", tname))
    local src = _method_source(mt)
    local keys = _sorted_keys(src, _not_private)
    if #keys == 0 then
      print("methods  : (none discovered)")
    else
      print(string.format("methods  : (%d)", #keys))
      for _, k in ipairs(keys) do
        print("  :" .. k)
      end
    end
  elseif t == "table" then
    print(tostring(obj))
    local keys = _sorted_keys(obj, _not_private)
    if #keys == 0 then
      print("(empty table)")
    else
      print(string.format("keys     : (%d)", #keys))
      for _, k in ipairs(keys) do
        local v = rawget(obj, k)
        print(string.format("  .%-30s %s", k, type(v)))
      end
    end
  elseif t == "function" then
    print("function (signature not introspectable from Lua)")
  else
    print(t, tostring(obj))
  end
end
_G.help = _help

local function _is_number_array(t)
  if type(t) ~= "table" then return false end
  local n = #t
  if n == 0 then return false end
  for i = 1, n do
    if type(t[i]) ~= "number" then return false end
  end
  return true
end

local function _is_matrix(t)
  if type(t) ~= "table" then return false end
  local n = #t
  if n == 0 then return false end
  local cols = nil
  for i = 1, n do
    if not _is_number_array(t[i]) then return false end
    if cols == nil then cols = #t[i]
    elseif cols ~= #t[i] then return false end
  end
  return true, n, cols
end

local function _fmt_num(x)
  if x == math.floor(x) and math.abs(x) < 1e15 then
    return string.format("% 12d", x)
  end
  return string.format("% 12.6f", x)
end

local function _vec_to_string(t)
  local parts = {}
  for i = 1, #t do parts[i] = _fmt_num(t[i]) end
  return string.format("vec[%d] [ %s ]", #t, table.concat(parts, " "))
end

local function _mat_to_string(t, rows, cols)
  local lines = {string.format("mat[%dx%d]", rows, cols)}
  for i = 1, rows do
    local parts = {}
    for j = 1, cols do parts[j] = _fmt_num(t[i][j]) end
    table.insert(lines,
      string.format("  [%d] %s", i, table.concat(parts, " ")))
  end
  return table.concat(lines, "\n")
end

local function _generic_repr(v, depth, indent)
  depth = depth or 0
  indent = indent or "  "
  local t = type(v)
  if t == "string" then return string.format("%q", v)
  elseif t == "number" or t == "boolean" or t == "nil" then
    return tostring(v)
  elseif t == "table" then
    if depth > 4 then return "{...}" end
    local lines = {"{"}
    local k = nil
    local any = false
    while true do
      k = next(v, k); if k == nil then break end
      any = true
      local key_repr
      if type(k) == "string" then key_repr = k
      else key_repr = "[" .. tostring(k) .. "]" end
      table.insert(lines, string.rep(indent, depth + 1) ..
        key_repr .. " = " .. _generic_repr(v[k], depth + 1, indent) .. ",")
    end
    if not any then return "{}" end
    table.insert(lines, string.rep(indent, depth) .. "}")
    return table.concat(lines, "\n")
  else
    return tostring(v)
  end
end

local function _pp(v)
  local s
  if type(v) == "table" then
    local is_mat, rows, cols = _is_matrix(v)
    if is_mat then
      s = _mat_to_string(v, rows, cols)
    elseif _is_number_array(v) then
      s = _vec_to_string(v)
    else
      s = _generic_repr(v)
    end
  else
    s = tostring(v)
  end
  print(s)
end

local function _pp_repr(v)
  if type(v) == "table" then
    local is_mat, rows, cols = _is_matrix(v)
    if is_mat then return _mat_to_string(v, rows, cols)
    elseif _is_number_array(v) then return _vec_to_string(v)
    else return _generic_repr(v) end
  end
  return tostring(v)
end

_G.pp = _pp
_G.pp_repr = _pp_repr
)LUA";

} // namespace

void open_occ_module(lua_State *L) {
  luaL_openlibs(L);

  // Create `occ` as a Lua table accessible from script-side. LuaBridge3
  // populates the namespace lazily via beginNamespace, so we just need
  // an empty table to exist before the Lua-side helper code runs.
  // (The first beginNamespace("occ") call from a binding will create
  // the namespace table; explicit creation here makes the help/pp
  // snippet's `occ.help = ...` assignment safe even if no bindings
  // registered yet.)
  luabridge::getGlobalNamespace(L).beginNamespace("occ").endNamespace();

  register_module_level_helpers(L);

  // Eigen userdata types FIRST so subsequent bindings can return them.
  register_eigen_matrix_types(L);

  register_core_bindings(L);
  register_crystal_bindings(L);
  register_qm_bindings(L);
  register_dft_bindings(L);
  register_descriptors_bindings(L);
  register_opt_bindings(L);
  register_interaction_bindings(L);
  register_dma_bindings(L);
  register_isosurface_bindings(L);
  register_mults_bindings(L);
  register_cg_bindings(L);
  register_xtb_bindings(L);

  // Lua-side conveniences (help, pp) — load AFTER bindings so the
  // namespace tables are populated.
  if (luaL_dostring(L, kHelpSnippet) != LUA_OK) {
    const char *err = lua_tostring(L, -1);
    occ::log::warn("failed to install occ.help: {}", err ? err : "(no msg)");
    lua_pop(L, 1);
  }
}

} // namespace occ::lua_bindings
