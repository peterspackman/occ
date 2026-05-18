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
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/parallel.h>
#include <occ/crystal/crystal.h>
#include <occ/io/load_geometry.h>

namespace occ::lua_bindings {

sol::table open_occ_module(sol::state &lua) {
  lua.open_libraries(sol::lib::base, sol::lib::package, sol::lib::string,
                     sol::lib::math, sol::lib::table, sol::lib::io,
                     sol::lib::os, sol::lib::coroutine, sol::lib::debug);

  sol::table occ_module = lua.create_named_table("occ");

  // Global utilities — mirror occpy.cpp's free-function surface.
  occ_module.set_function(
      "set_log_level",
      [](sol::object level) {
        if (level.is<std::string>()) {
          occ::log::set_log_level(level.as<std::string>());
        } else {
          occ::log::set_log_level(level.as<int>());
        }
      });
  occ_module.set_function("set_log_file", &occ::log::set_log_file);
  occ_module.set_function("set_num_threads",
                          [](int n) { occ::parallel::set_num_threads(n); });
  occ_module.set_function(
      "set_data_directory",
      [](const std::string &s) { occ::set_data_directory(s); });

  occ_module.set_function("load_molecule", [](const std::string &path) {
    return occ::io::load_molecule(path);
  });
  occ_module.set_function("load_crystal", [](const std::string &path) {
    return occ::io::load_crystal(path);
  });

  // The matrix / vector userdata layer registers FIRST so the
  // sol::stack getters specialized for Eigen types are available to
  // every subsequent binding's parameter conversion.
  register_eigen_matrix_types(occ_module);

  register_core_bindings(lua, occ_module);
  register_crystal_bindings(lua, occ_module);
  register_qm_bindings(lua, occ_module);
  register_dft_bindings(lua, occ_module);
  register_descriptors_bindings(lua, occ_module);
  register_opt_bindings(lua, occ_module);
  register_interaction_bindings(lua, occ_module);
  register_dma_bindings(lua, occ_module);
  register_isosurface_bindings(lua, occ_module);
  register_mults_bindings(lua, occ_module);
  register_cg_bindings(lua, occ_module);
  register_xtb_bindings(lua, occ_module);

  // Lua-side help. sol2's `__index` is a C closure (opaque to pairs/next)
  // but the bound method names live as plain string keys on the metatable
  // itself — we just have to bypass `__pairs` (which sol2 hijacks for its
  // container facade) and iterate the metatable directly via `next`.
  // For tables we use plain pairs(); for everything else, print the type
  // and tostring().
  const char *help_lua = R"LUA(
    local function _sorted_keys(t, filter)
      local out = {}
      local k = nil
      while true do
        k = next(t, k)
        if k == nil then break end
        if type(k) == "string" and (filter == nil or filter(k)) then
          table.insert(out, k)
        end
      end
      table.sort(out)
      return out
    end

    local function _not_private(k)
      return k:sub(1, 1) ~= "_" and k ~= "class_cast"
    end

    function occ.help(obj)
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
        local tname = rawget(mt, "__name") or "?"
        print(string.format("type     : %s", tname))
        local keys = _sorted_keys(mt, _not_private)
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

    -- Expose at top level too — `help(x)` is shorter than `occ.help(x)`.
    _G.help = occ.help

    -- pp(value): pretty-print, with smart formatting for nested-table
    -- shapes that come out of our Eigen→table conversion. We detect:
    --   • vector  : flat numeric table {a, b, c, ...}
    --   • matrix  : table of tables, all same length, all numeric
    --   • generic : recursive Lua-table dump
    --
    -- Returns a string; pp(x) also prints it (so users can use it like
    -- Python's `print(repr(x))`).
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

    -- Fixed-width number format used for matrices/vectors.
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

    function occ.pp(v)
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
      -- Intentionally no return — `pp(x)` in the REPL would otherwise
      -- print the string a second time (the REPL pretty-prints returned
      -- values). For scripting use `print(occ.pp_repr(x))` instead.
    end

    -- String-returning form for callers who want the formatted text
    -- without printing.
    function occ.pp_repr(v)
      if type(v) == "table" then
        local is_mat, rows, cols = _is_matrix(v)
        if is_mat then return _mat_to_string(v, rows, cols)
        elseif _is_number_array(v) then return _vec_to_string(v)
        else return _generic_repr(v) end
      end
      return tostring(v)
    end

    _G.pp = occ.pp
  )LUA";
  auto result = lua.safe_script(help_lua, sol::script_pass_on_error,
                                 "=occ.help");
  if (!result.valid()) {
    sol::error err = result;
    occ::log::warn("failed to install occ.help: {}", err.what());
  }

  return occ_module;
}

} // namespace occ::lua_bindings
