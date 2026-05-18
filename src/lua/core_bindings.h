#pragma once
#include <sol/sol.hpp>

namespace occ::lua_bindings {

// Register the contents of `occ::core` and a handful of free utility
// functions onto the `occ` table. Mirrors `register_core_bindings` in
// src/python/core_bindings.cpp, snake_case throughout.
void register_core_bindings(sol::state_view lua, sol::table &occ_module);

} // namespace occ::lua_bindings
