#pragma once
#include <sol/sol.hpp>

namespace occ::lua_bindings {

// Register `occ::interaction` (CE model, Wolf sum, classical Coulomb,
// wavefunction transformer) onto the `occ` table. Mirrors
// src/python/interaction_bindings.cpp. Some Eigen-heavy free functions
// take large argument packs; we wrap each with a Lua-table layer.
void register_interaction_bindings(sol::state_view lua, sol::table &occ_module);

} // namespace occ::lua_bindings
