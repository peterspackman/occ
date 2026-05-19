#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register `occ::interaction` (CE model, Wolf sum, classical Coulomb,
// wavefunction transformer) onto the `occ` namespace. Mirrors
// src/python/interaction_bindings.cpp. Some Eigen-heavy free functions
// take large argument packs; we wrap each with a Lua-table layer.
void register_interaction_bindings(lua_State *L);

} // namespace occ::lua_bindings
