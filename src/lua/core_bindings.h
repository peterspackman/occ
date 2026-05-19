#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register the contents of `occ::core` and a handful of free utility
// functions onto the `occ` namespace. Mirrors `register_core_bindings` in
// src/python/core_bindings.cpp, snake_case throughout.
void register_core_bindings(lua_State *L);

} // namespace occ::lua_bindings
