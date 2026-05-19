#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register `occ::crystal` types (Crystal, UnitCell, SpaceGroup,
// SymmetryOperation, Surface, ...) onto the `occ` namespace. Mirrors
// `register_crystal_bindings` in src/python/crystal_bindings.cpp.
void register_crystal_bindings(lua_State *L);

} // namespace occ::lua_bindings
