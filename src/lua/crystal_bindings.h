#pragma once
#include <sol/sol.hpp>

namespace occ::lua_bindings {

// Register `occ::crystal` types (Crystal, UnitCell, SpaceGroup,
// SymmetryOperation, Surface, ...) onto the `occ` table. Mirrors
// `register_crystal_bindings` in src/python/crystal_bindings.cpp.
void register_crystal_bindings(sol::state_view lua, sol::table &occ_module);

} // namespace occ::lua_bindings
