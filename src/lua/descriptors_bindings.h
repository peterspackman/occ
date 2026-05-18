#pragma once
#include <sol/sol.hpp>

namespace occ::lua_bindings {

// Register `occ::descriptors` (PDD, Steinhardt, promolecule shape) onto
// the `occ` table. Mirrors src/python/descriptors_bindings.cpp.
void register_descriptors_bindings(sol::state_view lua, sol::table &occ_module);

} // namespace occ::lua_bindings
