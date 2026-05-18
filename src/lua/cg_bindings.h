#pragma once
#include <sol/sol.hpp>

namespace occ::lua_bindings {

// Register `occ::cg` (CrystalGrowthResult, DimerResult, MoleculeResult,
// InteractionMapper, ...) onto the `occ` table. Mirrors
// src/python/cg_bindings.cpp.
void register_cg_bindings(sol::state_view lua, sol::table &occ_module);

} // namespace occ::lua_bindings
