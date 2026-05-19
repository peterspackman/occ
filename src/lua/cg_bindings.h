#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register `occ::cg` (CrystalGrowthResult, DimerResult, MoleculeResult,
// InteractionMapper, ...) onto the `occ` namespace. Mirrors
// src/python/cg_bindings.cpp.
void register_cg_bindings(lua_State *L);

} // namespace occ::lua_bindings
