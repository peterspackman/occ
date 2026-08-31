#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register `occ::solvent::cosmors` (CosmoRSSettings, CosmoRSEnergy,
// cosmo_rs_solvation_free_energy, ...) onto the `occ` namespace. Mirrors
// src/python/solvent_bindings.cpp.
void register_solvent_bindings(lua_State *L);

} // namespace occ::lua_bindings
