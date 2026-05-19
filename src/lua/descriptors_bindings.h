#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register `occ::descriptors` (PDD, Steinhardt, promolecule shape) onto
// the `occ` namespace. Mirrors src/python/descriptors_bindings.cpp.
void register_descriptors_bindings(lua_State *L);

} // namespace occ::lua_bindings
