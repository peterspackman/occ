#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register `occ::qm` (Shell, AOBasis, Wavefunction, HartreeFock, IntegralEngine
// and friends) onto the `occ` namespace. Mirrors `register_qm_bindings` in
// src/python/qm_bindings.cpp.
void register_qm_bindings(lua_State *L);

} // namespace occ::lua_bindings
