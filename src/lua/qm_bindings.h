#pragma once
#include <sol/sol.hpp>

namespace occ::lua_bindings {

// Register `occ::qm` (Shell, AOBasis, Wavefunction, HartreeFock, IntegralEngine
// and friends) onto the `occ` table. Mirrors `register_qm_bindings` in
// src/python/qm_bindings.cpp.
void register_qm_bindings(sol::state_view lua, sol::table &occ_module);

} // namespace occ::lua_bindings
