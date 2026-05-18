#pragma once
#include <sol/sol.hpp>

namespace occ::lua_bindings {

// Register `occ::dft::DFT`, `occ::qm::SCF<DFT>`, GridSettings, XDM, and the
// DFT-side HessianEvaluator. Mirrors src/python/dft_bindings.cpp.
void register_dft_bindings(sol::state_view lua, sol::table &occ_module);

} // namespace occ::lua_bindings
