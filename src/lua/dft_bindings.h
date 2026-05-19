#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register `occ::dft::DFT`, `occ::qm::SCF<DFT>`, GridSettings, XDM, and the
// DFT-side HessianEvaluator. Mirrors src/python/dft_bindings.cpp.
void register_dft_bindings(lua_State *L);

} // namespace occ::lua_bindings
