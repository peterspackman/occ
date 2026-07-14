#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register post-HF correlation methods (run_correlation, MP2, CCSD/(T),
// UCCSD) onto the `occ` namespace. Mirrors `register_correlation_bindings`
// in src/python/correlation_bindings.cpp.
void register_correlation_bindings(lua_State *L);

} // namespace occ::lua_bindings
