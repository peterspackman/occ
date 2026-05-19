#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register `occ::opt` (BondCoordinate, AngleCoordinate, InternalCoordinates,
// BernyOptimizer, ...) into `occ.opt`. Mirrors src/python/opt_bindings.cpp,
// which uses a submodule (the Python binding's only nested namespace).
void register_opt_bindings(lua_State *L);

} // namespace occ::lua_bindings
