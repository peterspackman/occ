#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register `occ::dma` (Mult, DMASettings, DMACalculator,
// LinearMultipoleCalculator, DMADriver) onto the `occ` namespace. Mirrors
// src/python/dma_bindings.cpp.
void register_dma_bindings(lua_State *L);

} // namespace occ::lua_bindings
