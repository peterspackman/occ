#pragma once
#include <sol/sol.hpp>

namespace occ::lua_bindings {

// Register `occ::dma` (Mult, DMASettings, DMACalculator, LinearMultipoleCalculator,
// DMADriver) onto the `occ` table. Mirrors src/python/dma_bindings.cpp.
void register_dma_bindings(sol::state_view lua, sol::table &occ_module);

} // namespace occ::lua_bindings
