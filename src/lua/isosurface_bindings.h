#pragma once
#include <sol/sol.hpp>

namespace occ::lua_bindings {

// Register `occ::isosurface` (SurfaceKind, IsosurfaceCalculator,
// VolumeCalculator + convenience wrappers) onto the `occ` table.
// Mirrors src/python/isosurface_bindings.cpp.
void register_isosurface_bindings(sol::state_view lua, sol::table &occ_module);

} // namespace occ::lua_bindings
