#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register `occ::isosurface` (SurfaceKind, IsosurfaceCalculator,
// VolumeCalculator + convenience wrappers) onto the `occ` namespace.
// Mirrors src/python/isosurface_bindings.cpp.
void register_isosurface_bindings(lua_State *L);

} // namespace occ::lua_bindings
