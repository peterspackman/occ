#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Register XtbCalculator + XtbResult into the `occ` namespace. Mirrors
// `register_xtb_bindings` in src/python and src/js, using snake_case to
// match the Python surface.
void register_xtb_bindings(lua_State *L);

} // namespace occ::lua_bindings
