#pragma once
#include <sol/sol.hpp>

namespace occ::lua_bindings {

// Register XtbCalculator + XtbResult into the `occ` table at the top of
// `lua`. Mirrors `register_xtb_bindings` in src/python and src/js, using
// snake_case to match the Python surface.
void register_xtb_bindings(sol::state_view lua, sol::table &occ_module);

} // namespace occ::lua_bindings
