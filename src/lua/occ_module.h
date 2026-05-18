#pragma once
#include <sol/sol.hpp>

namespace occ::lua_bindings {

// Open the standard Lua libraries on `lua`, then install the `occ` table
// with every registered binding submodule. After calling this, scripts can
// `require`-free reference `occ.XtbCalculator`, `occ.set_log_level`, etc.
//
// Returns the `occ` table for callers that want to install additional
// per-job state (input filename, working directory, ...).
sol::table open_occ_module(sol::state &lua);

} // namespace occ::lua_bindings
