#pragma once
struct lua_State;

namespace occ::lua_bindings {

// Open the standard Lua libraries on `L`, then install the `occ`
// namespace with every registered binding submodule. After calling,
// Lua scripts can call `occ.XtbCalculator(...)`, `occ.set_log_level(...)`,
// `occ.load_molecule(...)` etc. without `require`.
void open_occ_module(lua_State *L);

} // namespace occ::lua_bindings
