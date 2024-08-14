#pragma once
#include <CLI/App.hpp>

namespace occ::main {

struct OccLuaSettings {
    std::string filename{"input.lua"};
    bool interactive{false};
};


CLI::App *add_lua_subcommand(CLI::App &app);
void run_lua_subcommand(const OccLuaSettings &);

}
