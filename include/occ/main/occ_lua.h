#pragma once
#include <CLI/App.hpp>
#include <string>
#include <vector>

namespace occ::main {

struct LuaConfig {
  // Script to run. Empty means "read the script body from stdin" — handy
  // for piping (`echo 'print(occ)' | occ lua`) or heredocs. When a path
  // is given the script sees `arg[0] = script_path`; for stdin we set
  // `arg[0] = "-"`. Either way `arg[1..N] = script_args` matches the
  // standalone `lua` interpreter's shape.
  std::string script;
  // Positional arguments passed through to the script after `--`.
  std::vector<std::string> script_args;
  // -e snippets executed *before* the script. Multiple `-e` flags
  // accumulate, matching the behaviour of stock `lua`.
  std::vector<std::string> exec_snippets;
};

CLI::App *add_lua_subcommand(CLI::App &app);
void run_lua_subcommand(const LuaConfig &config);

} // namespace occ::main
