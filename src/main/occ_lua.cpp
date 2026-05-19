#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <cstdlib>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fstream>
#include <iostream>
#include <occ/core/log.h>
#include <occ/main/occ_lua.h>
#include <occ/main/version.h>
#include <sstream>
#include <string>

extern "C" {
#include <lauxlib.h>
#include <lua.h>
#include <lualib.h>
}

#ifdef OCC_HAVE_LINENOISE
#include <cerrno>
#include <linenoise.h>
#include <unistd.h>
#endif

#include "../lua/occ_module.h"

namespace occ::main {

namespace {

// Owning wrapper around lua_State so the REPL / script path don't have
// to manage close() manually.
struct LuaState {
  lua_State *L;
  LuaState() : L(luaL_newstate()) {}
  ~LuaState() {
    if (L) lua_close(L);
  }
  LuaState(const LuaState &) = delete;
  LuaState &operator=(const LuaState &) = delete;
  operator lua_State *() const { return L; }
};

std::string pop_lua_error(lua_State *L) {
  std::string msg;
  if (lua_gettop(L) > 0) {
    size_t len = 0;
    if (const char *s = lua_tolstring(L, -1, &len)) {
      msg.assign(s, len);
    }
    lua_pop(L, 1);
  }
  return msg;
}

#ifdef OCC_HAVE_LINENOISE

bool chunk_is_incomplete(const std::string &err) {
  static constexpr std::string_view marker = "<eof>";
  return err.size() >= marker.size() &&
         err.find(marker, err.size() - marker.size() - 4) != std::string::npos;
}

std::string history_file_path() {
  if (const char *home = std::getenv("HOME")) {
    return std::string(home) + "/.occ_lua_history";
  }
  return std::string{};
}

// Pretty-print returned values from a successful call. For tables we
// route through Lua's `pp` (which detects matrix/vector shapes); for
// everything else we fall back to `tostring` so __tostring metamethods
// fire. Pops the returns off the stack.
void print_results(lua_State *L, int base) {
  const int n = lua_gettop(L) - base;
  if (n <= 0) return;
  lua_getglobal(L, "pp");
  const bool have_pp = lua_isfunction(L, -1);
  lua_pop(L, 1);

  for (int i = 1; i <= n; ++i) {
    const int val_idx = base + i;
    if (have_pp && lua_type(L, val_idx) == LUA_TTABLE) {
      lua_getglobal(L, "pp");
      lua_pushvalue(L, val_idx);
      if (lua_pcall(L, 1, 0, 0) != LUA_OK) {
        std::cerr << "(pp error: " << lua_tostring(L, -1) << ")\n";
        lua_pop(L, 1);
      }
      continue;
    }
    lua_getglobal(L, "tostring");
    lua_pushvalue(L, val_idx);
    if (lua_pcall(L, 1, 1, 0) != LUA_OK) {
      std::cout << "<un-stringable value>\n";
      lua_pop(L, 1);
      continue;
    }
    size_t len = 0;
    if (const char *s = lua_tolstring(L, -1, &len)) {
      std::cout.write(s, len);
    }
    std::cout << "\n";
    lua_pop(L, 1);
  }
  lua_settop(L, base);
}

// Try compiling `chunk` with `return ... ;` prepended; if that compiles
// it's almost certainly a bare expression and we get free pretty-printing.
// Returns true on success (chunk left on top of stack).
bool try_load_as_expression(lua_State *L, const std::string &chunk) {
  std::string wrapped = "return " + chunk + ";";
  const int status =
      luaL_loadbuffer(L, wrapped.data(), wrapped.size(), "=stdin");
  if (status == LUA_OK) return true;
  lua_pop(L, 1);
  return false;
}

// ---------- tab completion ----------
//
// LuaBridge3 stores bound methods on `mt.__index` (a table), not directly
// on the metatable the way sol2 did. We follow that layout: when we land
// on a userdata, push its metatable, then push the metatable's __index
// field; if that's a table, iterate that for method names.

struct CompletionContext {
  std::string prefix;
  std::string parent_path;
  bool method_call;
  std::string partial;
};

CompletionContext split_completion(const char *buf) {
  CompletionContext ctx;
  const std::string line(buf);
  size_t start = line.size();
  while (start > 0) {
    char c = line[start - 1];
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' ||
        c == ':') {
      --start;
    } else {
      break;
    }
  }
  ctx.prefix = line.substr(0, start);
  std::string ident = line.substr(start);
  size_t sep = std::string::npos;
  for (size_t i = ident.size(); i-- > 0;) {
    if (ident[i] == '.' || ident[i] == ':') {
      sep = i;
      break;
    }
  }
  if (sep == std::string::npos) {
    ctx.parent_path.clear();
    ctx.method_call = false;
    ctx.partial = ident;
  } else {
    ctx.parent_path = ident.substr(0, sep);
    ctx.method_call = (ident[sep] == ':');
    ctx.partial = ident.substr(sep + 1);
  }
  return ctx;
}

// Push the table/userdata referred to by `dotted` onto the Lua stack.
// Returns true if found.
bool push_parent(lua_State *L, const std::string &dotted) {
  if (dotted.empty()) {
    lua_pushglobaltable(L);
    return true;
  }
  lua_pushglobaltable(L);
  size_t pos = 0;
  while (pos < dotted.size()) {
    size_t dot = dotted.find('.', pos);
    std::string segment = dotted.substr(pos, dot == std::string::npos
                                                  ? std::string::npos
                                                  : dot - pos);
    lua_getfield(L, -1, segment.c_str());
    lua_remove(L, -2);
    if (lua_isnil(L, -1)) {
      lua_pop(L, 1);
      return false;
    }
    if (dot == std::string::npos) break;
    pos = dot + 1;
  }
  return true;
}

void emit_completions_for_keys(lua_State *L, int iter_idx,
                                const std::string &partial, bool method_call,
                                const std::string &prefix_path,
                                const std::string &line_prefix,
                                linenoiseCompletions *lc) {
  lua_pushnil(L);
  while (lua_next(L, iter_idx) != 0) {
    if (lua_type(L, -2) == LUA_TSTRING) {
      size_t klen = 0;
      const char *kstr = lua_tolstring(L, -2, &klen);
      std::string key(kstr, klen);
      if (!key.empty() && key[0] != '_' &&
          key.compare(0, partial.size(), partial) == 0) {
        std::string sep = method_call ? ":" : ".";
        std::string completion = line_prefix + prefix_path + sep + key;
        if (prefix_path.empty()) completion = line_prefix + key;
        linenoiseAddCompletion(lc, completion.c_str());
      }
    }
    lua_pop(L, 1);
  }
}

void collect_keys(lua_State *L, const std::string &partial, bool method_call,
                  const std::string &prefix_path,
                  const std::string &line_prefix, linenoiseCompletions *lc) {
  if (!(lua_isuserdata(L, -1) || lua_istable(L, -1))) return;

  if (lua_isuserdata(L, -1)) {
    if (!lua_getmetatable(L, -1)) return;
    const int mt_idx = lua_gettop(L);
    // LuaBridge3 puts methods on mt.__index when it's a table.
    lua_getfield(L, mt_idx, "__index");
    if (lua_istable(L, -1)) {
      emit_completions_for_keys(L, lua_gettop(L), partial, method_call,
                                prefix_path, line_prefix, lc);
    }
    lua_pop(L, 1); // __index
    // Fall back to iterating the metatable directly too (catches any
    // bindings that put methods straight on mt — and a sanity net for
    // sol2-era helpers still present in script-side metatables).
    emit_completions_for_keys(L, mt_idx, partial, method_call, prefix_path,
                              line_prefix, lc);
    lua_pop(L, 1); // metatable
  } else {
    emit_completions_for_keys(L, lua_gettop(L), partial, method_call,
                              prefix_path, line_prefix, lc);
  }
}

lua_State *t_active_lua = nullptr;

void completion_callback(const char *buf, linenoiseCompletions *lc) {
  if (!t_active_lua) return;
  lua_State *L = t_active_lua;
  CompletionContext ctx = split_completion(buf);
  const int base = lua_gettop(L);
  if (!push_parent(L, ctx.parent_path)) return;
  collect_keys(L, ctx.partial, ctx.method_call, ctx.parent_path, ctx.prefix,
               lc);
  lua_settop(L, base);
}

int run_repl(lua_State *L) {
  const std::string history = history_file_path();
  if (!history.empty()) {
    linenoiseHistoryLoad(history.c_str());
    linenoiseHistorySetMaxLen(1000);
  }

  t_active_lua = L;
  linenoiseSetCompletionCallback(&completion_callback);

  std::cout << "occ-lua REPL — Ctrl-D to exit, Tab to complete, history at "
            << (history.empty() ? "<HOME unset, disabled>" : history) << "\n"
            << "(Note: `local` only scopes to the current line — drop it for "
               "values you want to reuse.)\n";

  std::string buffer;
  const char *primary_prompt = "occ> ";
  const char *continuation_prompt = "  ..> ";
  const char *prompt = primary_prompt;

  while (true) {
    errno = 0;
    char *raw = linenoise(prompt);
    if (raw == nullptr) {
      if (errno == EAGAIN) {
        std::cout << "^C\n";
        buffer.clear();
        prompt = primary_prompt;
        continue;
      }
      std::cout << "\n";
      break;
    }
    std::string line(raw);
    free(raw);

    if (buffer.empty() && line.empty()) continue;
    if (!buffer.empty()) buffer += "\n";
    buffer += line;

    const int base = lua_gettop(L);

    bool loaded = try_load_as_expression(L, buffer);
    if (!loaded) {
      if (luaL_loadbuffer(L, buffer.data(), buffer.size(), "=stdin") !=
          LUA_OK) {
        std::string err = pop_lua_error(L);
        if (chunk_is_incomplete(err)) {
          prompt = continuation_prompt;
          continue;
        }
        std::cerr << "parse error: " << err << std::endl;
        buffer.clear();
        prompt = primary_prompt;
        lua_settop(L, base);
        continue;
      }
    }

    if (!history.empty()) {
      linenoiseHistoryAdd(buffer.c_str());
      linenoiseHistorySave(history.c_str());
    }

    if (lua_pcall(L, 0, LUA_MULTRET, 0) != LUA_OK) {
      std::string msg = pop_lua_error(L);
      if (msg.empty()) msg = "(no error message)";
      std::cerr << "error: " << msg << std::endl;
    } else {
      print_results(L, base);
    }

    buffer.clear();
    prompt = primary_prompt;
  }

  t_active_lua = nullptr;
  linenoiseSetCompletionCallback(nullptr);
  return 0;
}

#endif // OCC_HAVE_LINENOISE

void run_chunk(lua_State *L, const std::string &source,
                const std::string &chunkname) {
  if (luaL_loadbuffer(L, source.data(), source.size(), chunkname.c_str()) !=
      LUA_OK) {
    throw std::runtime_error(
        fmt::format("Lua parse error ({}): {}", chunkname, pop_lua_error(L)));
  }
  if (lua_pcall(L, 0, 0, 0) != LUA_OK) {
    throw std::runtime_error(
        fmt::format("Lua error ({}): {}", chunkname, pop_lua_error(L)));
  }
}

} // namespace

CLI::App *add_lua_subcommand(CLI::App &app) {
  auto cfg = std::make_shared<LuaConfig>();
  CLI::App *lua = app.add_subcommand(
      "lua",
      "Run a Lua script with the `occ` module preloaded. Mirrors the API "
      "exposed by the Python (`occpy`) and JS (`occjs`) bindings. If no "
      "script path is given, reads the script body from stdin.");
  lua->fallthrough();

  lua->add_option("script", cfg->script,
                   "Path to a Lua script (omit to read from stdin)");
  lua->add_option(
      "-e,--execute", cfg->exec_snippets,
      "Execute a Lua statement before running the script. Repeatable.");
  lua->add_option("script_args", cfg->script_args,
                  "Arguments passed to the script (see `arg` global)");

  lua->callback([cfg]() { run_lua_subcommand(*cfg); });
  return lua;
}

void run_lua_subcommand(const LuaConfig &config) {
  const bool from_stdin = config.script.empty();
#ifdef OCC_HAVE_LINENOISE
  const bool interactive = from_stdin && ::isatty(STDIN_FILENO) != 0;
#else
  constexpr bool interactive = false;
#endif

  const std::string script_name =
      interactive ? "=repl" : (from_stdin ? "-" : config.script);

  occ::main::print_header();
  occ::log::info("");
  occ::log::info("{:-<72s}", "Lua ");
  occ::log::info("script    : {}",
                 interactive ? "<interactive>"
                             : (from_stdin ? "<stdin>" : config.script));
  if (!config.script_args.empty()) {
    occ::log::info("arguments : {}", fmt::join(config.script_args, " "));
  }
  occ::log::info("");

  LuaState lua;
  occ::lua_bindings::open_occ_module(lua);

  // Mirror stock `lua`'s `arg` table: arg[0] = script name, arg[1..N] =
  // positional args. Built directly via the Lua C API — small enough
  // that LuaBridge isn't worth pulling in.
  lua_createtable(lua, static_cast<int>(config.script_args.size()), 1);
  lua_pushstring(lua, script_name.c_str());
  lua_rawseti(lua, -2, 0);
  for (size_t i = 0; i < config.script_args.size(); ++i) {
    lua_pushstring(lua, config.script_args[i].c_str());
    lua_rawseti(lua, -2, static_cast<int>(i + 1));
  }
  lua_setglobal(lua, "arg");

  for (const auto &snippet : config.exec_snippets) {
    run_chunk(lua, snippet, "=-e");
  }

#ifdef OCC_HAVE_LINENOISE
  if (interactive) {
    run_repl(lua);
    return;
  }
#endif

  if (from_stdin) {
    std::ostringstream buf;
    buf << std::cin.rdbuf();
    run_chunk(lua, buf.str(), "=stdin");
    return;
  }

  // Slurp the script file then run it; using luaL_dofile would skip the
  // unified error path.
  std::ostringstream buf;
  {
    std::ifstream in(config.script);
    if (!in) {
      throw std::runtime_error(
          fmt::format("could not open Lua script '{}'", config.script));
    }
    buf << in.rdbuf();
  }
  run_chunk(lua, buf.str(), "@" + config.script);
}

} // namespace occ::main
