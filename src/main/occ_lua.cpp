#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <cstdlib>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <iostream>
#include <occ/core/log.h>
#include <occ/main/occ_lua.h>
#include <occ/main/version.h>
#include <sol/sol.hpp>
#include <sstream>
#include <string>

#ifdef OCC_HAVE_LINENOISE
#include <cerrno>
#include <linenoise.h>
#include <unistd.h>
#endif

#include "../lua/occ_module.h"

namespace occ::main {

namespace {

#ifdef OCC_HAVE_LINENOISE

// Drop the last "<eof>"-style continuation marker off a Lua parse error to
// detect "the chunk is incomplete, keep reading lines". Stock Lua's REPL
// (lua.c) does the same check.
bool chunk_is_incomplete(const std::string &err) {
  // Lua appends `'<eof>'` (literal, with the angles and quotes) to the end
  // of the error message when the parser ran out of input mid-statement.
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

// Pull the error message off the top of the Lua stack via lua_tostring,
// which works for both Lua-raised errors and sol2-converted C++ exceptions.
// Going through `sol::error` or `protected_function_result::get<std::string>`
// is unreliable here: when the intermediate `protected_function_result`
// is destroyed (e.g. returned from a helper) it pops its slot off the
// stack, and a delayed `get<>` sees nil. Touching lua_tostring directly
// avoids the lifetime hazard.
std::string lua_error_message(lua_State *L) {
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

// Print returned values from a successful call. For tables we route
// through Lua's `pp` (which detects matrix/vector shapes and prints them
// nicely); for everything else we fall back to `tostring` so userdata
// `__tostring` metamethods fire. Pops the returns off the stack.
void print_results(sol::state &lua, int base) {
  const int n = lua_gettop(lua) - base;
  if (n <= 0) return;
  // pp(x) is installed by occ_module.cpp's open_occ_module.
  lua_getglobal(lua, "pp");
  const bool have_pp = lua_isfunction(lua, -1);
  lua_pop(lua, 1);

  for (int i = 1; i <= n; ++i) {
    const int val_idx = base + i;
    if (have_pp && lua_type(lua, val_idx) == LUA_TTABLE) {
      // pp() prints internally with formatting (and trailing newline).
      lua_getglobal(lua, "pp");
      lua_pushvalue(lua, val_idx);
      if (lua_pcall(lua, 1, 0, 0) != LUA_OK) {
        std::cerr << "(pp error: " << lua_tostring(lua, -1) << ")\n";
        lua_pop(lua, 1);
      }
      continue;
    }
    // Non-table: use tostring so __tostring metamethods fire.
    lua_getglobal(lua, "tostring");
    lua_pushvalue(lua, val_idx);
    if (lua_pcall(lua, 1, 1, 0) != LUA_OK) {
      std::cout << "<un-stringable value>\n";
      lua_pop(lua, 1);
      continue;
    }
    size_t len = 0;
    if (const char *s = lua_tolstring(lua, -1, &len)) {
      std::cout.write(s, len);
    }
    std::cout << "\n";
    lua_pop(lua, 1);
  }
  // Pop the original returns.
  lua_settop(lua, base);
}

// Try to compile `chunk` with `return ... ;` prepended; if that compiles
// it's almost certainly a bare expression and we get free pretty-printing.
// Returns true if successful (chunk left on top of stack).
bool try_load_as_expression(sol::state &lua, const std::string &chunk) {
  std::string wrapped = "return " + chunk + ";";
  const int status = luaL_loadbuffer(lua, wrapped.data(), wrapped.size(),
                                     "=stdin");
  if (status == LUA_OK) return true;
  lua_pop(lua, 1); // pop the load error
  return false;
}

// ---------- tab completion ----------
//
// Linenoise asks for completions via a single callback. We split the
// current input at the rightmost token boundary, walk down the `.` / `:`
// chain in Lua to find the parent table/userdata, then emit every key
// that starts with the partial. Works for globals (`occ<TAB>`), nested
// access (`occ.Xt<TAB>`), and methods on userdata (`calc:s<TAB>`).

struct CompletionContext {
  std::string prefix;        // text before the token we're completing
  std::string parent_path;   // dotted path of the parent table (may be empty)
  bool method_call;          // true if last separator was ':' (method)
  std::string partial;       // token being completed
};

CompletionContext split_completion(const char *buf) {
  CompletionContext ctx;
  const std::string line(buf);
  // Walk backwards to find the start of the identifier being completed.
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
  // Find the rightmost `.` or `:` separator inside the identifier — that's
  // where the parent path ends and the partial token begins.
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

// Push the table/userdata referred to by `dotted` (e.g. "occ.crystal") onto
// the Lua stack. Returns true if found; if false, nothing is pushed.
bool push_parent(sol::state &lua, const std::string &dotted) {
  if (dotted.empty()) {
    lua_pushglobaltable(lua);
    return true;
  }
  lua_pushglobaltable(lua);
  size_t pos = 0;
  while (pos < dotted.size()) {
    size_t dot = dotted.find('.', pos);
    std::string segment = dotted.substr(pos, dot == std::string::npos
                                                  ? std::string::npos
                                                  : dot - pos);
    lua_getfield(lua, -1, segment.c_str());
    lua_remove(lua, -2);
    if (lua_isnil(lua, -1)) {
      lua_pop(lua, 1);
      return false;
    }
    if (dot == std::string::npos) break;
    pos = dot + 1;
  }
  return true;
}

// Iterate the keys of the table/usertype on top of the stack and add
// every string key starting with `partial` to the completion list.
//
// For userdata, sol2 stores bound methods directly on the metatable —
// the `__index` field is a closure (fast lookup), NOT a table you can
// iterate. So we use lua_getmetatable and iterate the metatable itself
// with lua_next (which bypasses sol2's `__pairs` container handler).
void collect_keys(sol::state &lua, const std::string &partial,
                  bool method_call, const std::string &prefix_path,
                  const std::string &line_prefix, linenoiseCompletions *lc) {
  if (!(lua_isuserdata(lua, -1) || lua_istable(lua, -1))) return;
  int iter_idx = lua_gettop(lua);
  bool pushed_metatable = false;
  if (lua_isuserdata(lua, -1)) {
    if (!lua_getmetatable(lua, -1)) return;
    iter_idx = lua_gettop(lua);
    pushed_metatable = true;
  }
  lua_pushnil(lua);
  while (lua_next(lua, iter_idx) != 0) {
    // key at -2, value at -1
    if (lua_type(lua, -2) == LUA_TSTRING) {
      size_t klen = 0;
      const char *kstr = lua_tolstring(lua, -2, &klen);
      std::string key(kstr, klen);
      // Skip private/dunder entries (sol2's __index, __gc, class_cast, etc.).
      if (!key.empty() && key[0] != '_' &&
          key.compare(0, partial.size(), partial) == 0) {
        std::string sep = method_call ? ":" : ".";
        std::string completion = line_prefix + prefix_path + sep + key;
        if (prefix_path.empty()) {
          completion = line_prefix + key;
        }
        linenoiseAddCompletion(lc, completion.c_str());
      }
    }
    lua_pop(lua, 1); // pop value, keep key for next iter
  }
  if (pushed_metatable) lua_pop(lua, 1);
}

void completion_callback(const char *buf, linenoiseCompletions *lc) {
  // The completion callback runs on a sol::state we stash via a thread-local.
  // Set up below in run_repl.
  extern thread_local sol::state *t_active_lua;
  if (!t_active_lua) return;
  sol::state &lua = *t_active_lua;

  CompletionContext ctx = split_completion(buf);

  const int base = lua_gettop(lua);
  if (!push_parent(lua, ctx.parent_path)) return;

  collect_keys(lua, ctx.partial, ctx.method_call, ctx.parent_path, ctx.prefix,
               lc);

  // Restore stack regardless of what collect_keys left.
  lua_settop(lua, base);
}

thread_local sol::state *t_active_lua = nullptr;

int run_repl(sol::state &lua) {
  const std::string history = history_file_path();
  if (!history.empty()) {
    linenoiseHistoryLoad(history.c_str());
    linenoiseHistorySetMaxLen(1000);
  }

  t_active_lua = &lua;
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
      // linenoise returns NULL on both Ctrl-C and Ctrl-D. Distinguish via
      // errno: EAGAIN means SIGINT-style cancellation (mirror Python's
      // REPL — print `^C`, drop the in-progress chunk, re-prompt);
      // anything else is real EOF and we exit.
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

    if (buffer.empty() && line.empty()) {
      continue;
    }
    if (!buffer.empty()) buffer += "\n";
    buffer += line;

    // Compile order matters: stock Lua's REPL tries `return <buf>;`
    // first, falls back to the plain chunk, and only treats *that*
    // failure's `<eof>` suffix as "incomplete, prompt for more". Doing
    // it the other way around treats bare expressions like
    // `mol.positions` as incomplete because they aren't valid Lua
    // statements.
    const int base = lua_gettop(lua);

    bool loaded = try_load_as_expression(lua, buffer);
    if (!loaded) {
      // Not a bare expression — try the literal chunk.
      if (luaL_loadbuffer(lua, buffer.data(), buffer.size(), "=stdin") !=
          LUA_OK) {
        std::string err = lua_error_message(lua);
        if (chunk_is_incomplete(err)) {
          prompt = continuation_prompt;
          continue;
        }
        std::cerr << "parse error: " << err << std::endl;
        buffer.clear();
        prompt = primary_prompt;
        lua_settop(lua, base);
        continue;
      }
    }

    if (!history.empty()) {
      linenoiseHistoryAdd(buffer.c_str());
      linenoiseHistorySave(history.c_str());
    }

    if (lua_pcall(lua, 0, LUA_MULTRET, 0) != LUA_OK) {
      std::string msg = lua_error_message(lua);
      if (msg.empty()) msg = "(no error message)";
      std::cerr << "error: " << msg << std::endl;
    } else {
      print_results(lua, base);
    }

    buffer.clear();
    prompt = primary_prompt;
  }

  t_active_lua = nullptr;
  linenoiseSetCompletionCallback(nullptr);
  return 0;
}

#endif // OCC_HAVE_LINENOISE

} // namespace

CLI::App *add_lua_subcommand(CLI::App &app) {
  auto cfg = std::make_shared<LuaConfig>();
  CLI::App *lua = app.add_subcommand(
      "lua",
      "Run a Lua script with the `occ` module preloaded. Mirrors the API "
      "exposed by the Python (`occpy`) and JS (`occjs`) bindings. If no "
      "script path is given, reads the script body from stdin.");
  lua->fallthrough();

  // `script` is intentionally not required — falling through to stdin
  // lets `occ lua < foo.lua` and `echo '...' | occ lua` both work. We
  // skip CLI::ExistingFile so the empty-path case isn't rejected.
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
  // If we have a tty on stdin and no script was specified, drop into the
  // REPL. Piped stdin (e.g. `occ lua < foo.lua`) still hits the slurp
  // path below because isatty returns 0 for non-terminals.
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

  sol::state lua;
  occ::lua_bindings::open_occ_module(lua);

  // Mirror stock `lua`'s `arg` table: arg[0] = script name, arg[1..N] =
  // positional args. REPL gets `arg[0] = "=repl"` to give scripts a way
  // to tell what mode they're running in.
  sol::table arg_table = lua.create_named_table("arg");
  arg_table[0] = script_name;
  for (size_t i = 0; i < config.script_args.size(); ++i) {
    arg_table[i + 1] = config.script_args[i];
  }

  for (const auto &snippet : config.exec_snippets) {
    auto result = lua.safe_script(snippet, sol::script_pass_on_error);
    if (!result.valid()) {
      sol::error err = result;
      throw std::runtime_error(
          fmt::format("error in -e snippet: {}", err.what()));
    }
  }

#ifdef OCC_HAVE_LINENOISE
  if (interactive) {
    run_repl(lua);
    return;
  }
#endif

  if (from_stdin) {
    // Slurp the whole stream up-front rather than passing std::cin to
    // lua_load — this gives sol2 the chunk name "stdin" for tracebacks
    // and matches what `lua -` does internally.
    std::ostringstream buf;
    buf << std::cin.rdbuf();
    auto result = lua.safe_script(buf.str(), sol::script_pass_on_error,
                                   /*chunkname=*/"=stdin");
    if (!result.valid()) {
      sol::error err = result;
      throw std::runtime_error(
          fmt::format("Lua script (stdin) failed: {}", err.what()));
    }
    return;
  }

  auto result = lua.safe_script_file(config.script, sol::script_pass_on_error);
  if (!result.valid()) {
    sol::error err = result;
    throw std::runtime_error(
        fmt::format("Lua script '{}' failed: {}", config.script, err.what()));
  }
}

} // namespace occ::main
