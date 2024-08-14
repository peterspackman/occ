#include <fmt/core.h>
#include <fmt/color.h>
#include <occ/main/occ_lua.h>
#include <occ/core/molecule.h>
#include <occ/io/xyz.h>

#ifdef OCC_LUA_BINDINGS
#include <sol/sol.hpp>
#endif

using occ::core::Molecule;

namespace occ::main {

#ifdef OCC_LUA_BINDINGS

template<typename Derived>
sol::table eigen_to_table(sol::state& lua, const Eigen::MatrixBase<Derived>& matrix) {
    sol::table table = lua.create_table(matrix.rows(), matrix.cols());
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            table[i + 1][j + 1] = matrix(i, j);
        }
    }
    return table;
}

void bind_molecule(sol::state& lua) {
    lua.new_usertype<Molecule>("Molecule",
        sol::constructors<Molecule(const IVec&, const Mat3N&)>(),
        "size", &Molecule::size,
        "elements", [](const Molecule& mol) {
            return sol::as_table(mol.elements());
        },
        "positions", [&lua](const Molecule& mol) {
            return eigen_to_table(lua, mol.positions());
        },
        "name", sol::property(&Molecule::name, &Molecule::set_name),
        "atomic_numbers", [&lua](const Molecule& mol) {
            return eigen_to_table(lua, mol.atomic_numbers());
        },
        "vdw_radii", [&lua](const Molecule& mol) {
            return eigen_to_table(lua, mol.vdw_radii());
        },
        "molar_mass", &Molecule::molar_mass,
        "atoms", [](const Molecule& mol) {
            return sol::as_table(mol.atoms());
        },
        "center_of_mass", [&lua](const Molecule& mol) -> sol::table {
            Eigen::Vector3d com = mol.center_of_mass();
            sol::table result = lua.create_table(3, 0);
            result[1] = com.x();
            result[2] = com.y();
            result[3] = com.z();
            return result;
        },
        "from_xyz_file", sol::factories([](const std::string& filename) {
            return occ::io::molecule_from_xyz_file(filename);
        }),
        "from_xyz_string", sol::factories([](const std::string& contents) {
            return occ::io::molecule_from_xyz_string(contents);
        }),
        sol::meta_function::to_string, [](const Molecule& mol) {
            auto com = mol.center_of_mass();
            return fmt::format("<Molecule {} @[{:.5f}, {:.5f}, {:.5f}]>",
                               mol.name(), com.x(), com.y(), com.z());
        }
    );
}

void run_interactive() {
  sol::state lua;
  lua.open_libraries(sol::lib::base, sol::lib::package, sol::lib::coroutine,
                     sol::lib::string, sol::lib::os, sol::lib::math,
                     sol::lib::table, sol::lib::debug, sol::lib::bit32,
                     sol::lib::io, sol::lib::ffi, sol::lib::jit);

  fmt::print(fg(fmt::color::cyan),
             "Lua Interactive Prompt (type 'exit' to quit)\n");

  std::string input;
  while (true) {
    fmt::print(fg(fmt::color::green), ">>> ");
    std::getline(std::cin, input);

    if (input == "exit") {
      break;
    }

    try {
      // Try to execute as a statement
      auto result = lua.safe_script(input, sol::script_pass_on_error);
      if (!result.valid()) {
        // If it's not a valid statement, try to evaluate it as an expression
        result = lua.safe_script("return " + input, sol::script_pass_on_error);
        if (result.valid()) {
          // Print the result
          sol::object ret = result;
          if (ret.is<sol::table>()) {
            fmt::print(fg(fmt::color::yellow), "Table:\n");
            lua["print"](ret);
          } else {
            fmt::print(fg(fmt::color::yellow), "Result: {}\n",
                       lua["tostring"](ret).get<std::string>());
          }
        } else {
          // If both attempts fail, print the error
          sol::error err = result;
          fmt::print(fg(fmt::color::red), "Error: {}\n", err.what());
        }
      }
    } catch (const sol::error &e) {
      fmt::print(fg(fmt::color::red), "Error: {}\n", e.what());
    }
  }
}
#endif


CLI::App *add_lua_subcommand(CLI::App &app) {
  CLI::App *lua = app.add_subcommand("lua", "use lua interface");
  auto settings = std::make_shared<OccLuaSettings>();

  lua->add_option("filename", settings->filename, "input lua file");

  lua->add_flag("--interactive", settings->interactive, "interactive session");
  lua->fallthrough();
  lua->callback([settings]() { run_lua_subcommand(*settings); });
  return lua;
}

void run_lua_subcommand(const OccLuaSettings &settings) {
#ifndef OCC_LUA_BINDINGS
  throw std::runtime_exception("occ has not been compiled with lua enabled");
#else
  if (settings.interactive) {
    run_interactive();
  } else {

    sol::state lua;
    lua.open_libraries(sol::lib::base, sol::lib::package, sol::lib::string, 
                       sol::lib::os, sol::lib::math, sol::lib::table);
    
    // Add your custom bindings here
    bind_molecule(lua);
    // Add other bindings as needed...

    try {
      // Read the content of the script file
      std::ifstream script_file(settings.filename);
      if (!script_file.is_open()) {
        throw std::runtime_error("Unable to open script file: " + settings.filename);
      }
      std::string script_content((std::istreambuf_iterator<char>(script_file)),
                                  std::istreambuf_iterator<char>());
      
      // Execute the script
      auto result = lua.safe_script(script_content, &sol::script_pass_on_error);
      if (!result.valid()) {
        sol::error err = result;
        throw std::runtime_error("Lua script execution failed: " + std::string(err.what()));
      }
    } catch (const std::exception& e) {
      fmt::print(fg(fmt::color::red), "Error: {}\n", e.what());
      std::exit(1);
    }
  }
#endif
}

} // namespace occ::main
