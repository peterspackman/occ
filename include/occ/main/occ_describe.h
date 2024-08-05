#pragma once
#include <CLI/App.hpp>
#include <vector>

namespace occ::main {

struct DescribeConfig {
  enum class Descriptor {
    Steinhardt,
  };

  std::string geometry_filename{""};
  std::vector<std::string> descriptor_strings{};

  std::vector<Descriptor> descriptors() const;
};

CLI::App *add_describe_subcommand(CLI::App &app);
void run_describe_subcommand(DescribeConfig const &);
} // namespace occ::main
