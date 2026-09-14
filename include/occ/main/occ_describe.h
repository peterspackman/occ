#pragma once
#include <CLI/App.hpp>
#include <optional>
#include <vector>

namespace occ::main {

struct DescribeConfig {
  enum class Descriptor {
    Steinhardt,
    Rinse,
  };

  std::string geometry_filename{""};
  std::vector<std::string> descriptor_strings{};
  /// Words in the RINSE hash; each carries 16 bits
  int hash_words{1};
  /// Isotropic U, in A^2, given to every atom in place of its ADPs for RINSE
  std::optional<double> fixed_uiso{};

  std::vector<Descriptor> descriptors() const;
};

CLI::App *add_describe_subcommand(CLI::App &app);
void run_describe_subcommand(DescribeConfig const &);
} // namespace occ::main
