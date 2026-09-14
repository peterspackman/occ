#pragma once
#include <CLI/App.hpp>
#include <optional>
#include <string>
#include <vector>

namespace occ::main {

struct DescribeConfig {
  enum class Descriptor {
    Steinhardt,
    Rinse,
  };

  /// Structure files to describe, each handled according to its extension: xyz
  /// for molecules, cif or res/ins for crystals
  std::vector<std::string> structure_filenames{};
  std::vector<std::string> descriptor_strings{};
  /// Words in the RINSE hash; each carries 16 bits
  int hash_words{1};
  /// Isotropic U, in A^2, given to every atom in place of its ADPs for RINSE
  std::optional<double> fixed_uiso{};

  std::vector<Descriptor> descriptors() const;
};

/// What became of one input structure.
struct DescribeOutcome {
  enum class Status { Described, Skipped, Failed };

  std::string filename;
  Status status{Status::Described};
  /// The RINSE hash of a crystal; for a skipped or failed structure, the reason
  std::string detail;
};

CLI::App *add_describe_subcommand(CLI::App &app);

/// Describe every structure in the configuration in turn, logging as it goes.
/// A structure that cannot be read is recorded as failed rather than stopping
/// the rest.
std::vector<DescribeOutcome> describe_structures(DescribeConfig const &);

/// describe_structures(), then a summary when there was more than one input.
/// \throws std::runtime_error if any structure failed, or none was described.
void run_describe_subcommand(DescribeConfig const &);
} // namespace occ::main
