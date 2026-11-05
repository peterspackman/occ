#pragma once
#include <CLI/App.hpp>
#include <functional>
#include <occ/core/linear_algebra.h>
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
  /// Write the RINSE records to this file as one JSON document
  std::string json_filename{};
  /// Stream the RINSE records to this file as JSON Lines: a header line, then
  /// one line per structure as it is described
  std::string jsonl_filename{};

  /// Whether RINSE records are written to a file, which restricts the run to
  /// RINSE and makes a structure that fails a record rather than an error
  bool writes_records() const {
    return !json_filename.empty() || !jsonl_filename.empty();
  }

  std::vector<Descriptor> descriptors() const;
};

/// What became of one input structure: a file, or one data block of a CIF.
struct DescribeOutcome {
  enum class Status { Described, Skipped, Failed };

  std::string filename;
  /// The CIF data block, or the file's stem for other formats
  std::string block;
  /// Position of the block in its file, from 0
  int index{0};
  Status status{Status::Described};
  /// The RINSE hash of a crystal; for a skipped or failed structure, the reason
  std::string detail;
  /// The flattened RINSE descriptor of a described crystal, otherwise empty
  occ::Vec values;
};

CLI::App *add_describe_subcommand(CLI::App &app);

/// Describe every structure in the configuration in turn, logging as it goes:
/// each data block of a multi-block CIF is a structure of its own. A structure
/// that cannot be read is recorded as failed rather than stopping the rest.
/// `on_outcome`, if given, sees each outcome as soon as it is known.
std::vector<DescribeOutcome> describe_structures(
    DescribeConfig const &,
    const std::function<void(const DescribeOutcome &)> &on_outcome = {});

/// describe_structures(), then a summary when there was more than one input,
/// and the RINSE records if a JSON or JSON Lines file was asked for.
/// \throws std::runtime_error if none was described, or, unless the records
/// went to a file (where each failure is a record of its own), if any failed.
void run_describe_subcommand(DescribeConfig const &);
} // namespace occ::main
