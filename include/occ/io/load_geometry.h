#pragma once
#include <occ/core/molecule.h>
#include <occ/crystal/crystal.h>
#include <optional>
#include <string>
#include <vector>

namespace occ::io {

occ::crystal::Crystal load_crystal(const std::string &filename);

/// One structure from a file that may hold several, such as a multi-block CIF.
struct CrystalBlock {
  /// The CIF data block name, or the file's stem for single-structure formats
  std::string name;
  /// Position of the block in its file, from 0
  int index{0};
  /// Unset when the block could not be read as a crystal; see `error`
  std::optional<occ::crystal::Crystal> crystal;
  std::string error;
};

/// Every crystal in the file: one per data block of a CIF, otherwise the
/// single structure load_crystal() reads. A block that is not a crystal (no
/// cell or symmetry, say) is returned with its error rather than skipped, so
/// the result lines up with the file's blocks; so is a single-structure file
/// that cannot be read.
/// \throws std::runtime_error if a CIF cannot be read or holds no data block.
std::vector<CrystalBlock> load_crystal_blocks(const std::string &filename);
occ::core::Molecule load_molecule(const std::string &filename);

} // namespace occ::io
