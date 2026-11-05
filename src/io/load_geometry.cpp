#include <filesystem>
#include <fmt/core.h>
#include <gemmi/cif.hpp>
#include <occ/io/cifparser.h>
#include <occ/io/dftb_gen.h>
#include <occ/io/load_geometry.h>
#include <occ/io/shelxfile.h>
#include <occ/io/xyz.h>

namespace occ::io {

occ::crystal::Crystal load_crystal(const std::string &filename) {
  if (CifParser::is_likely_cif_filename(filename)) {
    occ::io::CifParser parser;
    auto crystal = parser.parse_crystal_from_file(filename);
    if (!crystal)
      throw std::runtime_error(fmt::format("Could not read CIF '{}': {}",
                                           filename,
                                           parser.failure_description()));
    return *crystal;
  } else if (ShelxFile::is_likely_shelx_filename(filename)) {
    ShelxFile parser;
    auto crystal = parser.read_crystal_from_file(filename);
    if (!crystal)
      throw std::runtime_error(fmt::format("Could not read SHELX file '{}': {}",
                                           filename, parser.error_message()));
    return *crystal;
  } else if (DftbGenFormat::is_likely_gen_filename(filename)) {
    DftbGenFormat parser;
    parser.parse(filename);
    return parser.crystal().value();
  } else
    throw std::runtime_error(fmt::format(
        "Unknown filetype when reading crystal from '{}'", filename));
}

std::vector<CrystalBlock> load_crystal_blocks(const std::string &filename) {
  if (!CifParser::is_likely_cif_filename(filename)) {
    CrystalBlock block{std::filesystem::path(filename).stem().string()};
    try {
      block.crystal = load_crystal(filename);
    } catch (const std::exception &e) {
      block.error = e.what();
    }
    return {std::move(block)};
  }

  const auto doc = gemmi::cif::read_file(filename);
  std::vector<CrystalBlock> result;
  result.reserve(doc.blocks.size());
  for (const auto &cif_block : doc.blocks) {
    CrystalBlock block{cif_block.name, static_cast<int>(result.size())};
    CifParser parser;
    block.crystal = parser.parse_crystal_from_block(cif_block);
    if (!block.crystal)
      block.error = parser.failure_description();
    result.push_back(std::move(block));
  }
  if (result.empty())
    throw std::runtime_error(
        fmt::format("No data block found in CIF '{}'", filename));
  return result;
}

occ::core::Molecule load_molecule(const std::string &filename) {
  if (XyzFileReader::is_likely_xyz_filename(filename)) {
    return molecule_from_xyz_file(filename);
  } else if (DftbGenFormat::is_likely_gen_filename(filename)) {
    DftbGenFormat parser;
    parser.parse(filename);
    return parser.molecule().value();
  } else
    throw std::runtime_error(fmt::format(
        "Unknown filetype when reading molecule from '{}'", filename));
}

} // namespace occ::io
