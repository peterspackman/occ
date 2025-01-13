#include <occ/io/cifparser.h>
#include <occ/io/dftb_gen.h>
#include <occ/io/load_geometry.h>
#include <occ/io/xyz.h>

namespace occ::io {

occ::crystal::Crystal load_crystal(const std::string &filename) {
  if (CifParser::is_likely_cif_filename(filename)) {
    occ::io::CifParser parser;
    return parser.parse_crystal_from_file(filename).value();
  } else if (DftbGenFormat::is_likely_gen_filename(filename)) {
    DftbGenFormat parser;
    parser.parse(filename);
    return parser.crystal().value();
  } else
    throw std::runtime_error(fmt::format(
        "Unknown filetype when reading crystal from '{}'", filename));
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
