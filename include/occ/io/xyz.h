#pragma once
#include <array>
#include <istream>
#include <occ/core/element.h>
#include <occ/core/molecule.h>

namespace occ::io {

struct OccInput;

struct XyzFileReader {
  using Position = std::array<double, 3>;

  XyzFileReader(std::istream &);
  XyzFileReader(const std::string &);

  std::vector<occ::core::Element> elements;
  std::vector<Position> positions;
  std::string comment;

  OccInput as_occ_input() const;
  void update_occ_input(OccInput &) const;

  static bool is_likely_xyz_filename(const std::string &filename);

private:
  void parse(std::istream &);
};

occ::core::Molecule molecule_from_xyz_file(const std::string &);
occ::core::Molecule molecule_from_xyz_string(const std::string &);

} // namespace occ::io
