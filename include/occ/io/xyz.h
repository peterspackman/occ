#pragma once
#include <array>
#include <istream>
#include <occ/core/molecule.h>
#include <occ/core/element.h>

namespace occ::io {

struct OccInput;

struct XyzFileReader {
    using Position = std::array<double, 3>;

    XyzFileReader(std::istream &);
    XyzFileReader(const std::string &);

    std::vector<occ::chem::Element> elements;
    std::vector<Position> positions;
    std::string comment;

    OccInput as_occ_input() const;
    void update_occ_input(OccInput &) const;
private:
    void parse(std::istream &);
};


occ::chem::Molecule molecule_from_xyz_file(const std::string &);
occ::chem::Molecule molecule_from_xyz_string(const std::string &);


} // namespace occ::io
