#include <occ/io/xyz.h>
#include <occ/io/occ_input.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <scn/scn.h>
#include <occ/core/logger.h>
#include <fstream>

namespace occ::io {
using occ::chem::Element;
using Position = std::array<double, 3>;

XyzFileReader::XyzFileReader(const std::string &filename) {
    std::ifstream file(filename);
    if (not file.good()) {
        throw std::runtime_error(
            fmt::format("Could not open file: '{}'", filename));
    }
    parse(file);
}

XyzFileReader::XyzFileReader(std::istream &stream) { parse(stream); }


void XyzFileReader::parse(std::istream &is) {
    std::string line;
    std::getline(is, line);
    int num_atoms;
    scn::scan(line, "{}", num_atoms);
    std::vector<occ::core::Atom> atoms;
    positions.reserve(num_atoms);
    elements.reserve(num_atoms);
    // comment line
    std::getline(is, comment);
    double x, y, z;
    std::string el;
    while (std::getline(is, line) && num_atoms > 0) {
        auto result = scn::scan(line, "{} {} {} {}", el, x, y, z);
        if (!result) {
            occ::log::error("failed reading {}", result.error().msg());
            continue;
        }
        occ::log::debug("Found atom line: {} {} {} {}\n", el, x, y, z);
        elements.emplace_back(occ::chem::Element(el));
        positions.emplace_back(std::array<double, 3>{x, y, z});
        num_atoms--;
    }
}

void XyzFileReader::update_occ_input(OccInput &result) const {
    result.geometry.positions = positions;
    result.geometry.elements = elements;
}


OccInput XyzFileReader::as_occ_input() const {
    OccInput result;
    update_occ_input(result);
    return result;
}

occ::chem::Molecule molecule_from_xyz_file(const std::string &filename) {
    XyzFileReader xyz(filename);
    return occ::chem::Molecule(xyz.elements, xyz.positions);
}

occ::chem::Molecule molecule_from_xyz_string(const std::string &contents) {
    std::istringstream is(contents);
    if (not is.good()) {
        throw std::runtime_error(
            fmt::format("Could read xyz from string: '{}'", contents));
    }
    XyzFileReader xyz(is);
    return occ::chem::Molecule(xyz.elements, xyz.positions);
}




} // namespace occ::io