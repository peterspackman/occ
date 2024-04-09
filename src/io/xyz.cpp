#include <fstream>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/units.h>
#include <occ/io/occ_input.h>
#include <occ/io/xyz.h>
#include <filesystem>
#include <scn/scan.h>

namespace fs = std::filesystem;

namespace occ::io {
using occ::core::Element;
using Position = std::array<double, 3>;

XyzFileReader::XyzFileReader(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.good()) {
        throw std::runtime_error(
            fmt::format("Could not open file: '{}'", filename));
    }
    parse(file);
}

XyzFileReader::XyzFileReader(std::istream &stream) { parse(stream); }

void XyzFileReader::parse(std::istream &is) {
    std::string line;
    std::getline(is, line);
    auto result = scn::scan<int>(line, "{}");
    if(!result) {
	occ::log::error("failed reading atom count line");
	return;
    }
    int num_atoms = result->value();
    std::vector<occ::core::Atom> atoms;
    positions.reserve(num_atoms);
    elements.reserve(num_atoms);
    // comment line
    std::getline(is, comment);
    while (std::getline(is, line) && num_atoms > 0) {
        auto result = scn::scan<std::string, double, double, double>(line, "{} {} {} {}");
        if (!result) {
            occ::log::error("failed reading {}", result.error().msg());
            continue;
        }
	auto &[el, x, y, z] = result->values();
        occ::log::debug("Found atom line: {} {} {} {}", el, x, y, z);
        elements.emplace_back(occ::core::Element(el));
        positions.emplace_back(std::array<double, 3>{x, y, z});
        num_atoms--;
    }
}

void XyzFileReader::update_occ_input(OccInput &result) const {
    result.geometry.positions = positions;
    result.geometry.elements = elements;
}

bool XyzFileReader::is_likely_xyz_filename(const std::string &filename) {
    fs::path path(filename);
    std::string ext = path.extension().string();
    if(ext == ".xyz") return true;
    return false;
}

OccInput XyzFileReader::as_occ_input() const {
    OccInput result;
    update_occ_input(result);
    return result;
}

occ::core::Molecule molecule_from_xyz_file(const std::string &filename) {
    XyzFileReader xyz(filename);
    return occ::core::Molecule(xyz.elements, xyz.positions);
}

occ::core::Molecule molecule_from_xyz_string(const std::string &contents) {
    std::istringstream is(contents);
    if (not is.good()) {
        throw std::runtime_error(
            fmt::format("Could read xyz from string: '{}'", contents));
    }
    XyzFileReader xyz(is);
    return occ::core::Molecule(xyz.elements, xyz.positions);
}

} // namespace occ::io
