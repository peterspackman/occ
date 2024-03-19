#include <fstream>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/io/occ_input.h>
#include <occ/io/pc.h>
#include <scn/scan.h>

namespace occ::io {

PointChargeFileReader::PointChargeFileReader(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.good()) {
        throw std::runtime_error(
            fmt::format("Could not open file: '{}'", filename));
    }
    parse(file);
}

PointChargeFileReader::PointChargeFileReader(std::istream &stream) { parse(stream); }

void PointChargeFileReader::parse(std::istream &is) {
    std::string line;
    std::getline(is, line);
    auto scan_result = scn::scan<int>(line, "{}");
    if(!scan_result) {
	occ::log::error("failed reading charge count line");
	return;
    }
    int num_charges = scan_result->value();
    point_charges.reserve(num_charges);
    // no comment line
    while (std::getline(is, line) && num_charges > 0) {
        auto result = scn::scan<double, double, double, double>(line, "{} {} {} {}");
        if (!result) {
            occ::log::error("failed reading {}", result.error().msg());
            continue;
        }
	auto [q, x, y, z] = result->values();
        x *= occ::units::ANGSTROM_TO_BOHR;
        y *= occ::units::ANGSTROM_TO_BOHR;
        z *= occ::units::ANGSTROM_TO_BOHR;
        occ::log::debug("Found point charge line: {} {} {} {}", q, x, y, z);
        point_charges.emplace_back(occ::core::PointCharge{q, {x, y, z}});
        num_charges--;
    }
}

void PointChargeFileReader::update_occ_input(OccInput &result) const {
    result.geometry.point_charges = point_charges;
}

PointChargeList point_charges_from_file(const std::string &filename) {
    PointChargeFileReader pc(filename);
    return pc.point_charges;
}

PointChargeList point_charges_from_string(const std::string &contents) {
    std::istringstream is(contents);
    if (not is.good()) {
        throw std::runtime_error(
            fmt::format("Could read point charges from string: '{}'", contents));
    }
    PointChargeFileReader pc(is);
    return pc.point_charges;
}

} // namespace occ::io
