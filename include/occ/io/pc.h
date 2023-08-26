#pragma once
#include <vector>
#include <istream>
#include <occ/core/point_charge.h>

namespace occ::io {

struct OccInput;

struct PointChargeFileReader {
    using PointChargeList = std::vector<occ::core::PointCharge>;

    PointChargeFileReader(std::istream &);
    PointChargeFileReader(const std::string &);

    PointChargeList point_charges;

    void update_occ_input(OccInput &) const;

  private:
    void parse(std::istream &);
};

PointChargeList point_charges_from_file(const std::string &);
PointChargeList point_charges_from_string(const std::string &);

} // namespace occ::io
