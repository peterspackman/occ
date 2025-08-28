#pragma once

#include <vector>
#include <complex>

#include <trajan/core/linear_algebra.h>
#include <trajan/core/unit_cell.h>

namespace trajan::core {

class Ewald {
public:
    Ewald(double alpha);

    void update(const Mat3N& positions, const std::vector<double>& charges, const UnitCell& unit_cell);
    double reciprocal_sum(bool use_fft = false) const;

private:
    std::vector<std::complex<double>> calculate_structure_factor_fft() const;
    std::vector<std::complex<double>> calculate_structure_factor_direct() const;

    double m_alpha;
    const Mat3N* m_positions = nullptr;
    const std::vector<double>* m_charges = nullptr;
    const UnitCell* m_unit_cell = nullptr;
};

} // namespace trajan::core