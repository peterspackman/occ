#pragma once
#include "linear_algebra.h"
#include "numgrid.h"
#include <vector>
#include <xc.h>
#include <string>

namespace libint2 {
class BasisSet;
class Atom;
}

namespace tonto::dft {

using tonto::Mat3N;
using tonto::MatRM;
using tonto::MatN4;
using tonto::Vec;
using tonto::IVec;

class DFTGrid {
public:
    DFTGrid(const libint2::BasisSet&, const std::vector<libint2::Atom>&);
    const auto& atomic_numbers() const { return m_atomic_numbers; }
    const auto n_atoms() const { return m_atomic_numbers.size(); }
    MatN4 grid_points(size_t idx) const;
    void set_radial_precision(double prec) { m_radial_precision = prec; }
    void set_min_angular_points(size_t n) { m_min_angular = n; }
    void set_max_angular_points(size_t n) { m_max_angular = n; }

private:
    double m_radial_precision{1e-12};
    size_t m_min_angular{86};
    size_t m_max_angular{302};
    IVec m_l_max;
    Vec m_x;
    Vec m_y;
    Vec m_z;
    IVec m_atomic_numbers;
    Vec m_alpha_max;
    MatRM m_alpha_min;
};


class DensityFunctional {
public:
    DensityFunctional(const std::string&);
private:
    xc_func_type m_exchange;
    xc_func_type m_correlation;
};
}
