#pragma once
#include <occ/crystal/miller.h>
#include <occ/core/linear_algebra.h>
#include <array>

namespace occ::crystal {

class Crystal;

class Surface {
public:
    Surface(const MillerIndex&, const Crystal&);

    double depth() const;
    double d() const;
    void print() const;
    std::array<double, 3> dipole() const;

private:
    MillerIndex m_hkl;
    double m_depth{0.0};
    Mat3 m_lattice;
    Mat3 m_reciprocal_lattice;
};

std::vector<Surface> generate_surfaces(const Crystal &c, double d_min);

}
