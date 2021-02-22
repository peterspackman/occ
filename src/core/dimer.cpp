#include <tonto/core/dimer.h>

namespace tonto::chem {

Dimer::Dimer(const Molecule &a, const Molecule &b) : m_a(a), m_b(b) {}

Dimer::Dimer(const std::vector<libint2::Atom> a, const std::vector<libint2::Atom> b) : m_a(a), m_b(b) {}

double Dimer::centroid_distance() const
{
    return 0.0;
}

double Dimer::center_of_mass_distance() const
{
    return 0.0;
}

double Dimer::nearest_distance() const
{
    return 0.0;
}

std::optional<tonto::Mat> Dimer::symmetry_relation() const
{
    return std::nullopt;
}

const Vec Dimer::vdw_radii() const
{
    return Vec();
}

IVec Dimer::atomic_numbers() const
{
    return IVec();
}

Mat3N Dimer::positions() const
{
    return Mat3N();
}

}
