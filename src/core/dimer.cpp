#include <tonto/core/dimer.h>
#include <tonto/core/kabsch.h>

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

std::optional<tonto::Mat4> Dimer::symmetry_relation() const
{
    if(!m_a.comparable_to(m_b)) return std::nullopt;
    using tonto::Vec3;
    using tonto::linalg::kabsch_rotation_matrix;

    Vec3 o_a = m_a.centroid();
    Vec3 o_b = m_b.centroid();
    Vec3 v_ab = o_b - o_a;
    Mat3N pos_a = m_a.positions() - o_a;
    Mat3N pos_b = m_b.positions() - o_b;

    tonto::Mat4 result = tonto::Mat4::Zero();
    result.block<3, 3>(0, 0) = kabsch_rotation_matrix(pos_a, pos_b);
    result.block<3, 1>(0, 3) = v_ab;
    return result;
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
