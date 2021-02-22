#include <tonto/core/dimer.h>
#include <tonto/core/kabsch.h>

#include <fmt/ostream.h>

namespace tonto::chem {

Dimer::Dimer(const Molecule &a, const Molecule &b) : m_a(a), m_b(b) {}

Dimer::Dimer(const std::vector<libint2::Atom> a, const std::vector<libint2::Atom> b) : m_a(a), m_b(b) {}

double Dimer::centroid_distance() const
{
    return (m_a.centroid() - m_b.centroid()).norm();
}

double Dimer::center_of_mass_distance() const
{
    return (m_a.center_of_mass() - m_b.center_of_mass()).norm();
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
    Mat3N pos_a = m_a.positions();
    pos_a.colwise() -= o_a;
    Mat3N pos_b = m_b.positions();
    pos_b.colwise() -= o_b;
    fmt::print("pos_a\n{}\n", pos_a);
    fmt::print("pos_b\n{}\n", pos_b);

    tonto::Mat4 result = tonto::Mat4::Identity();
    result.block<3, 3>(0, 0) = kabsch_rotation_matrix(pos_a, pos_b);
    result.block<3, 1>(0, 3) = v_ab;
    return result;
}

const Vec Dimer::vdw_radii() const
{
    Vec result(m_a.size() + m_b.size());
    result << m_a.vdw_radii(), m_b.vdw_radii();
    return result;
}

IVec Dimer::atomic_numbers() const
{
    IVec result(m_a.size() + m_b.size());
    result << m_a.atomic_numbers(), m_b.atomic_numbers();
    return result;
}

Mat3N Dimer::positions() const
{
    Mat3N result(3, m_a.size() + m_b.size());
    result << m_a.positions(), m_b.positions();
    return result;
}

}
