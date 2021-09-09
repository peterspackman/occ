#include <occ/core/dimer.h>
#include <occ/core/kabsch.h>

#include <fmt/ostream.h>

namespace occ::chem {

Dimer::Dimer(const Molecule &a, const Molecule &b) : m_a(a), m_b(b) {}

Dimer::Dimer(const std::vector<occ::core::Atom> a,
             const std::vector<occ::core::Atom> b)
    : m_a(a), m_b(b) {}

double Dimer::centroid_distance() const {
    return (m_a.centroid() - m_b.centroid()).norm();
}

double Dimer::center_of_mass_distance() const {
    return (m_a.center_of_mass() - m_b.center_of_mass()).norm();
}

double Dimer::nearest_distance() const {
    return std::get<2>(m_a.nearest_atom(m_b));
}

Vec3 Dimer::v_ab() const {
    Vec3 o_a = m_a.centroid();
    Vec3 o_b = m_b.centroid();
    return o_b - o_a;
}

std::optional<occ::Mat4> Dimer::symmetry_relation() const {
    if (!m_a.comparable_to(m_b))
        return std::nullopt;
    using occ::Vec3;
    using occ::linalg::kabsch_rotation_matrix;

    Vec3 o_a = m_a.centroid();
    Vec3 o_b = m_b.centroid();
    Vec3 v_ab = o_b - o_a;
    Mat3N pos_a = m_a.positions();
    pos_a.colwise() -= o_a;
    Mat3N pos_b = m_b.positions();
    pos_b.colwise() -= o_b;

    occ::Mat4 result = occ::Mat4::Identity();
    result.block<3, 3>(0, 0) = kabsch_rotation_matrix(pos_a, pos_b);
    result.block<3, 1>(0, 3) = v_ab;
    return result;
}

const Vec Dimer::vdw_radii() const {
    Vec result(m_a.size() + m_b.size());
    result << m_a.vdw_radii(), m_b.vdw_radii();
    return result;
}

IVec Dimer::atomic_numbers() const {
    IVec result(m_a.size() + m_b.size());
    result << m_a.atomic_numbers(), m_b.atomic_numbers();
    return result;
}

Mat3N Dimer::positions() const {
    Mat3N result(3, m_a.size() + m_b.size());
    result << m_a.positions(), m_b.positions();
    return result;
}

bool Dimer::same_asymmetric_molecule_idxs(const Dimer &rhs) const {
    bool same_idxs = false;
    const int a1_idx = m_a.asymmetric_molecule_idx();
    const int b1_idx = m_b.asymmetric_molecule_idx();
    const int a2_idx = rhs.m_a.asymmetric_molecule_idx();
    const int b2_idx = rhs.m_b.asymmetric_molecule_idx();
    if ((a1_idx < 0) || (b1_idx < 0) || (a2_idx < 0) || (b2_idx < 0))
        same_idxs = true;
    else {
        if ((a1_idx == a2_idx) && (b1_idx == b2_idx))
            same_idxs = true;
        else if ((a1_idx == b2_idx) && (a2_idx == b1_idx))
            same_idxs = true;
    }
    return same_idxs;
}

bool Dimer::operator==(const Dimer &rhs) const {
    if (!same_asymmetric_molecule_idxs(rhs))
        return false;
    constexpr double eps = 1e-7;
    double centroid_diff = abs(centroid_distance() - rhs.centroid_distance());
    if (centroid_diff > eps)
        return false;
    double com_diff =
        abs(center_of_mass_distance() - rhs.center_of_mass_distance());
    if (com_diff > eps)
        return false;
    double nearest_diff = abs(nearest_distance() - rhs.nearest_distance());
    if (nearest_diff > eps)
        return false;
    bool aa_eq = m_a.equivalent_to(rhs.m_a);
    bool bb_eq = m_b.equivalent_to(rhs.m_b);
    if (aa_eq && bb_eq)
        return true;
    bool ba_eq = m_b.equivalent_to(rhs.m_a);
    bool ab_eq = m_a.equivalent_to(rhs.m_b);
    return ab_eq && ba_eq;
}

} // namespace occ::chem
