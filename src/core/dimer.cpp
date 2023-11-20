#include <occ/core/dimer.h>
#include <occ/core/kabsch.h>
#include <occ/core/util.h>

#include <fmt/ostream.h>

namespace occ::core {
using occ::core::linalg::kabsch_rotation_matrix;

Dimer::Dimer(const Molecule &a, const Molecule &b) : m_a(a), m_b(b) {}

Dimer::Dimer(const std::vector<occ::core::Atom> &a,
             const std::vector<occ::core::Atom> &b)
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
    if (!m_a.is_comparable_to(m_b))
        return std::nullopt;
    using occ::Vec3;

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

Vec Dimer::vdw_radii(MoleculeOrder order) const {
    Vec result(m_a.size() + m_b.size());
    switch (order) {
    case MoleculeOrder::AB:
        result << m_a.vdw_radii(), m_b.vdw_radii();
        break;
    case MoleculeOrder::BA:
        result << m_b.vdw_radii(), m_a.vdw_radii();
        break;
    }
    return result;
}

IVec Dimer::atomic_numbers(MoleculeOrder order) const {
    IVec result(m_a.size() + m_b.size());
    switch (order) {
    case MoleculeOrder::AB:
        result << m_a.atomic_numbers(), m_b.atomic_numbers();
        break;
    case MoleculeOrder::BA:
        result << m_b.atomic_numbers(), m_a.atomic_numbers();
        break;
    }
    return result;
}

Mat3N Dimer::positions(MoleculeOrder order) const {
    Mat3N result(3, m_a.size() + m_b.size());
    switch (order) {
    case MoleculeOrder::AB:
        result << m_a.positions(), m_b.positions();
        break;
    case MoleculeOrder::BA:
        result << m_b.positions(), m_a.positions();
        break;
    }
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
    bool aa_eq = m_a.is_equivalent_to(rhs.m_a);
    bool bb_eq = m_b.is_equivalent_to(rhs.m_b);
    if (aa_eq && bb_eq)
        return true;
    bool ba_eq = m_b.is_equivalent_to(rhs.m_a);
    bool ab_eq = m_a.is_equivalent_to(rhs.m_b);
    return ab_eq && ba_eq;
}

bool Dimer::equivalent_in_opposite_frame(const Dimer &rhs) const {
    size_t d1_na = m_a.size();
    size_t d2_na = rhs.m_a.size();
    size_t d1_nb = m_b.size();
    size_t d2_nb = rhs.m_b.size();
    if ((d1_na != d2_nb) || (d1_nb != d2_na))
        return false;
    if (*this != rhs)
        return false;

    Vec3 Od1 = m_b.centroid();
    Vec3 Od2 = rhs.m_a.centroid();
    // positions d1 (with A <-> B swapped)
    Mat3N posd1 = positions(MoleculeOrder::BA);
    posd1.colwise() -= Od1;
    // positions d2
    Mat3N posd2 = rhs.positions(MoleculeOrder::AB);
    posd2.colwise() -= Od2;

    occ::Mat3 rot = kabsch_rotation_matrix(posd1, posd2, false);
    Mat3N posd1_rot = rot * posd1;
    return occ::util::all_close(posd1_rot, posd2, 1e-5, 1e-5);
}

bool Dimer::equivalent(const occ::core::Dimer &rhs) const {
    size_t d1_na = m_a.size();
    size_t d2_na = rhs.m_a.size();
    size_t d1_nb = m_b.size();
    size_t d2_nb = rhs.m_b.size();
    if ((d1_na != d2_na) || (d1_nb != d2_nb))
        return false;
    if (*this != rhs)
        return false;

    Vec3 Od1 = m_a.centroid();
    Vec3 Od2 = rhs.m_a.centroid();
    // positions d1
    Mat3N posd1 = positions(MoleculeOrder::AB);
    posd1.colwise() -= Od1;
    // positions d2
    Mat3N posd2 = rhs.positions(MoleculeOrder::AB);
    posd2.colwise() -= Od2;

    occ::Mat3 rot = kabsch_rotation_matrix(posd1, posd2);
    Mat3N posd1_rot = rot * posd1;
    return occ::util::all_close(posd1_rot, posd2, 1e-5, 1e-5);
}

bool Dimer::equivalent_under_rotation(const occ::core::Dimer &rhs,
                                      const occ::Mat3 &rotation) const {
    size_t d1_na = m_a.size();
    size_t d2_na = rhs.m_a.size();
    size_t d1_nb = m_b.size();
    size_t d2_nb = rhs.m_b.size();
    if ((d1_na != d2_na) || (d1_nb != d2_nb))
        return false;
    if (*this != rhs)
        return false;

    Vec3 Od1 = rotation * m_a.centroid();
    Vec3 Od2 = rhs.m_a.centroid();
    // positions d1
    Mat3N posd1 = rotation * positions(MoleculeOrder::AB);
    posd1.colwise() -= Od1;
    // positions d2
    Mat3N posd2 = rhs.positions(MoleculeOrder::AB);
    posd2.colwise() -= Od2;

    occ::Mat3 rot = kabsch_rotation_matrix(posd1, posd2);
    Mat3N posd1_rot = rot * posd1;
    return occ::util::all_close(posd1_rot, posd2, 1e-5, 1e-5);
}

std::string Dimer::xyz_string() const {
    using occ::core::Element;
    std::string result;

    const auto &pos = positions();
    const auto &nums = atomic_numbers();
    result += fmt::format("{}\n\n", nums.rows());
    for (size_t i = 0; i < nums.rows(); i++) {
        result += fmt::format("{:5s} {:12.5f} {:12.5f} {:12.5f}\n",
                              Element(nums(i)).symbol(), pos(0, i), pos(1, i),
                              pos(2, i));
    }
    return result;
}

} // namespace occ::core
