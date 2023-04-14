#include <cmath>
#include <occ/crystal/unitcell.h>

namespace occ::crystal {

UnitCell::UnitCell()
    : m_lengths{1, 1, 1}, m_angles{occ::units::PI / 2, occ::units::PI / 2,
                                   occ::units::PI / 2} {}

UnitCell::UnitCell(const Vec3 &lengths, const Vec3 &angles)
    : m_lengths{lengths}, m_angles{angles} {
    update_cell_matrices();
}

UnitCell::UnitCell(double a, double b, double c, double alpha, double beta,
                   double gamma)
    : m_lengths{a, b, c}, m_angles{alpha, beta, gamma} {
    update_cell_matrices();
}

UnitCell::UnitCell(const Mat3 &vectors) : m_lengths(vectors.colwise().norm()) {
    Vec3 u_a = vectors.col(0) / m_lengths(0);
    Vec3 u_b = vectors.col(1) / m_lengths(1);
    Vec3 u_c = vectors.col(2) / m_lengths(2);
    m_angles(0) = std::acos(std::clamp(u_b.dot(u_c), -1.0, 1.0));
    m_angles(1) = std::acos(std::clamp(u_c.dot(u_a), -1.0, 1.0));
    m_angles(2) = std::acos(std::clamp(u_a.dot(u_b), -1.0, 1.0));
    update_cell_matrices();
}

void UnitCell::set_a(double a) {
    if (a >= 0.0) {
        m_lengths(0) = a;
        update_cell_matrices();
    }
}

void UnitCell::set_b(double b) {
    if (b >= 0.0) {
        m_lengths[1] = b;
        update_cell_matrices();
    }
}

void UnitCell::set_c(double c) {
    if (c >= 0.0) {
        m_lengths[2] = c;
        update_cell_matrices();
    }
}

void UnitCell::set_alpha(double a) {
    if (a >= 0.0) {
        m_angles[0] = a;
        update_cell_matrices();
    }
}

void UnitCell::set_beta(double b) {
    if (b >= 0.0) {
        m_angles[1] = b;
        update_cell_matrices();
    }
}

void UnitCell::set_gamma(double c) {
    if (c >= 0.0) {
        m_angles[2] = c;
        update_cell_matrices();
    }
}

void UnitCell::update_cell_matrices() {
    for (int i = 0; i < 3; i++) {
        m_sin[i] = sin(m_angles[i]);
        m_cos[i] = cos(m_angles[i]);
    }

    const double a = m_lengths[0], ca = m_cos[0];
    const double b = m_lengths[1], cb = m_cos[1];
    const double c = m_lengths[2], sg = m_sin[2], cg = m_cos[2];
    m_volume =
        a * b * c * sqrt(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg);

    m_direct << a, b * cg, c * cb, 0.0, b * sg, c * (ca - cb * cg) / sg, 0.0,
        0.0, m_volume / (a * b * sg);

    m_reciprocal << 1 / a, 0.0, 0.0, -cg / (a * sg), 1 / (b * sg), 0.0,
        b * c * (ca * cg - cb) / (m_volume * sg),
        a * c * (cb * cg - ca) / (m_volume * sg), a * b * sg / m_volume;

    m_inverse = m_reciprocal.transpose();
}

bool UnitCell::is_cubic() const { return _abc_close() && is_orthogonal(); }

bool UnitCell::is_triclinic() const {
    return _abc_different() && _a_abc_different();
}

bool UnitCell::is_monoclinic() const {
    return _a_ac_close() && _abc_different();
}

bool UnitCell::is_orthorhombic() const {
    return is_orthogonal() && _abc_different();
}

bool UnitCell::is_tetragonal() const {
    return _ab_close() && !_ac_close() && is_orthogonal();
}

bool UnitCell::is_rhombohedral() const {
    return _abc_close() && _a_abc_close() && !_a_90();
}

bool UnitCell::is_hexagonal() const {
    return _ab_close() && !_ac_close() && _a_90() &&
           is_close(m_angles[2], 2 * occ::units::PI / 3);
}

std::string UnitCell::cell_type() const {
    if (is_cubic())
        return "cubic";
    if (is_rhombohedral())
        return "rhombohedral";
    if (is_hexagonal())
        return "hexagonal";
    if (is_tetragonal())
        return "tetragonal";
    if (is_orthorhombic())
        return "orthorhombic";
    if (is_monoclinic())
        return "monoclinic";
    return "triclinic";
}

HKL UnitCell::hkl_limits(double d_min) const {
    return {static_cast<int>(m_lengths[0] / d_min),
            static_cast<int>(m_lengths[1] / d_min),
            static_cast<int>(m_lengths[2] / d_min)};
}

/*
 * UnitCell builders
 */

UnitCell cubic_cell(double length) {
    return UnitCell(length, length, length, occ::units::PI / 2,
                    occ::units::PI / 2, occ::units::PI / 2);
}

UnitCell rhombohedral_cell(double length, double angle) {
    return UnitCell(length, length, length, angle, angle, angle);
}

UnitCell tetragonal_cell(double a, double c) {
    return UnitCell(a, a, c, occ::units::PI / 2, occ::units::PI / 2,
                    occ::units::PI / 2);
}

UnitCell hexagonal_cell(double a, double c) {
    return UnitCell(a, a, c, occ::units::PI / 2, occ::units::PI / 2,
                    2 * occ::units::PI / 3);
}

UnitCell orthorhombic_cell(double a, double b, double c) {
    return UnitCell(a, b, c, occ::units::PI / 2, occ::units::PI / 2,
                    occ::units::PI / 2);
}

UnitCell monoclinic_cell(double a, double b, double c, double angle) {

    return UnitCell(a, b, c, occ::units::PI / 2, angle, occ::units::PI / 2);
}

UnitCell triclinic_cell(double a, double b, double c, double alpha, double beta,
                        double gamma) {
    return UnitCell(a, b, c, alpha, beta, gamma);
}

} // namespace occ::crystal
