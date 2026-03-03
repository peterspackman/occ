#include <occ/mults/short_range.h>
#include <cmath>
#include <stdexcept>

namespace occ::mults {

// ==================== Lennard-Jones Implementation ====================

double ShortRangeInteraction::lennard_jones_energy(double r, const LennardJonesParams& params) {
    if (r < MIN_DISTANCE) {
        throw std::domain_error("Distance r must be greater than MIN_DISTANCE");
    }

    const double sigma_over_r = params.sigma / r;
    const double sr6 = sigma_over_r * sigma_over_r * sigma_over_r *
                       sigma_over_r * sigma_over_r * sigma_over_r;
    const double sr12 = sr6 * sr6;

    return 4.0 * params.epsilon * (sr12 - sr6);
}

double ShortRangeInteraction::lennard_jones_derivative(double r, const LennardJonesParams& params) {
    if (r < MIN_DISTANCE) {
        throw std::domain_error("Distance r must be greater than MIN_DISTANCE");
    }

    const double sigma_over_r = params.sigma / r;
    const double sr6 = sigma_over_r * sigma_over_r * sigma_over_r *
                       sigma_over_r * sigma_over_r * sigma_over_r;
    const double sr12 = sr6 * sr6;

    return 24.0 * params.epsilon / r * (sr6 - 2.0 * sr12);
}

double ShortRangeInteraction::lennard_jones_second_derivative(double r, const LennardJonesParams& params) {
    if (r < MIN_DISTANCE) {
        throw std::domain_error("Distance r must be greater than MIN_DISTANCE");
    }

    const double sigma_over_r = params.sigma / r;
    const double sr6 = sigma_over_r * sigma_over_r * sigma_over_r *
                       sigma_over_r * sigma_over_r * sigma_over_r;
    const double sr12 = sr6 * sr6;

    return 24.0 * params.epsilon / (r * r) * (26.0 * sr12 - 7.0 * sr6);
}

ShortRangeInteraction::EnergyAndDerivatives
ShortRangeInteraction::lennard_jones_all(double r, const LennardJonesParams& params) {
    if (r < MIN_DISTANCE) {
        throw std::domain_error("Distance r must be greater than MIN_DISTANCE");
    }

    // Compute intermediate terms once
    const double sigma_over_r = params.sigma / r;
    const double sr2 = sigma_over_r * sigma_over_r;
    const double sr6 = sr2 * sr2 * sr2;
    const double sr12 = sr6 * sr6;

    const double four_epsilon = 4.0 * params.epsilon;
    const double r_inv = 1.0 / r;
    const double r_inv2 = r_inv * r_inv;

    EnergyAndDerivatives result;
    result.energy = four_epsilon * (sr12 - sr6);
    result.first_derivative = 24.0 * params.epsilon * r_inv * (sr6 - 2.0 * sr12);
    result.second_derivative = 24.0 * params.epsilon * r_inv2 * (26.0 * sr12 - 7.0 * sr6);

    return result;
}

// ==================== Buckingham Implementation ====================

double ShortRangeInteraction::buckingham_energy(double r, const BuckinghamParams& params) {
    if (r < MIN_DISTANCE) {
        throw std::domain_error("Distance r must be greater than MIN_DISTANCE");
    }

    const double exp_term = params.A * std::exp(-params.B * r);
    const double r2 = r * r;
    const double r6 = r2 * r2 * r2;
    const double disp_term = params.C / r6;

    return exp_term - disp_term;
}

double ShortRangeInteraction::buckingham_derivative(double r, const BuckinghamParams& params) {
    if (r < MIN_DISTANCE) {
        throw std::domain_error("Distance r must be greater than MIN_DISTANCE");
    }

    const double exp_term = params.A * std::exp(-params.B * r);
    const double r2 = r * r;
    const double r6 = r2 * r2 * r2;
    const double r7 = r6 * r;

    const double rep_deriv = -params.A * params.B * std::exp(-params.B * r);
    const double disp_deriv = 6.0 * params.C / r7;

    return rep_deriv + disp_deriv;
}

double ShortRangeInteraction::buckingham_second_derivative(double r, const BuckinghamParams& params) {
    if (r < MIN_DISTANCE) {
        throw std::domain_error("Distance r must be greater than MIN_DISTANCE");
    }

    const double r2 = r * r;
    const double r6 = r2 * r2 * r2;
    const double r8 = r6 * r2;

    const double rep_deriv2 = params.A * params.B * params.B * std::exp(-params.B * r);
    const double disp_deriv2 = -42.0 * params.C / r8;

    return rep_deriv2 + disp_deriv2;
}

ShortRangeInteraction::EnergyAndDerivatives
ShortRangeInteraction::buckingham_all(double r, const BuckinghamParams& params) {
    if (r < MIN_DISTANCE) {
        throw std::domain_error("Distance r must be greater than MIN_DISTANCE");
    }

    // Compute intermediate terms once
    const double exp_Br = std::exp(-params.B * r);
    const double A_exp_Br = params.A * exp_Br;
    const double AB_exp_Br = params.A * params.B * exp_Br;
    const double AB2_exp_Br = AB_exp_Br * params.B;

    const double r2 = r * r;
    const double r4 = r2 * r2;
    const double r6 = r4 * r2;
    const double r7 = r6 * r;
    const double r8 = r6 * r2;

    const double C_r6 = params.C / r6;
    const double six_C_r7 = 6.0 * params.C / r7;
    const double fortytwo_C_r8 = 42.0 * params.C / r8;

    EnergyAndDerivatives result;
    result.energy = A_exp_Br - C_r6;
    result.first_derivative = -AB_exp_Br + six_C_r7;
    result.second_derivative = AB2_exp_Br - fortytwo_C_r8;

    return result;
}

// ==================== Anisotropic Repulsion ====================

ShortRangeInteraction::AnisotropicResult
ShortRangeInteraction::anisotropic_repulsion(
    const Vec3& pos_a, const Vec3& pos_b,
    const Vec3& axis_a, const Vec3& axis_b,
    const AnisotropicRepulsionParams& params) {

    AnisotropicResult result;
    result.energy = 0.0;
    result.force_A = Vec3::Zero();
    result.force_B = Vec3::Zero();
    result.torque_axis_A = Vec3::Zero();
    result.torque_axis_B = Vec3::Zero();

    const Vec3 r_ab = pos_b - pos_a;
    const double R = r_ab.norm();
    if (R < MIN_DISTANCE) {
        return result;
    }

    const double R_inv = 1.0 / R;
    const Vec3 u = r_ab * R_inv;  // unit vector from A to B

    // Compute cosθ for each axis (zero axis = no anisotropy for that site)
    const bool has_axis_a = axis_a.squaredNorm() > 0.5;
    const bool has_axis_b = axis_b.squaredNorm() > 0.5;

    const double cos_a = has_axis_a ? axis_a.dot(u) : 0.0;
    const double cos_b = has_axis_b ? axis_b.dot(u) : 0.0;

    // P₂(x) = (3x² - 1)/2, dP₂/dx = 3x
    // When an axis is zero (not aniso), its P₂ contribution is zero.
    const double P2_a = has_axis_a ? 0.5 * (3.0 * cos_a * cos_a - 1.0) : 0.0;
    const double P2_b = has_axis_b ? 0.5 * (3.0 * cos_b * cos_b - 1.0) : 0.0;

    const double rho = params.rho_00 + params.rho_20 * P2_a + params.rho_02 * P2_b;

    // E = exp(-α·R + α·ρ) = exp(α·(ρ - R))
    const double alpha = params.alpha;
    const double E = std::exp(alpha * (rho - R));

    result.energy = E;

    // Derivatives: E = exp(α(ρ - R))
    // dE/d(pos_a) = E·α·[dρ/d(pos_a) - dR/d(pos_a)]
    //             = E·α·[dρ/d(pos_a) + u]   since dR/d(pos_a) = -u
    // dE/d(pos_b) = E·α·[dρ/d(pos_b) - u]   since dR/d(pos_b) = +u

    // d(cosθ₁)/d(pos_a) = -(1/R)(axis_a - cosθ₁·u)
    // d(cosθ₁)/d(pos_b) = +(1/R)(axis_a - cosθ₁·u)
    // dρ/d(pos_a) = ρ₂₀·3cosθ₁·d(cosθ₁)/d(pos_a) + ρ₀₂·3cosθ₂·d(cosθ₂)/d(pos_a)

    // Perpendicular projections of axes onto plane normal to u
    Vec3 perp_a = Vec3::Zero();
    Vec3 perp_b = Vec3::Zero();
    if (has_axis_a) perp_a = axis_a - cos_a * u;
    if (has_axis_b) perp_b = axis_b - cos_b * u;

    // dρ/d(pos) contributions (factor of 1/R from d(cosθ)/d(pos))
    // dρ/d(pos_a) = -(1/R) * [ρ₂₀·3cosθ_a·perp_a + ρ₀₂·3cosθ_b·perp_b]
    // dρ/d(pos_b) = +(1/R) * [ρ₂₀·3cosθ_a·perp_a + ρ₀₂·3cosθ_b·perp_b]
    const Vec3 drho_angular = 3.0 * (params.rho_20 * cos_a * perp_a +
                                      params.rho_02 * cos_b * perp_b);

    const Vec3 drho_dposA = -R_inv * drho_angular;
    const Vec3 drho_dposB =  R_inv * drho_angular;

    // Forces: F_A = -dE/d(pos_a), F_B = -dE/d(pos_b)
    const Vec3 grad_posA = E * alpha * (drho_dposA + u);
    const Vec3 grad_posB = E * alpha * (drho_dposB - u);

    result.force_A = -grad_posA;
    result.force_B = -grad_posB;

    // Axis rotation torques:
    // For an infinitesimal rotation ψ of axis_a: δ(axis_a) = ψ × axis_a
    // d(cosθ₁)/d(axis_a) = u (but restricted to rotation)
    // dE/dψ_a = E·α·ρ₂₀·3cosθ₁ · (axis_a × u)
    if (has_axis_a && std::abs(params.rho_20) > 1e-20) {
        result.torque_axis_A = E * alpha * params.rho_20 * 3.0 * cos_a *
                               axis_a.cross(u);
    }
    if (has_axis_b && std::abs(params.rho_02) > 1e-20) {
        result.torque_axis_B = E * alpha * params.rho_02 * 3.0 * cos_b *
                               axis_b.cross(u);
    }

    return result;
}

// ==================== Force and Hessian Transformations ====================

Vec3 ShortRangeInteraction::derivative_to_force(double dE_dr, const Vec3& r_vec) {
    const double r = r_vec.norm();

    if (r < MIN_DISTANCE) {
        return Vec3::Zero();
    }

    // F_A = +(dE/dr) * (r_vec/r)
    // where r_vec = r_B - r_A
    // This is because F_A = -grad_A(E) = -dE/dr * d(r)/d(r_A) = -dE/dr * (-r_hat) = +dE/dr * r_hat
    const Vec3 unit_vec = r_vec / r;
    return dE_dr * unit_vec;
}

ShortRangeInteraction::ForceAndHessian
ShortRangeInteraction::compute_force_hessian(
    const Vec3& r_A, const Vec3& r_B,
    double dE_dr, double d2E_dr2) {

    ForceAndHessian result;

    // Compute displacement vector and distance
    const Vec3 r_vec = r_B - r_A;
    const double r = r_vec.norm();

    if (r < MIN_DISTANCE) {
        // Return zeros for very small distances to avoid singularities
        result.force_A = Vec3::Zero();
        result.force_B = Vec3::Zero();
        result.hessian_AA = Mat3::Zero();
        result.hessian_AB = Mat3::Zero();
        result.hessian_BB = Mat3::Zero();
        return result;
    }

    // Compute unit vector
    const Vec3 unit_vec = r_vec / r;

    // Forces (Newton's 3rd law: F_A + F_B = 0)
    result.force_A = -dE_dr * unit_vec;
    result.force_B = -result.force_A;

    // Hessian for central potential:
    // H_ij = d²E/dx_i dx_j
    //      = (d²E/dr²) * (r_i/r) * (r_j/r) + (dE/dr/r) * [δ_ij - (r_i/r)(r_j/r)]
    //      = curvature_term + angular_term
    //
    // This can be written as:
    // H = (d²E/dr² - dE/dr/r) * (unit_vec ⊗ unit_vec) + (dE/dr/r) * I

    const double r_inv = 1.0 / r;
    const double curvature_coeff = d2E_dr2 - dE_dr * r_inv;
    const double angular_coeff = dE_dr * r_inv;

    // Outer product: unit_vec ⊗ unit_vec
    const Mat3 outer_product = unit_vec * unit_vec.transpose();
    const Mat3 identity = Mat3::Identity();

    // Force Hessian = ∂F/∂r = -∂²E/∂r² (negative of energy Hessian)
    // Since F = -∇E, we need to negate the energy Hessian
    result.hessian_AA = -(curvature_coeff * outer_product + angular_coeff * identity);

    // By symmetry: ∂F_A/∂r_B = -∂F_A/∂r_A, ∂F_B/∂r_B = ∂F_A/∂r_A
    result.hessian_AB = -result.hessian_AA;
    result.hessian_BB = result.hessian_AA;

    return result;
}

} // namespace occ::mults
