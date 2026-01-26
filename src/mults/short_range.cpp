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
