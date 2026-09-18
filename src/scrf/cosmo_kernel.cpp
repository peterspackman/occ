#include <Eigen/LU>
#include <cmath>
#include <occ/scrf/cosmo_kernel.h>
#include <numbers>    

namespace occ::scrf::detail {

Mat build_cosmo_A(const Mat3N &cavity_points, const Vec &cavity_areas) {
  const Eigen::Index n = cavity_points.cols();
  Mat A(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = i + 1; j < n; ++j) {
      const double d = (cavity_points.col(i) - cavity_points.col(j)).norm();
      const double off = (d > 1e-6) ? 1.0 / d : 0.0;
      A(i, j) = off;
      A(j, i) = off;
    }
  }
  // 1.07 · √(4π) ≈ 3.793051240937804.
  A.diagonal().array() = 3.793051240937804 / cavity_areas.array().sqrt();
  return A;
}

Mat build_atom_cavity_coulomb(const Mat3N &cavity_points,
                              const Mat3N &atom_positions) {
  const Eigen::Index ncav = cavity_points.cols();
  const Eigen::Index natom = atom_positions.cols();
  Mat B(ncav, natom);
  for (Eigen::Index a = 0; a < natom; ++a) {
    for (Eigen::Index i = 0; i < ncav; ++i) {
      const double d = (cavity_points.col(i) - atom_positions.col(a)).norm();
      B(i, a) = (d > 1e-6) ? 1.0 / d : 0.0;
    }
  }
  return B;
}

namespace {

// f(d) = 1/(1 + 6·(rco/d)^k) and its d-derivative f'(d) = f²·6k·(rco/d)^k / d.
struct DampingFactor {
  double f{1.0};
  double df_dd{0.0};
  double df_drco{0.0};
};

DampingFactor damping_factor(double d, double rco, double k) {
  DampingFactor out;
  if (rco <= 0.0)
    return out;
  const double u = std::pow(rco / d, k);
  out.f = 1.0 / (1.0 + 6.0 * u);
  const double t = out.f * out.f * 6.0 * k * u;
  out.df_dd = t / d;
  out.df_drco = -t / rco;
  return out;
}

} // namespace

MultipoleKernel multipole_kernel(const Vec3 &d, MultipoleDamping damping) {
  const double r2 = d.squaredNorm();
  const double g1 = 1.0 / std::sqrt(r2);
  const double g3 = g1 / r2;
  const double g5 = g3 / r2;
  const double third = g3 / 3.0;
  MultipoleKernel k;
  k.t0 = g1;
  k.t1 = d * g3;
  k.t2 = {d.x() * d.x() * g5 - third, 2.0 * d.x() * d.y() * g5,
          d.y() * d.y() * g5 - third, 2.0 * d.x() * d.z() * g5,
          2.0 * d.y() * d.z() * g5, d.z() * d.z() * g5 - third};
  if (damping.active()) {
    const double dist = 1.0 / g1;
    const auto d3 = damping_factor(dist, damping.rco, damping.kdmp3);
    const auto d5 = damping_factor(dist, damping.rco, damping.kdmp5);
    k.t1 *= d3.f;
    for (auto &v : k.t2)
      v *= d5.f;
    k.f3 = d3.f;
    k.f5 = d5.f;
    k.df3_drco = d3.df_drco;
    k.df5_drco = d5.df_drco;
  }
  return k;
}

Vec3 multipole_kernel_gradient(const Vec3 &d, double q, const Vec3 &mu,
                               const double *theta, MultipoleDamping damping) {
  const double r2 = d.squaredNorm();
  const double g1 = 1.0 / std::sqrt(r2);
  const double dist = 1.0 / g1;
  const double g3 = g1 / r2;
  const double g5 = g3 / r2;
  const double g7 = g5 / r2;
  const Vec3 d_hat = d * g1;

  // ∇(q/d) = −q·d/d³. Never damped.
  Vec3 grad = -q * g3 * d;

  // ∇(μ·d/d³) = μ/d³ − 3(μ·d)·d/d⁵, then the product rule for f₃(d).
  const double mu_dot_d = mu.dot(d);
  const double phi_dipole = mu_dot_d * g3;
  Vec3 grad_dipole = g3 * mu - 3.0 * mu_dot_d * g5 * d;
  if (damping.active()) {
    const auto f3 = damping_factor(dist, damping.rco, damping.kdmp3);
    grad_dipole = f3.f * grad_dipole + phi_dipole * f3.df_dd * d_hat;
  }
  grad += grad_dipole;

  if (theta != nullptr) {
    // Θ·d, with the stored off-diagonals used on both sides of the matrix.
    const Vec3 theta_d(theta[0] * d.x() + theta[1] * d.y() + theta[3] * d.z(),
                       theta[1] * d.x() + theta[2] * d.y() + theta[4] * d.z(),
                       theta[3] * d.x() + theta[4] * d.y() + theta[5] * d.z());
    const double dtd = theta_d.dot(d);
    const double trace = theta[0] + theta[2] + theta[5];
    const double phi_quad = dtd * g5 - trace * g3 / 3.0;
    // ∇(d·Θ·d/d⁵) = 2Θd/d⁵ − 5(d·Θ·d)·d/d⁷;  ∇(−tr(Θ)/(3d³)) = tr(Θ)·d/d⁵
    Vec3 grad_quad =
        2.0 * g5 * theta_d - 5.0 * dtd * g7 * d + trace * g5 * d;
    if (damping.active()) {
      const auto f5 = damping_factor(dist, damping.rco, damping.kdmp5);
      grad_quad = f5.f * grad_quad + phi_quad * f5.df_dd * d_hat;
    }
    grad += grad_quad;
  }
  return grad;
}

CosmoResponse build_cosmo_response(const Mat3N &atom_positions_bohr,
                                   const Mat3N &cavity_points,
                                   const Vec &cavity_areas, double epsilon,
                                   double x) {
  CosmoResponse out;
  const Eigen::Index ncav = cavity_points.cols();
  const Eigen::Index natom = atom_positions_bohr.cols();
  if (ncav == 0) {
    out.B = Mat(0, natom);
    out.G = Mat(0, natom);
    out.J_solv = Mat::Zero(natom, natom);
    return out;
  }

  const double f_eps = (epsilon - 1.0) / (epsilon + x);
  Mat A = build_cosmo_A(cavity_points, cavity_areas);
  Eigen::PartialPivLU<Mat> lu(A);
  out.B = build_atom_cavity_coulomb(cavity_points, atom_positions_bohr);
  out.G = lu.solve(-f_eps * out.B);
  // B^T · G is mathematically symmetric; symmetrise to absorb round-off.
  Mat J = out.B.transpose() * out.G;
  out.J_solv = 0.5 * (J + J.transpose()).eval();
  return out;
}

Mat3N cosmo_gradient_frozen(const Mat3N &atom_positions_bohr,
                            const Mat3N &cavity_points,
                            const Vec &cavity_areas,
                            const IVec &cavity_atom_index,
                            const Vec &atom_charges, const Vec &sigma,
                            double f_epsilon, const Vec &atom_radii_bohr,
                            double smoothing_width_bohr,
                            const Mat3N &atom_dipoles,
                            const Mat &atom_quadrupoles,
                            const Vec &damping_rco_bohr, double kdmp3,
                            double kdmp5) {
  const Eigen::Index natom = atom_positions_bohr.cols();
  const Eigen::Index ncav = cavity_points.cols();
  Mat3N grad = Mat3N::Zero(3, natom);
  if (ncav == 0 || std::abs(f_epsilon) < 1e-14)
    return grad;

  const bool have_dipoles = atom_dipoles.cols() == natom;
  const bool have_quadrupoles = atom_quadrupoles.cols() == natom;
  const bool have_damping = damping_rco_bohr.size() == natom;

  // Source term. For each (cavity point i, atom a) pair, u = ∂φ_i/∂d with
  // d = r_i − R_a. The cavity point rides on its parent atom and the source
  // sits on atom a, so the pair contributes ±σ_i·u to the two of them —
  // equal and opposite, which keeps the total gradient translation-invariant.
  // With no dipoles or quadrupoles this is exactly the old g/h pair of loops.
  Mat3N source_grad = Mat3N::Zero(3, natom);
  for (Eigen::Index i = 0; i < ncav; ++i) {
    const int c = cavity_atom_index(i);
    Vec3 on_cavity = Vec3::Zero();
    for (Eigen::Index a = 0; a < natom; ++a) {
      const Vec3 d = cavity_points.col(i) - atom_positions_bohr.col(a);
      if (d.squaredNorm() < 1e-20)
        continue;
      const MultipoleDamping damping{
          have_damping ? damping_rco_bohr(a) : 0.0, kdmp3, kdmp5};
      const Vec3 u = multipole_kernel_gradient(
          d, atom_charges(a),
          have_dipoles ? Vec3(atom_dipoles.col(a)) : Vec3::Zero(),
          have_quadrupoles ? atom_quadrupoles.col(a).data() : nullptr,
          damping);
      on_cavity += u;
      source_grad.col(a) -= sigma(i) * u;
    }
    source_grad.col(c) += sigma(i) * on_cavity;
  }

  // t_i — field at cavity point i from σ on every other cavity point.
  Mat3N t_field = Mat3N::Zero(3, ncav);
  for (Eigen::Index i = 0; i < ncav; ++i) {
    Vec3 t = Vec3::Zero();
    for (Eigen::Index j = 0; j < ncav; ++j) {
      if (i == j)
        continue;
      Vec3 d = cavity_points.col(i) - cavity_points.col(j);
      const double d2 = d.squaredNorm();
      if (d2 > 1e-20) {
        const double r3 = d2 * std::sqrt(d2);
        t -= sigma(j) * d / r3;
      }
    }
    t_field.col(i) = t;
  }

  // Assemble the per-atom gradient.
  const double inv_f = 1.0 / f_epsilon;
  grad += source_grad;
  for (Eigen::Index i = 0; i < ncav; ++i) {
    grad.col(cavity_atom_index(i)) += inv_f * sigma(i) * t_field.col(i);
  }

  // Smooth-cavity diagonal A term. Only contributes when the caller opts in
  // via `smoothing_width_bohr > 0` (the cavity itself must have been built
  // with the same smoothing for the formula to be self-consistent).
  //
  //   ∂A_ii/∂R_c = -½ A_ii · ∂ln(weight_i)/∂R_c
  //   ∂ln(weight_i)/∂R_c = Σ_{k ≠ a_i} (s'/s)|d_ik · ∂d_ik/∂R_c
  //   contribution to ∂E/∂R_c: (1/(2 f(ε))) Σ_i σ_i² ∂A_ii/∂R_c
  //                          = -(1/(4 f(ε))) Σ_i σ_i² A_ii ∂ln(weight_i)/∂R_c
  if (smoothing_width_bohr > 0.0 && atom_radii_bohr.size() == natom) {
    const double sqrt_pi = std::sqrt(std::numbers::pi_v<double>);
    for (Eigen::Index i = 0; i < ncav; ++i) {
      const int atom_i = cavity_atom_index(i);
      // A_ii = 1.07·√(4π/area_i) = 3.793051240937804 / √area_i
      const double a_ii = 3.793051240937804 / std::sqrt(cavity_areas(i));
      const double prefac = -0.25 / f_epsilon * sigma(i) * sigma(i) * a_ii;
      const Vec3 r_i = cavity_points.col(i);
      for (Eigen::Index k = 0; k < natom; ++k) {
        if (k == atom_i)
          continue;
        const Vec3 d_vec = r_i - atom_positions_bohr.col(k);
        const double d = d_vec.norm();
        if (d < 1e-12)
          continue;
        const double t_k = atom_radii_bohr(k);
        const double arg = (d - t_k) / smoothing_width_bohr;
        const double s = 0.5 * (1.0 + std::erf(arg));
        if (s < 1e-12)
          continue; // weight effectively zero — contribution negligible
        const double s_prime =
            std::exp(-arg * arg) / (smoothing_width_bohr * sqrt_pi);
        const double dlog_dd = s_prime / s;
        const Vec3 d_hat = d_vec / d;
        // ∂d_ik/∂R_atom_i = +d_hat;  ∂d_ik/∂R_k = -d_hat (rigid attachment).
        grad.col(atom_i) += prefac * dlog_dd * d_hat;
        grad.col(k) -= prefac * dlog_dd * d_hat;
      }
    }
  }

  return grad;
}

} // namespace occ::scrf::detail
