#include <occ/xtb/cosmo_response.h>

namespace occ::xtb::cosmo {

namespace {

// off-diag(i, j) = 1/|r_i − r_j|; diag(i) = 1.07·√(4π/S_i) (same convention
// as occ::solvent::COSMO).
Mat build_A(const Mat3N &points, const Vec &areas) {
  const Eigen::Index n = points.cols();
  Mat A(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = i + 1; j < n; ++j) {
      const double d = (points.col(i) - points.col(j)).norm();
      const double off = (d > 1e-6) ? 1.0 / d : 0.0;
      A(i, j) = off;
      A(j, i) = off;
    }
  }
  // 1.07 · √(4π) ≈ 3.793051240937804.
  A.diagonal().array() = 3.793051240937804 / areas.array().sqrt();
  return A;
}

// B(i, a) = 1/|r_i − R_a|.
Mat build_B(const Mat3N &surface_points, const Mat3N &atom_positions) {
  const Eigen::Index ncav = surface_points.cols();
  const Eigen::Index natom = atom_positions.cols();
  Mat B(ncav, natom);
  for (Eigen::Index a = 0; a < natom; ++a) {
    for (Eigen::Index i = 0; i < ncav; ++i) {
      const double d = (surface_points.col(i) - atom_positions.col(a)).norm();
      B(i, a) = (d > 1e-6) ? 1.0 / d : 0.0;
    }
  }
  return B;
}

} // namespace

Response build(const Mat3N &atom_positions_bohr,
               const occ::solvent::surface::Surface &surface,
               double epsilon, double x) {
  Response out;
  const Eigen::Index ncav = surface.vertices.cols();
  const Eigen::Index natom = atom_positions_bohr.cols();
  if (ncav == 0) {
    out.B = Mat(0, natom);
    out.G = Mat(0, natom);
    out.J_solv = Mat::Zero(natom, natom);
    return out;
  }

  const double f_eps = (epsilon - 1.0) / (epsilon + x);
  Mat A = build_A(surface.vertices, surface.areas);
  Eigen::PartialPivLU<Mat> lu(A);
  out.B = build_B(surface.vertices, atom_positions_bohr);
  out.G = lu.solve(-f_eps * out.B);
  // B^T · G is mathematically symmetric; symmetrise to absorb round-off.
  Mat J = out.B.transpose() * out.G;
  out.J_solv = 0.5 * (J + J.transpose()).eval();
  return out;
}

Mat3N gradient(const Mat3N &atom_positions_bohr,
               const occ::solvent::surface::Surface &surface,
               const Vec &atom_charges, const Vec &sigma,
               double f_epsilon) {
  const Eigen::Index natom = atom_positions_bohr.cols();
  const Eigen::Index ncav = surface.vertices.cols();
  Mat3N grad = Mat3N::Zero(3, natom);
  if (ncav == 0 || std::abs(f_epsilon) < 1e-14)
    return grad;

  // g_i — field at each cavity point from the atomic source charges.
  Mat3N g_field(3, ncav);
  for (Eigen::Index i = 0; i < ncav; ++i) {
    Mat3N diff = atom_positions_bohr;
    diff.colwise() -= surface.vertices.col(i);
    // -(r_i - R_a) = (R_a - r_i)
    Vec r2 = diff.colwise().squaredNorm();
    Vec3 g = Vec3::Zero();
    for (Eigen::Index a = 0; a < natom; ++a) {
      const double d2 = r2(a);
      if (d2 > 1e-20) {
        const double r3 = d2 * std::sqrt(d2);
        // r_i - R_a = -diff.col(a)
        g -= atom_charges(a) * diff.col(a) / r3;
      }
    }
    g_field.col(i) = g;
  }

  // t_i — field at cavity point i from σ on every other cavity point.
  Mat3N t_field = Mat3N::Zero(3, ncav);
  for (Eigen::Index i = 0; i < ncav; ++i) {
    Vec3 t = Vec3::Zero();
    for (Eigen::Index j = 0; j < ncav; ++j) {
      if (i == j)
        continue;
      Vec3 d = surface.vertices.col(i) - surface.vertices.col(j);
      const double d2 = d.squaredNorm();
      if (d2 > 1e-20) {
        const double r3 = d2 * std::sqrt(d2);
        t -= sigma(j) * d / r3;
      }
    }
    t_field.col(i) = t;
  }

  // h_c — field at each atom from σ on the whole cavity.
  Mat3N h_field = Mat3N::Zero(3, natom);
  for (Eigen::Index c = 0; c < natom; ++c) {
    Vec3 h = Vec3::Zero();
    for (Eigen::Index i = 0; i < ncav; ++i) {
      Vec3 d = surface.vertices.col(i) - atom_positions_bohr.col(c);
      const double d2 = d.squaredNorm();
      if (d2 > 1e-20) {
        const double r3 = d2 * std::sqrt(d2);
        h += sigma(i) * d / r3;
      }
    }
    h_field.col(c) = h;
  }

  // Assemble the per-atom gradient.
  const double inv_f = 1.0 / f_epsilon;
  for (Eigen::Index i = 0; i < ncav; ++i) {
    const int c = surface.atom_index(i);
    grad.col(c) -= sigma(i) * g_field.col(i);
    grad.col(c) += inv_f * sigma(i) * t_field.col(i);
  }
  for (Eigen::Index c = 0; c < natom; ++c) {
    grad.col(c) += atom_charges(c) * h_field.col(c);
  }

  return grad;
}

} // namespace occ::xtb::cosmo
