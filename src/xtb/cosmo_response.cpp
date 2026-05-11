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
    out.G = Mat(0, natom);
    out.J_solv = Mat::Zero(natom, natom);
    return out;
  }

  const double f_eps = (epsilon - 1.0) / (epsilon + x);
  Mat A = build_A(surface.vertices, surface.areas);
  Eigen::PartialPivLU<Mat> lu(A);
  Mat B = build_B(surface.vertices, atom_positions_bohr);
  out.G = lu.solve(-f_eps * B);
  // B^T · G is mathematically symmetric; symmetrise to absorb round-off.
  Mat J = B.transpose() * out.G;
  out.J_solv = 0.5 * (J + J.transpose()).eval();
  return out;
}

} // namespace occ::xtb::cosmo
