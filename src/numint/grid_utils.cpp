#include <occ/core/parallel.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/numint/grid_utils.h>
#include <occ/numint/lebedev.h>
#include <stdexcept>
#include <numbers>

namespace occ::numint {

// Available Lebedev grid levels
const std::array<uint_fast16_t, 33> lebedev_grid_levels{
    1,    6,    14,   26,   38,   50,   74,   86,   110,  146,  170,
    194,  230,  266,  302,  350,  434,  590,  770,  974,  1202, 1454,
    1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810};

// Bragg radii for all elements (need to multiply by (1.0 / BOHR) to get the
// real RADII_BRAGG)
const std::array<double, 131> bragg_radii = {
    0.35, 1.40,                                                 // 1s
    1.45, 1.05, 0.85, 0.70, 0.65, 0.60, 0.50, 1.50,             // 2s2p
    1.80, 1.50, 1.25, 1.10, 1.00, 1.00, 1.00, 1.80,             // 3s3p
    2.20, 1.80,                                                 // 4s
    1.60, 1.40, 1.35, 1.40, 1.40, 1.40, 1.35, 1.35, 1.35, 1.35, // 3d
    1.30, 1.25, 1.15, 1.15, 1.15, 1.90,                         // 4p
    2.35, 2.00,                                                 // 5s
    1.80, 1.55, 1.45, 1.45, 1.35, 1.30, 1.35, 1.40, 1.60, 1.55, // 4d
    1.55, 1.45, 1.45, 1.40, 1.40, 2.10,                         // 5p
    2.60, 2.15,                                                 // 6s
    1.95, 1.85, 1.85, 1.85, 1.85, 1.85, 1.85,                   // La, Ce-Eu
    1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,             // Gd, Tb-Lu
    1.55, 1.45, 1.35, 1.35, 1.30, 1.35, 1.35, 1.35, 1.50,       // 5d
    1.90, 1.80, 1.60, 1.90, 1.45, 2.10,                         // 6p
    1.80, 2.15,                                                 // 7s
    1.95, 1.80, 1.80, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75,
    1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75};

uint_fast16_t nearest_grid_level_at_or_above(uint_fast16_t n) {
  for (int i = 0; i < lebedev_grid_levels.size(); i++) {
    auto m = lebedev_grid_levels[i];
    if (m >= n)
      return m;
  }

  throw std::runtime_error("Requested number of angular points exceeds 5810");
  // unreachable
  return 1;
}

uint_fast16_t nearest_grid_level_below(uint_fast16_t n) {
  if (n <= lebedev_grid_levels[0]) {
    return lebedev_grid_levels[0];
  }

  for (int i = lebedev_grid_levels.size() - 1; i >= 0; i--) {
    auto m = lebedev_grid_levels[i];
    if (m < n)
      return m;
  }

  // If no level is below, return the smallest
  return lebedev_grid_levels[0];
}

size_t angular_point_count(size_t level) {
  // Make sure the level is a valid Lebedev grid level
  size_t actual_level = nearest_grid_level_at_or_above(level);

  // For valid Lebedev grid levels, the level is the number of points
  return actual_level;
}

double get_atomic_radius(size_t atomic_number) {
  if (atomic_number < 1 || atomic_number > bragg_radii.size()) {
    occ::log::warn("Invalid atomic number {}. Using default radius of 1.0",
                   atomic_number);
    return 1.0 * occ::units::ANGSTROM_TO_BOHR;
  }

  return bragg_radii[atomic_number - 1] * occ::units::ANGSTROM_TO_BOHR;
}

Vec becke_partition(const Vec &w) {
  Vec result = w;
  for (size_t i = 0; i < 3; i++) {
    result.array() =
        (3 - result.array() * result.array()) * result.array() * 0.5;
  }
  return result;
}

Vec stratmann_scuseria_partition(const Vec &w) {
  Vec result(w.rows());
  constexpr double a = 0.64;
  for (size_t i = 0; i < w.rows(); i++) {
    double ma = w(i) / a;
    double ma2 = ma * ma;
    double det = ma / 16 * (35 + ma2 * (-35 + ma2 * (21 - 5 * ma2)));
    result(i) = (det <= a) ? -1 : 1;
  }
  return result;
}

Mat calculate_interatomic_distances(const Mat3N &positions) {
  size_t natoms = positions.cols();
  Mat dists(natoms, natoms);
  for (size_t i = 0; i < natoms; i++) {
    dists(i, i) = 0;
    for (size_t j = i + 1; j < natoms; j++) {
      double dx = positions(0, i) - positions(0, j);
      double dy = positions(1, i) - positions(1, j);
      double dz = positions(2, i) - positions(2, j);
      dists(i, j) = sqrt(dx * dx + dy * dy + dz * dz);
      dists(j, i) = dists(i, j);
    }
  }
  return dists;
}

struct TreutlerAlrichsAdjustment {
  inline void operator()(double r_i, double r_j, Eigen::Ref<Vec> w) {
    // Apply Treutler-Alrichs adjustment
    if (std::fabs(r_i - r_j) > 1e-14) {
      double xi = sqrt(r_i / r_j);
      double u_ij = (xi - 1) / (xi + 1);
      double a_ij = std::clamp(u_ij / (u_ij * u_ij - 1.0), -0.5, 0.5);
      w.array() += a_ij * (1 - w.array() * w.array());
    }
  }
};

struct BeckeAdjustment {
  inline void operator()(double r_i, double r_j, Eigen::Ref<Vec> w) {
    // Apply Treutler-Alrichs adjustment
    if (std::fabs(r_i - r_j) > 1e-14) {
      double xi = r_i / r_j;
      double u_ij = (xi - 1) / (xi + 1);
      double a_ij = std::clamp(u_ij / (u_ij * u_ij - 1.0), -0.5, 0.5);
      w.array() += a_ij * (1 - w.array() * w.array());
    }
  }
};

template <class RadiiAdjustment = TreutlerAlrichsAdjustment,
          typename PartitionFn>
Mat calculate_atomic_grid_weights_impl(const Mat &grid_points,
                                       const Mat &atomic_positions,
                                       const Mat &interatomic_distances,
                                       const Vec &radii,
                                       PartitionFn &&partition_fn) {

  const size_t n_grid_points = grid_points.cols();
  const size_t n_atoms = atomic_positions.cols();

  // Calculate distances from grid points to all atoms
  Mat grid_dists(n_grid_points, n_atoms);
  for (size_t i = 0; i < n_atoms; i++) {
    Vec3 xyz = atomic_positions.col(i);
    grid_dists.col(i) = (grid_points.colwise() - xyz).colwise().norm();
  }

  // Initialize weights matrix
  Mat weights = Mat::Ones(n_grid_points, n_atoms);

  RadiiAdjustment adjust_for_radii;

  // Calculate weights using Becke's scheme with Treutler-Alrichs adjustment
  for (size_t i = 0; i < n_atoms; i++) {
    double r_i = radii(i);

    for (size_t j = 0; j < i; j++) {
      double r_j = radii(j);

      // Calculate raw weights
      Vec w = (grid_dists.col(i).array() - grid_dists.col(j).array()) /
              interatomic_distances(i, j);

      adjust_for_radii(r_i, r_j, w);

      w = partition_fn(w);

      for (size_t idx = 0; idx < w.rows(); idx++) {
        double v = w(idx);
        if (std::fabs(1.0 - v) < 1e-14) {
          weights(idx, i) = 0.0;
        } else {
          weights(idx, i) *= 0.5 * (1.0 - v);
          weights(idx, j) *= 0.5 * (1.0 + v);
        }
      }
    }
  }

  return weights;
}

namespace {

// Size-adjustment parameter a_ij exactly as the adjustment functors above
// compute it for the ordered pair (i, j), i > j, including Becke's clamp to
// |a_ij| <= 1/2 (beyond it nu(mu) is no longer monotonic; reached for H next
// to the heavy alkali metals).
inline double size_adjustment(double r_i, double r_j, bool treutler_ahlrichs) {
  if (std::fabs(r_i - r_j) <= 1e-14)
    return 0.0;
  const double xi = treutler_ahlrichs ? std::sqrt(r_i / r_j) : r_i / r_j;
  const double u = (xi - 1) / (xi + 1);
  return std::clamp(u / (u * u - 1.0), -0.5, 0.5);
}

// a(i, j) for every ordered pair, a(j, i) = -a(i, j).
Mat size_adjustment_matrix(const Vec &radii, bool treutler_ahlrichs) {
  const Eigen::Index n = radii.size();
  Mat a = Mat::Zero(n, n);
  for (Eigen::Index i = 0; i < n; i++)
    for (Eigen::Index j = 0; j < i; j++) {
      a(i, j) = size_adjustment(radii(i), radii(j), treutler_ahlrichs);
      a(j, i) = -a(i, j);
    }
  return a;
}

// Becke's iterated polynomial f(nu) = p(p(p(nu))), p(x) = (3 - x^2) x / 2,
// and its derivative.
inline void becke_f(double nu, double &f, double &df) {
  double x = nu, d = 1.0;
  for (int k = 0; k < 3; k++) {
    d *= 1.5 * (1.0 - x * x);
    x = 0.5 * (3.0 - x * x) * x;
  }
  f = x;
  df = d;
}

// f, f' and f'' of the iterated Becke polynomial f = p(p(p(nu))).
inline void becke_f2(double nu, double &f, double &df, double &d2f) {
  // chain rule through three applications of p(x) = (3 - x^2) x / 2,
  // p' = 3/2 (1 - x^2), p'' = -3 x
  double x = nu, d = 1.0, dd = 0.0;
  for (int k = 0; k < 3; k++) {
    const double p1 = 1.5 * (1.0 - x * x), p2 = -3.0 * x;
    dd = p2 * d * d + p1 * dd;
    d = p1 * d;
    x = 0.5 * (3.0 - x * x) * x;
  }
  f = x;
  df = d;
  d2f = dd;
}

} // namespace

BeckeWeightDerivative::BeckeWeightDerivative(const Mat3N &positions,
                                             const Vec &radii,
                                             bool treutler_ahlrichs)
    : m_positions(positions), m_natom(static_cast<int>(positions.cols())) {
  const Mat R = calculate_interatomic_distances(positions);
  m_invR = Mat::Zero(m_natom, m_natom);
  for (int i = 0; i < m_natom; i++)
    for (int j = 0; j < m_natom; j++)
      if (i != j)
        m_invR(i, j) = 1.0 / R(i, j);
  m_aT = size_adjustment_matrix(radii, treutler_ahlrichs).transpose();
}

Mat BeckeWeightDerivative::batch(Eigen::Ref<const Mat3N> points,
                                 Eigen::Ref<const Vec> w, int parent) const {
  const Eigen::Index n3 = 3 * m_natom;
  Mat dW = Mat::Zero(points.cols(), n3);
  Mat3N dw(3, m_natom);
  for (Eigen::Index g = 0; g < points.cols(); g++)
    if ((*this)(points.col(g), w(g), parent, dw))
      dW.row(g) = Eigen::Map<const Vec>(dw.data(), n3).transpose();
  return dW;
}

bool BeckeWeightDerivative::operator()(const Vec3 &r, double wg, int A,
                                       Mat3N &dw) const {
  const int natom = m_natom;
  dw.setZero(3, natom);
  if (natom < 2 || wg == 0.0)
    return false;
  const Mat3N &positions = m_positions;

  // Called once per grid point: per-thread scratch, no allocation here.
  // S(D, C) = s(C, D) and DS(D, C) = ds(C, D)/dmu_CD, column per cell.
  struct Scratch {
    Vec dist, ux, uy, uz, P;
    Vec dZx, dZy, dZz, dPx, dPy, dPz, cx, cy, cz;
    Mat S, DS;
    std::vector<char> full;
    void resize(int n) {
      if (dist.size() == n)
        return;
      for (Vec *v : {&dist, &ux, &uy, &uz, &P, &dZx, &dZy, &dZz, &dPx, &dPy,
                     &dPz, &cx, &cy, &cz})
        v->resize(n);
      S.resize(n, n);
      DS.resize(n, n);
      full.resize(n);
    }
  };
  thread_local Scratch t;
  t.resize(natom);
  for (int C = 0; C < natom; C++) {
    const Vec3 d = r - positions.col(C);
    const double dc = d.norm();
    t.dist(C) = dc;
    const double inv = dc > 0 ? 1.0 / dc : 0.0;
    t.ux(C) = d(0) * inv;
    t.uy(C) = d(1) * inv;
    t.uz(C) = d(2) * inv;
  }
  const double *dist = t.dist.data();

  // Cell functions s(C, D) and ds(C, D)/dmu_CD for ordered pairs, with
  // nu_CD = mu_CD + a_CD (1 - mu_CD^2), mu_CD = (r_C - r_D) / R_CD, and
  // P_C = prod_D s(C, D) (s(C, C) = 1).
  auto cell_row = [&](int C) {
    const double *ir = m_invR.col(C).data();
    const double *ac = m_aT.col(C).data();
    double *sc = t.S.col(C).data();
    double *dsc = t.DS.col(C).data();
    const double dC = dist[C];
    for (int D = 0; D < natom; D++) {
      const double mu = (dC - dist[D]) * ir[D];
      const double aa = ac[D];
      double x = mu + aa * (1.0 - mu * mu), d = 1.0;
      for (int k = 0; k < 3; k++) {
        d *= 1.5 * (1.0 - x * x);
        x = 0.5 * (3.0 - x * x) * x;
      }
      sc[D] = 0.5 * (1.0 - x);
      dsc[D] = -0.5 * d * (1.0 - 2.0 * aa * mu);
    }
    sc[C] = 1.0;
    dsc[C] = 0.0;
    double p = 1.0;
    for (int D = 0; D < natom; D++)
      p *= sc[D];
    t.P(C) = p;
    t.full[C] = 1;
  };
  cell_row(A);
  if (t.P(A) == 0.0)
    return false;

  // Only cells with P_C >= 1e-18 Z are needed (below). Since s <= 1 a
  // partial product only decreases, and Z >= every P, a cell whose product
  // over a few of its factors is already under 1e-18 max(P so far) is one
  // of those skipped; Z keeps that partial product (an upper bound below
  // 1e-18 Z). The factors tried are those of the atoms nearest the point
  // (the smallest s(C, D)), and cells are visited nearest first (the
  // largest P); the parent's cell is always complete.
  constexpr int nscreen = 6;
  // the nscreen + 1 atoms nearest the point, nearest first
  const int nnear = std::min(natom, nscreen + 1);
  int nearest[nscreen + 1];
  int nfound = 0;
  for (int C = 0; C < natom; C++) {
    if (nfound == nnear && dist[C] >= dist[nearest[nnear - 1]])
      continue;
    int k = (nfound < nnear) ? nfound++ : nnear - 1;
    while (k > 0 && dist[nearest[k - 1]] > dist[C]) {
      nearest[k] = nearest[k - 1];
      k--;
    }
    nearest[k] = C;
  }
  double pmax = t.P(A);
  auto visit = [&](int C) {
    const double skip = 1e-18 * pmax;
    double p = 1.0;
    int tried = 0;
    for (int k = 0; k < nnear && tried < nscreen; k++) {
      const int D = nearest[k];
      if (D == C)
        continue;
      tried++;
      const double mu = (dist[C] - dist[D]) * m_invR(D, C);
      const double nu = mu + m_aT(D, C) * (1.0 - mu * mu);
      double f, df;
      becke_f(nu, f, df);
      p *= 0.5 * (1.0 - f);
      if (p < skip)
        break;
    }
    if (p < skip) {
      t.P(C) = p;
      t.full[C] = 0;
    } else {
      cell_row(C);
      pmax = std::max(pmax, t.P(C));
    }
  };
  // nearest cells first (the largest P), then the rest
  for (int k = 0; k < nnear; k++)
    if (nearest[k] != A)
      visit(nearest[k]);
  for (int C = 0; C < natom; C++) {
    if (C == A)
      continue;
    bool done = false;
    for (int k = 0; k < nnear; k++)
      done |= (nearest[k] == C);
    if (!done)
      visit(C);
  }
  const double Z = t.P.sum();
  if (Z == 0.0)
    return false;
  // A cell's derivatives scale as P_C ds/s, and near saturation s ~
  // (1 - nu)^8 so ds/s ~ 8 / (1 - nu): for P_C < 1e-18 Z that is still
  // below 1e-12 of Z, so such cells are skipped (the parent never is).
  const double pcut = 1e-18 * Z;

  // dP_C/dR_B = sum_{D != C} (prod_{E != C, D} s_CE) ds_CD/dmu dmu_CD/dR_B,
  //   dmu/dR_C = -u_C / R_CD - mu (R_C - R_D) / R_CD^2
  //   dmu/dR_D =  u_D / R_CD + mu (R_C - R_D) / R_CD^2   (r fixed)
  t.dZx.setZero();
  t.dZy.setZero();
  t.dZz.setZero();
  t.dPx.setZero();
  t.dPy.setZero();
  t.dPz.setZero();
  const double *px = positions.data(); // 3 x n, column-major
  for (int C = 0; C < natom; C++) {
    if (!t.full[C])
      continue;
    const double *sc = t.S.col(C).data();
    const double *dsc = t.DS.col(C).data();
    int nzero = 0, zero_at = -1;
    for (int D = 0; D < natom; D++)
      if (sc[D] == 0.0) {
        nzero++;
        zero_at = D;
      }
    // A zero s_CD makes P_C zero while its own derivative term need not
    // be; keep those cells (rare: exactly saturated pairs).
    if (C != A && t.P(C) < pcut && nzero == 0)
      continue;
    if (nzero > 1)
      continue;
    double pnz = 1.0;
    if (nzero == 1) {
      for (int D = 0; D < natom; D++)
        if (D != zero_at)
          pnz *= sc[D];
    } else {
      pnz = t.P(C);
    }
    const double *ir = m_invR.col(C).data();
    const double xC = px[3 * C], yC = px[3 * C + 1], zC = px[3 * C + 2];
    const double uxC = t.ux(C), uyC = t.uy(C), uzC = t.uz(C);
    const double dC = dist[C];
    const bool parent = (C == A);
    for (int D = 0; D < natom; D++) {
      const double excl =
          (nzero == 1) ? ((D == zero_at) ? pnz : 0.0) : pnz / sc[D];
      const double c = excl * dsc[D]; // zero for D == C
      const double inv = ir[D], inv2 = inv * inv;
      const double mu = (dC - dist[D]) * inv;
      const double rx = xC - px[3 * D], ry = yC - px[3 * D + 1],
                   rz = zC - px[3 * D + 2];
      const double cm = c * mu * inv2, ci = c * inv;
      // c dmu_dD; c dmu_dC = -c (u_C / R) - c mu R_CD / R^2
      const double gx = t.ux(D) * ci + cm * rx;
      const double gy = t.uy(D) * ci + cm * ry;
      const double gz = t.uz(D) * ci + cm * rz;
      t.cx(D) = -uxC * ci - cm * rx;
      t.cy(D) = -uyC * ci - cm * ry;
      t.cz(D) = -uzC * ci - cm * rz;
      t.dZx(D) += gx;
      t.dZy(D) += gy;
      t.dZz(D) += gz;
      if (parent) {
        t.dPx(D) += gx;
        t.dPy(D) += gy;
        t.dPz(D) += gz;
      }
    }
    const double sx = t.cx.sum(), sy = t.cy.sum(), sz = t.cz.sum();
    t.dZx(C) += sx;
    t.dZy(C) += sy;
    t.dZz(C) += sz;
    if (parent) {
      t.dPx(C) += sx;
      t.dPy(C) += sy;
      t.dPz(C) += sz;
    }
  }
  // dw/dR_B = w (dP_A/dR_B / P_A - dZ/dR_B / Z) for B != A; the parent by
  // translational invariance (the point moves with A).
  const double fa = wg / t.P(A), fz = wg / Z;
  Vec3 total = Vec3::Zero();
  for (int B = 0; B < natom; B++) {
    if (B == A)
      continue;
    dw(0, B) = fa * t.dPx(B) - fz * t.dZx(B);
    dw(1, B) = fa * t.dPy(B) - fz * t.dZy(B);
    dw(2, B) = fa * t.dPz(B) - fz * t.dZz(B);
    total += dw.col(B);
  }
  dw.col(A) = -total;
  return true;
}

Mat becke_weight_hessian(
    const Mat3N &points, const Vec &weights,
    const std::vector<std::pair<size_t, size_t>> &atom_blocks, const Vec &e,
    const Mat3N &positions, const Vec &radii, bool treutler_ahlrichs) {
  const int natom = static_cast<int>(positions.cols());
  const int n3 = 3 * natom;
  Mat result = Mat::Zero(n3, n3);
  if (natom < 2)
    return result;
  const Mat R = calculate_interatomic_distances(positions);
  const Mat a = size_adjustment_matrix(radii, treutler_ahlrichs);
  const Mat3 I3 = Mat3::Identity();

  // Per point, with kappa_C = e w0 P_C (delta_CA / Z - P_A / Z^2), the
  // point-fixed second derivative of e w is
  //
  //   sum_C kappa_C (BS_C + gl_C gl_C^T)
  //     - e w0 (gPA gZ^T + gZ gPA^T) / Z^2 + 2 e w0 P_A gZ gZ^T / Z^3,
  //
  // where BS_C (the d2 ln s terms) touches only the 3x3 blocks of C with
  // each D. Those blocks are added directly; the rank-1 terms are queued as
  // column pairs and applied as one GEMM per batch, so no point builds a
  // dense 3N x 3N matrix.
  struct Scratch {
    Mat H, U, V;
    Eigen::Index ncol{0};
    Vec dist, P;
    Mat s;
    std::vector<Vec3> u, gC, gD;
    std::vector<Mat3> hCC, hCD, hDD;
    std::vector<Vec> gl;
    Vec gZ, gPA;
    // queue the rank-1 term scale * a b^T
    void push(const Vec &a, const Vec &b, double scale) {
      if (ncol == U.cols()) {
        H.noalias() += U * V.transpose();
        ncol = 0;
      }
      U.col(ncol) = a;
      V.col(ncol) = scale * b;
      ++ncol;
    }
    void flush() {
      if (ncol > 0)
        H.noalias() += U.leftCols(ncol) * V.leftCols(ncol).transpose();
      ncol = 0;
    }
  };
  constexpr Eigen::Index batch_cols = 512;

  for (int A = 0; A < natom; A++) {
    const auto [off, cnt] = atom_blocks[A];
    occ::parallel::thread_local_storage<Scratch> tls([n3, natom]() {
      Scratch t;
      t.H = Mat::Zero(n3, n3);
      t.U.resize(n3, batch_cols);
      t.V.resize(n3, batch_cols);
      t.dist.resize(natom);
      t.P.resize(natom);
      t.s.resize(natom, natom);
      t.u.resize(natom);
      t.gC.resize(natom * natom);
      t.gD.resize(natom * natom);
      t.hCC.resize(natom * natom);
      t.hCD.resize(natom * natom);
      t.hDD.resize(natom * natom);
      t.gl.assign(natom, Vec(n3));
      t.gZ.resize(n3);
      t.gPA.resize(n3);
      return t;
    });
    occ::parallel::parallel_for(size_t(0), cnt, [&](size_t k) {
      const size_t g = off + k;
      const double wg = weights(g), eg = e(g);
      if (wg == 0.0 || eg == 0.0)
        return;
      auto &t = tls.local();
      const Vec3 r = points.col(g);
      for (int C = 0; C < natom; C++) {
        Vec3 d = r - positions.col(C);
        t.dist(C) = d.norm();
        t.u[C] = t.dist(C) > 0 ? Vec3(d / t.dist(C)) : Vec3::Zero();
      }
      // s(C, D) for every pair (the cell functions need them all), then
      // the derivatives only for pairs whose first atom has a cell that
      // contributes (P_C >= 1e-18 Z, or the parent): typically a few atoms.
      t.s.setOnes();
      for (int C = 0; C < natom; C++)
        for (int D = 0; D < natom; D++) {
          if (C == D)
            continue;
          const double mu = (t.dist(C) - t.dist(D)) / R(C, D);
          const double nu = mu + a(C, D) * (1.0 - mu * mu);
          double f, df, d2f;
          becke_f2(nu, f, df, d2f);
          t.s(C, D) = 0.5 * (1.0 - f);
        }
      for (int C = 0; C < natom; C++) {
        double p = 1.0;
        for (int D = 0; D < natom; D++)
          if (D != C)
            p *= t.s(C, D);
        t.P(C) = p;
      }
      {
        const double Zs = t.P.sum();
        for (int C = 0; C < natom; C++) {
          if (t.P(C) == 0.0 || (C != A && t.P(C) < 1e-18 * Zs))
            continue;
          for (int D = 0; D < natom; D++) {
            if (C == D)
              continue;
            const double Rcd = R(C, D);
            const double N = t.dist(C) - t.dist(D);
            const double mu = N / Rcd;
            const Vec3 eCD = (positions.col(C) - positions.col(D)) / Rcd;
            const Vec3 dN_C = -t.u[C], dN_D = t.u[D];
            const Vec3 dR_C = eCD, dR_D = -eCD;
            const Mat3 d2N_CC =
                t.dist(C) > 0
                    ? Mat3((I3 - t.u[C] * t.u[C].transpose()) / t.dist(C))
                    : Mat3(Mat3::Zero());
            const Mat3 d2N_DD =
                t.dist(D) > 0
                    ? Mat3(-(I3 - t.u[D] * t.u[D].transpose()) / t.dist(D))
                    : Mat3(Mat3::Zero());
            const Mat3 Pe = (I3 - eCD * eCD.transpose()) / Rcd;
            const Mat3 d2R_CC = Pe, d2R_DD = Pe, d2R_CD = -Pe;
            const double R2 = Rcd * Rcd, R3 = R2 * Rcd;
            auto dmu = [&](const Vec3 &dN, const Vec3 &dR) -> Vec3 {
              return dN / Rcd - N * dR / R2;
            };
            auto d2mu = [&](const Mat3 &d2N, const Vec3 &dNx,
                            const Vec3 &dRx, const Vec3 &dNy,
                            const Vec3 &dRy, const Mat3 &d2R) -> Mat3 {
              return d2N / Rcd -
                     (dNx * dRy.transpose() + dRx * dNy.transpose()) / R2 -
                     N * d2R / R2 + 2.0 * N * dRx * dRy.transpose() / R3;
            };
            const Vec3 mC = dmu(dN_C, dR_C), mD = dmu(dN_D, dR_D);
            const Mat3 mCC = d2mu(d2N_CC, dN_C, dR_C, dN_C, dR_C, d2R_CC);
            const Mat3 mDD = d2mu(d2N_DD, dN_D, dR_D, dN_D, dR_D, d2R_DD);
            const Mat3 mCD =
                d2mu(Mat3::Zero(), dN_C, dR_C, dN_D, dR_D, d2R_CD);
            const double acd = a(C, D);
            const double nu = mu + acd * (1.0 - mu * mu);
            const double dnu = 1.0 - 2.0 * acd * mu;
            double f, df, d2f;
            becke_f2(nu, f, df, d2f);
            const double s1 = -0.5 * df, s2 = -0.5 * d2f;
            const Vec3 nC = dnu * mC, nD = dnu * mD;
            const Mat3 nCC = dnu * mCC - 2.0 * acd * mC * mC.transpose();
            const Mat3 nDD = dnu * mDD - 2.0 * acd * mD * mD.transpose();
            const Mat3 nCD = dnu * mCD - 2.0 * acd * mC * mD.transpose();
            const int k2 = C * natom + D;
            t.gC[k2] = s1 * nC;
            t.gD[k2] = s1 * nD;
            t.hCC[k2] = s2 * nC * nC.transpose() + s1 * nCC;
            t.hDD[k2] = s2 * nD * nD.transpose() + s1 * nDD;
            t.hCD[k2] = s2 * nC * nD.transpose() + s1 * nCD;
          }
        }
      }
      const double Z = t.P.sum();
      const double PA = t.P(A);
      if (PA <= 0.0 || Z <= 0.0)
        return;
      const double w0 = wg * Z / PA;
      // Cells with P_C < 1e-18 Z are skipped (see BeckeWeightDerivative:
      // their derivative terms stay below ~1e-12 of Z even though d2s/s
      // grows near saturation); exactly zero cells contribute nothing.
      const double pcut = 1e-18 * Z;
      Vec &gZ = t.gZ, &gPA = t.gPA;
      gZ.setZero();
      gPA.setZero();
      for (int C = 0; C < natom; C++) {
        if (t.P(C) == 0.0 || (C != A && t.P(C) < pcut))
          continue;
        const double kappa =
            eg * w0 * t.P(C) * ((C == A ? 1.0 / Z : 0.0) - PA / (Z * Z));
        Vec &gl = t.gl[C];
        gl.setZero();
        for (int D = 0; D < natom; D++) {
          if (D == C)
            continue;
          const int k2 = C * natom + D;
          const double sc = t.s(C, D);
          const Vec3 vC = t.gC[k2] / sc, vD = t.gD[k2] / sc;
          gl.segment<3>(3 * C) += vC;
          gl.segment<3>(3 * D) += vD;
          t.H.block<3, 3>(3 * C, 3 * C) +=
              kappa * (t.hCC[k2] / sc - vC * vC.transpose());
          t.H.block<3, 3>(3 * D, 3 * D) +=
              kappa * (t.hDD[k2] / sc - vD * vD.transpose());
          const Mat3 cd = kappa * (t.hCD[k2] / sc - vC * vD.transpose());
          t.H.block<3, 3>(3 * C, 3 * D) += cd;
          t.H.block<3, 3>(3 * D, 3 * C) += cd.transpose();
        }
        t.push(gl, gl, kappa);
        gZ += t.P(C) * gl;
        if (C == A)
          gPA = t.P(C) * gl;
      }
      const double c1 = -eg * w0 / (Z * Z);
      t.push(gPA, gZ, c1);
      t.push(gZ, gPA, c1);
      t.push(gZ, gZ, 2.0 * eg * w0 * PA / (Z * Z * Z));
    });
    Mat Hfix = Mat::Zero(n3, n3);
    for (auto &t : tls) {
      t.flush();
      Hfix += t.H;
    }
    // The points move with A: d_X = d_X|fixed - delta_{XA} sum_E d_E|fixed,
    // per Cartesian direction.
    Mat T = Mat::Identity(n3, n3);
    for (int E = 0; E < natom; E++)
      for (int x = 0; x < 3; x++)
        T(3 * A + x, 3 * E + x) -= 1.0;
    result += T * Hfix * T.transpose();
  }
  return result;
}

Mat3N becke_weight_gradient(
    const Mat3N &points, const Vec &weights,
    const std::vector<std::pair<size_t, size_t>> &atom_blocks, const Vec &e,
    const Mat3N &positions, const Vec &radii, bool treutler_ahlrichs) {
  const int natom = static_cast<int>(positions.cols());
  Mat3N result = Mat3N::Zero(3, natom);
  if (natom < 2)
    return result;
  const BeckeWeightDerivative dwdr(positions, radii, treutler_ahlrichs);
  occ::parallel::thread_local_storage<std::pair<Mat3N, Mat3N>> tls(
      [natom]() {
        return std::make_pair(Mat3N(Mat3N::Zero(3, natom)),
                              Mat3N(Mat3N::Zero(3, natom)));
      });
  for (int A = 0; A < natom; A++) {
    const auto [off, cnt] = atom_blocks[A];
    occ::parallel::parallel_for(size_t(0), cnt, [&](size_t k) {
      const size_t g = off + k;
      if (weights(g) == 0.0 || e(g) == 0.0)
        return;
      auto &[acc, dw] = tls.local();
      if (dwdr(points.col(g), weights(g), A, dw))
        acc += e(g) * dw;
    });
  }
  for (const auto &t : tls)
    result += t.first;
  return result;
}

Mat calculate_atomic_grid_weights(PartitionMethod method,
                                  const Mat &grid_points,
                                  const Mat &atomic_positions, const Vec &radii,
                                  const Mat &interatomic_distances) {

  switch (method.partition_function) {
  case PartitionFunction::StratmannScuseria: {
    if (method.treutler_alrichs_radii_adjustment) {
      return calculate_atomic_grid_weights_impl<TreutlerAlrichsAdjustment>(
          grid_points, atomic_positions, interatomic_distances, radii,
          stratmann_scuseria_partition);
    } else {
      return calculate_atomic_grid_weights_impl<BeckeAdjustment>(
          grid_points, atomic_positions, interatomic_distances, radii,
          stratmann_scuseria_partition);
    }
  }
  // Becke (default)
  default: {
    if (method.treutler_alrichs_radii_adjustment) {
      return calculate_atomic_grid_weights_impl<TreutlerAlrichsAdjustment>(
          grid_points, atomic_positions, interatomic_distances, radii,
          becke_partition);
    } else {
      return calculate_atomic_grid_weights_impl<BeckeAdjustment>(
          grid_points, atomic_positions, interatomic_distances, radii,
          becke_partition);
    }
  }
  }
}

IVec prune_nwchem_scheme(size_t nuclear_charge, size_t max_angular,
                         size_t num_radial, const Vec &radii) {

  std::array<int, 5> lebedev_level;
  IVec angular_grids(num_radial);

  if (max_angular < 50) {
    angular_grids.setConstant(max_angular);
    return angular_grids;
  } else if (max_angular == 50) {
    lebedev_level = {5, 6, 6, 6, 5};
  } else {
    size_t i;
    for (i = 6; i < lebedev_grid_levels.size(); i++) {
      if (lebedev_grid_levels[i] == max_angular)
        break;
    }
    lebedev_level[0] = 5;
    lebedev_level[1] = 6;
    lebedev_level[2] = i - 1;
    lebedev_level[3] = i;
    lebedev_level[4] = i - 1;
  }

  std::array<double, 4> alphas{0.1, 0.4, 0.8, 2.5};
  if (nuclear_charge <= 2) {
    alphas = {0.25, 0.5, 1.0, 4.5};
  } else if (nuclear_charge <= 10) {
    alphas = {0.16666667, 0.5, 0.9, 3.5};
  }

  double radius = get_atomic_radius(nuclear_charge);
  for (size_t i = 0; i < num_radial; i++) {
    double scale = radii(i) / radius;
    size_t place = std::distance(
        alphas.begin(), std::find_if(alphas.begin(), alphas.end(),
                                     [scale](double x) { return x > scale; }));
    angular_grids(i) = lebedev_grid_levels[lebedev_level[place]];
  }
  return angular_grids;
}

IVec prune_numgrid_scheme(size_t atomic_number, size_t max_angular,
                          size_t min_angular, const Vec &radii) {

  IVec result(radii.rows());
  double rb = get_atomic_radius(atomic_number) / 5.0;

  for (int i = 0; i < radii.rows(); i++) {
    double r = radii(i);
    size_t num_angular = max_angular;
    if (r < rb) {
      num_angular = static_cast<size_t>(max_angular * (r / rb));
      num_angular = nearest_grid_level_at_or_above(num_angular);
      if (num_angular < min_angular)
        num_angular = min_angular;
    }
    result(i) = num_angular;
  }
  return result;
}

IVec prune_orca_scheme(size_t atomic_number,
                       const std::array<size_t, 5> &region_grids,
                       const Vec &radii) {
  // ORCA divides radial grids into 5 regions based on scaled atomic radius
  // Region boundaries are approximately:
  // Region 1: r < 0.25 * R_bragg (core)
  // Region 2: 0.25 * R_bragg <= r < 0.50 * R_bragg (inner valence)
  // Region 3: 0.50 * R_bragg <= r < 1.00 * R_bragg (valence)
  // Region 4: 1.00 * R_bragg <= r < 2.50 * R_bragg (outer valence)
  // Region 5: r >= 2.50 * R_bragg (diffuse)
  //
  // These are approximate values - ORCA uses element-specific optimized cutoffs

  double R = get_atomic_radius(atomic_number);
  double r1 = 0.25 * R;
  double r2 = 0.50 * R;
  double r3 = 1.00 * R;
  double r4 = 2.50 * R;

  IVec result(radii.rows());

  for (int i = 0; i < radii.rows(); i++) {
    double r = radii(i);
    size_t region;
    if (r < r1) {
      region = 0; // Region 1
    } else if (r < r2) {
      region = 1; // Region 2
    } else if (r < r3) {
      region = 2; // Region 3
    } else if (r < r4) {
      region = 3; // Region 4
    } else {
      region = 4; // Region 5
    }
    result(i) = nearest_grid_level_at_or_above(region_grids[region]);
  }

  return result;
}

RadialGrid generate_gauss_chebyshev_radial_grid(size_t num_points) {
  RadialGrid result(num_points);
  result.points.setLinSpaced(num_points, 1, num_points * 2 - 1);
  result.points.array() *= std::numbers::pi_v<double> / (2 * num_points);
  result.points.array() = result.points.array().cos();
  result.weights.setConstant(std::numbers::pi_v<double> / num_points);
  return result;
}

RadialGrid generate_gauss_chebyshev_m3_radial_grid(size_t num_points,
                                                    size_t atomic_number) {
  // Generate base Gauss-Chebyshev quadrature on [-1, 1]
  RadialGrid result(num_points);

  // Get scaling parameter (xi) based on atomic radius
  // Treutler-Alrichs use xi = 0.5 * R_Bragg for the M3 mapping
  double xi = 0.5 * get_atomic_radius(atomic_number);
  const double ln2 = std::log(2.0);

  // Generate Gauss-Chebyshev second kind points: x_k = cos(k*pi/(n+1))
  // for k = 1, ..., n
  for (size_t k = 1; k <= num_points; k++) {
    double theta = std::numbers::pi_v<double> * k / (num_points + 1);
    double x = std::cos(theta);

    // M3 mapping: r = (xi / ln2) * ln(2 / (1-x))
    double r = (xi / ln2) * std::log(2.0 / (1.0 - x));

    // Weight transformation: w(r) = w(x) * dr/dx
    // For M3: dr/dx = xi / (ln2 * (1-x))
    // Gauss-Chebyshev second kind weight: w_k = pi/(n+1) * sin^2(theta)
    double sin_theta = std::sin(theta);
    double gc_weight = std::numbers::pi_v<double> / (num_points + 1) * sin_theta * sin_theta;
    double drdx = xi / (ln2 * (1.0 - x));
    double weight = gc_weight * drdx;

    result.points(k - 1) = r;
    result.weights(k - 1) = weight;
  }

  return result;
}

RadialGrid generate_euler_maclaurin_radial_grid(size_t num_points,
                                                double alpha) {
  RadialGrid result(num_points - 1);
  const int m_r = 2;
  const double f = m_r * num_points * std::pow(alpha, 3);

  for (int i = 1; i < num_points; i++) {

    const double n_minus_i = num_points - i;
    result.points(i - 1) = alpha * std::pow(i / n_minus_i, m_r);

    result.weights(i - 1) =
        f * std::pow(i, 3 * m_r - 1) / std::pow(n_minus_i, 3 * m_r + 1);
  }
  return result;
}

// Helper for LMG radial grid generation (TCA 106, 178 (2001), eq. 25)
double lmg_inner(const double max_error, const double alpha_inner) {
  int m = 0;
  double d = 1.9;

  double r = d - std::log(1.0 / max_error);
  r = r * 2.0 / (m + 3.0);
  r = std::exp(r) / (alpha_inner);
  r = std::sqrt(r);

  return r;
}

// Helper for LMG radial grid generation (TCA 106, 178 (2001), eq. 19)
double lmg_outer(const double max_error, const double alpha_outer, const int l,
                 const double guess) {

  int m = 2 * l;
  double r = guess;
  double r_old = 1.0e50;
  double c, a, e;
  double step = 0.5;
  double sign, sign_old;
  double f = 1.0e50;

  (f > max_error) ? (sign = 1.0) : (sign = -1.0);

  while (std::fabs(r_old - r) > 1e-14) {
    c = tgamma((m + 3.0) / 2.0);
    a = std::pow(alpha_outer * r * r, (m + 1.0) / 2.0);
    e = std::exp(-alpha_outer * r * r);
    f = c * a * e;

    sign_old = sign;
    (f > max_error) ? (sign = 1.0) : (sign = -1.0);
    if (r < 0.0)
      sign = 1.0;
    if (sign != sign_old)
      step *= 0.1;

    r_old = r;
    r += sign * step;
  }

  return r;
}

// Helper for LMG radial grid generation (TCA 106, 178 (2001), eqs. 17 and 18)
double lmg_h(const double max_error, const int l, const double guess) {
  int m = 2 * l;
  double h = guess;
  double h_old = 1.0e50;
  double step = 0.1 * guess;
  double sign, sign_old;
  double f = 1.0e50;
  double c0, cm, p0, e0, pm, rd0;

  (f > max_error) ? (sign = -1.0) : (sign = 1.0);

  while (std::fabs(h_old - h) > 1e-14) {
    c0 = 4.0 * std::sqrt(2.0) * std::numbers::pi_v<double>;
    cm = tgamma(3.0 / 2.0) / tgamma((m + 3.0) / 2.0);
    p0 = 1.0 / h;
    e0 = std::exp(-std::numbers::pi_v<double> * std::numbers::pi_v<double> / (2.0 * h));
    pm = std::pow(std::numbers::pi_v<double> / h, m / 2.0);
    rd0 = c0 * p0 * e0;
    f = cm * pm * rd0;

    sign_old = sign;
    (f > max_error) ? (sign = -1.0) : (sign = 1.0);
    if (h < 0.0)
      sign = 1.0;
    if (sign != sign_old)
      step *= 0.1;

    h_old = h;
    h += sign * step;
  }

  return h;
}

RadialGrid generate_lmg_radial_grid(size_t atomic_number,
                                    double radial_precision, double alpha_max,
                                    int l_max, const occ::Vec &alpha_min) {

  double r_inner = lmg_inner(radial_precision, 2 * alpha_max);
  double h = std::numeric_limits<float>::max();
  double r_outer = 0.0;
  double br = get_atomic_radius(atomic_number);

  for (int i = 0; i <= l_max; i++) {
    if (alpha_min[i] > 0.0) {
      r_outer = std::max(
          r_outer, lmg_outer(radial_precision, alpha_min(i), i, 4.0 * br));

      assert(r_outer > r_inner);

      h = std::min(h, lmg_h(radial_precision, i, 0.1 * (r_outer - r_inner)));
    }
  }

  occ::log::debug("LMG grid r_inner = {}, r_outer = {}, h = {}", r_inner,
                  r_outer, h);

  assert(r_outer > h);
  double c = r_inner / (std::exp(h) - 1.0);
  size_t num_radial = static_cast<int>(std::log(1.0 + (r_outer / c)) / h);

  RadialGrid result(num_radial);
  for (int i = 0; i < num_radial; i++) {
    double r = c * (std::exp((i + 1) * h) - 1.0);
    result.points(i) = r;
    result.weights(i) = (result.points(i) + c) * h * r * r;
  }

  return result;
}

// Mura-Knowles [JCP 104, 9848 (1996) - doi:10.1063/1.471749] log3 quadrature
RadialGrid generate_mura_knowles_radial_grid(size_t num_points, size_t charge) {
  RadialGrid result(num_points);
  double far_bound = 5.2;

  // Adjust far_bound parameter based on atomic number
  switch (charge) {
  case 3:
  case 4:
  case 11:
  case 12:
  case 19:
  case 20:
    far_bound = 7;
    break;
  }

  for (size_t i = 0; i < num_points; i++) {
    double x = (i + 0.5) / num_points;
    double x2 = x * x;
    double x3 = x2 * x;
    result.points(i) = -far_bound * std::log(1 - x3);
    result.weights(i) = far_bound * 3 * x2 / ((1 - x3) * num_points);
  }

  return result;
}

// Becke [JCP 88, 2547 (1988) - doi:10.1063/1.454033] quadrature
RadialGrid generate_becke_radial_grid(size_t num_points, double rm) {
  RadialGrid result = generate_gauss_chebyshev_radial_grid(num_points);

  for (size_t i = 0; i < num_points; i++) {
    double tp1 = 1 + result.points(i);
    double tm1 = 1 - result.points(i);
    result.points(i) = (tp1 / tm1) * rm;
    result.weights(i) *= 2 / (tm1 * tm1) * rm;
  }

  return result;
}

// Treutler-Alrichs [JCP 102, 346 (1995) - doi:10.1063/1.469408] M4 quadrature
RadialGrid generate_treutler_alrichs_radial_grid(size_t num_points) {
  RadialGrid result(num_points);
  double step = std::numbers::pi_v<double> / (num_points + 1);
  double ln2 = 1 / std::log(2);
  for (size_t i = 1; i <= num_points; i++) {
    double x = cos(i * step);
    double tmp1 = ln2 * pow((1 + x), 0.6);
    double tmp2 = std::log((1 - x) / 2);
    result.points(num_points - i) = -tmp1 * tmp2;
    result.weights(num_points - i) =
        step * sin(i * step) * tmp1 * (-0.6 / (1 + x) * tmp2 + 1 / (1 - x));
  }
  return result;
}

RadialGrid generate_perez_jorda_m3_radial_grid(size_t num_points, double xi) {
  RadialGrid result(num_points);
  const double pi = std::numbers::pi_v<double>;
  const double n1 = static_cast<double>(num_points + 1);
  const double scale = xi / std::log(2.0);
  for (size_t i = 1; i <= num_points; i++) {
    const double t = i * pi / n1;
    const double st = std::sin(t), ct = std::cos(t);
    const double x =
        (n1 - 2.0 * i) / n1 + 2.0 / pi * (1.0 + 2.0 / 3.0 * st * st) * ct * st;
    const double wx = 16.0 / (3.0 * n1) * st * st * st * st;
    const double r = scale * std::log(2.0 / (1.0 - x));
    const double drdx = scale / (1.0 - x);
    // x decreases with i, so r does too: store innermost first.
    result.points(num_points - i) = r;
    result.weights(num_points - i) = wx * drdx * r * r;
  }
  return result;
}

namespace {

// ORCA 6.1 COSX atomic grid from the measured tables (no partitioning).
AtomGrid generate_orca_cosx_atom_grid(size_t atomic_number,
                                      const OrcaCOSXAtomicGrid &spec) {
  const RadialGrid radial =
      generate_perez_jorda_m3_radial_grid(spec.num_radial, spec.xi);
  size_t total = 0;
  for (const auto &[order, count] : *spec.angular_runs)
    total += static_cast<size_t>(order) * count;
  AtomGrid result(total);
  size_t shell = 0, offset = 0;
  for (const auto &[order, count] : *spec.angular_runs) {
    const auto lebedev = grid::lebedev(order);
    const Eigen::Index n = lebedev.rows();
    for (int k = 0; k < count; k++, shell++) {
      const double r = radial.points(shell);
      const double w = 4 * std::numbers::pi_v<double> * radial.weights(shell);
      result.points.block(0, offset, 3, n) = lebedev.leftCols(3).transpose() * r;
      result.weights.segment(offset, n) = lebedev.col(3) * w;
      offset += n;
    }
  }
  result.atomic_number = atomic_number;
  return result;
}

} // namespace

AtomGrid generate_atom_grid(size_t atomic_number, const GridSettings &settings,
                            RadialGridMethod method, double alpha_max,
                            int l_max, const Vec &alpha_min) {

  if (settings.orca_cosx_level > 0) {
    const auto spec = orca_cosx_atomic_grid(static_cast<int>(atomic_number),
                                            settings.orca_cosx_level);
    if (spec.num_radial > 0) {
      occ::timing::start(occ::timing::category::grid_init);
      AtomGrid result = generate_orca_cosx_atom_grid(atomic_number, spec);
      occ::log::debug("ORCA GRIDX {} grid for Z={}: {} shells, xi={}, {} points",
                      settings.orca_cosx_level, atomic_number, spec.num_radial,
                      spec.xi, result.num_points());
      occ::timing::stop(occ::timing::category::grid_init);
      return result;
    }
  }

  // Start timer
  occ::timing::start(occ::timing::category::grid_init);

  // Create the appropriate radial grid based on the method
  RadialGrid radial;

  // Determine number of radial points
  size_t n_radial;
  if (settings.int_acc > 0) {
    // ORCA-style IntAcc-based radial point count
    n_radial = occ::numint::calculate_radial_points_orca(settings.int_acc, atomic_number);
    occ::log::debug("IntAcc {:.3f}: {} radial points for Z={}",
                    settings.int_acc, n_radial, atomic_number);
  } else {
    n_radial = settings.radial_points > 0 ? settings.radial_points : 50;
  }

  // If using ORCA-style COSX grids, use Gauss-Chebyshev with M3 mapping
  bool use_cosx_radial = settings.has_angular_regions();

  if (use_cosx_radial) {
    // ORCA COSX uses Treutler-Alrichs M3 mapping with Gauss-Chebyshev
    // Use the existing implementation and apply atomic scaling
    radial = generate_treutler_alrichs_radial_grid(n_radial);
    // Scale by atomic radius (Bragg radius)
    double rm = get_atomic_radius(atomic_number);
    radial.points.array() *= rm;
    radial.weights.array() *= rm;
    // Apply r^2 factor for spherical integration
    radial.weights.array() *= radial.points.array() * radial.points.array();
  } else {
    // Standard method selection
    switch (method) {
  case RadialGridMethod::LMG: {
    // Use default values if basis information isn't provided
    if (alpha_max <= 0.0 || l_max <= 0 || alpha_min.size() == 0) {
      alpha_max = 20.0; // Default reasonable value
      l_max = 3;        // Default reasonable value
      Vec default_alpha_min = Vec::Constant(l_max + 1, 0.1);

      radial =
          generate_lmg_radial_grid(atomic_number, settings.radial_precision,
                                   alpha_max, l_max, default_alpha_min);
    } else {
      radial =
          generate_lmg_radial_grid(atomic_number, settings.radial_precision,
                                   alpha_max, l_max, alpha_min);
    }
    break;
  }

  case RadialGridMethod::TreutlerAlrichs: {
    radial = generate_treutler_alrichs_radial_grid(n_radial);
    break;
  }

  case RadialGridMethod::MuraKnowles: {
    radial = generate_mura_knowles_radial_grid(n_radial, atomic_number);
    break;
  }

  case RadialGridMethod::Becke: {
    double rm = get_atomic_radius(atomic_number);
    radial = generate_becke_radial_grid(n_radial, rm);
    radial.weights.array() *= radial.points.array() * radial.points.array();
    break;
  }

  case RadialGridMethod::GaussChebyshev: {
    radial = generate_gauss_chebyshev_radial_grid(n_radial);

    // Transform to make suitable for atoms
    double rm = get_atomic_radius(atomic_number);
    for (size_t i = 0; i < n_radial; i++) {
      double tp1 = 1 + radial.points(i);
      double tm1 = 1 - radial.points(i);
      radial.points(i) = (tp1 / tm1) * rm;
      radial.weights(i) *= 2 / (tm1 * tm1) * rm;
    }
    radial.weights.array() *= radial.points.array() * radial.points.array();
    break;
  }

  case RadialGridMethod::EulerMaclaurin: {
    double alpha = 2 * get_atomic_radius(atomic_number);
    radial = generate_euler_maclaurin_radial_grid(n_radial, alpha);
    break;
  }

  default: {
    occ::log::warn("Unsupported grid method, falling back to LMG");
    // Recursive call with LMG method
    return generate_atom_grid(atomic_number, settings, RadialGridMethod::LMG,
                              alpha_max, l_max, alpha_min);
  }
    } // end switch
  } // end else (non-COSX path)

  // Apply 4π factor to weights for angular integration
  radial.weights.array() *= 4 * std::numbers::pi_v<double>;

  // Determine number of angular points, potentially with reduced grid for H, He
  int max_angular = settings.max_angular_points;
  if (settings.reduced_first_row_element_grid && atomic_number < 3) {
    max_angular = nearest_grid_level_below(max_angular);
    occ::log::debug("Reduced grid size for element {} = {}", atomic_number,
                    max_angular);
  }

  // Apply pruning scheme to determine angular points for each radial shell
  IVec n_angular;

  // Check for ORCA-style angular regions first
  if (settings.has_angular_regions()) {
    // Use ORCA 5-region pruning with the provided angular grids
    // Apply reduced grid for H/He by shifting to lower Lebedev levels
    std::array<size_t, 5> regions = settings.angular_regions;
    if (settings.reduced_first_row_element_grid && atomic_number < 3) {
      for (auto &r : regions) {
        r = nearest_grid_level_below(r);
      }
    }
    n_angular = prune_orca_scheme(atomic_number, regions, radial.points);
    occ::log::debug("ORCA pruning for Z={}: regions [{},{},{},{},{}]",
                    atomic_number, regions[0], regions[1], regions[2],
                    regions[3], regions[4]);
  } else {
    // Standard pruning schemes
    switch (settings.pruning_scheme) {
    case PruningScheme::NWChem:
      n_angular = prune_nwchem_scheme(atomic_number, max_angular,
                                      radial.num_points(), radial.points);
      break;

    case PruningScheme::NumGrid:
      n_angular = prune_numgrid_scheme(
          atomic_number, max_angular, settings.min_angular_points, radial.points);
      break;

    case PruningScheme::None:
    default:
      n_angular = IVec::Constant(radial.num_points(), max_angular);
      break;
    }
  }

  // Count total angular points
  size_t total_points = 0;
  for (size_t i = 0; i < radial.num_points(); i++) {
    total_points += angular_point_count(n_angular(i));
  }

  // Create the atom grid
  AtomGrid result(total_points);

  // Add points from each radial shell
  size_t point_offset = 0;
  for (size_t i = 0; i < radial.num_points(); i++) {
    auto lebedev = grid::lebedev(n_angular(i));
    double r = radial.points(i);
    double w = radial.weights(i);

    size_t n_angular_points = lebedev.rows();

    result.points.block(0, point_offset, 3, n_angular_points) =
        lebedev.leftCols(3).transpose() * r;
    result.weights.segment(point_offset, n_angular_points) = lebedev.col(3) * w;

    point_offset += n_angular_points;
  }

  result.atomic_number = atomic_number;

  occ::log::debug("{} total grid points for element {} ({} radial)",
                  result.num_points(), atomic_number, radial.num_points());

  occ::timing::stop(occ::timing::category::grid_init);
  return result;
}
} // namespace occ::numint
