#include <Eigen/Geometry>
#include <Eigen/QR>
#include <Eigen/LU>
#include <array>
#include <cmath>
#include <occ/core/eeq.h>
#include <occ/core/units.h>

namespace occ::core::charges {
namespace impl {
constexpr int max_elem{87};
// electronegativity
inline const std::array<double, max_elem> chi{
    -1.0,       1.23695041, 1.26590957, 0.54341808, 0.99666991, 1.26691604,
    1.40028282, 1.55819364, 1.56866440, 1.57540015, 1.15056627, 0.55936220,
    0.72373742, 1.12910844, 1.12306840, 1.52672442, 1.40768172, 1.48154584,
    1.31062963, 0.40374140, 0.75442607, 0.76482096, 0.98457281, 0.96702598,
    1.05266584, 0.93274875, 1.04025281, 0.92738624, 1.07419210, 1.07900668,
    1.04712861, 1.15018618, 1.15388455, 1.36313743, 1.36485106, 1.39801837,
    1.18695346, 0.36273870, 0.58797255, 0.71961946, 0.96158233, 0.89585296,
    0.81360499, 1.00794665, 0.92613682, 1.09152285, 1.14907070, 1.13508911,
    1.08853785, 1.11005982, 1.12452195, 1.21642129, 1.36507125, 1.40340000,
    1.16653482, 0.34125098, 0.58884173, 0.68441115, 0.56999999, 0.56999999,
    0.56999999, 0.56999999, 0.56999999, 0.56999999, 0.56999999, 0.56999999,
    0.56999999, 0.56999999, 0.56999999, 0.56999999, 0.56999999, 0.56999999,
    0.87936784, 1.02761808, 0.93297476, 1.10172128, 0.97350071, 1.16695666,
    1.23997927, 1.18464453, 1.14191734, 1.12334192, 1.01485321, 1.12950808,
    1.30804834, 1.33689961, 1.27465977};
// hardness
inline const std::array<double, max_elem> eta{
    -1.0,        -0.35015861, 1.04121227,  0.09281243,  0.09412380,
    0.26629137,  0.19408787,  0.05317918,  0.03151644,  0.32275132,
    1.30996037,  0.24206510,  0.04147733,  0.11634126,  0.13155266,
    0.15350650,  0.15250997,  0.17523529,  0.28774450,  0.42937314,
    0.01896455,  0.07179178,  -0.01121381, -0.03093370, 0.02716319,
    -0.01843812, -0.15270393, -0.09192645, -0.13418723, -0.09861139,
    0.18338109,  0.08299615,  0.11370033,  0.19005278,  0.10980677,
    0.12327841,  0.25345554,  0.58615231,  0.16093861,  0.04548530,
    -0.02478645, 0.01909943,  0.01402541,  -0.03595279, 0.01137752,
    -0.03697213, 0.08009416,  0.02274892,  0.12801822,  -0.02078702,
    0.05284319,  0.07581190,  0.09663758,  0.09547417,  0.07803344,
    0.64913257,  0.15348654,  0.05054344,  0.11000000,  0.11000000,
    0.11000000,  0.11000000,  0.11000000,  0.11000000,  0.11000000,
    0.11000000,  0.11000000,  0.11000000,  0.11000000,  0.11000000,
    0.11000000,  0.11000000,  -0.02786741, 0.01057858,  -0.03892226,
    -0.04574364, -0.03874080, -0.03782372, -0.07046855, 0.09546597,
    0.21953269,  0.02522348,  0.15263050,  0.08042611,  0.01878626,
    0.08715453,  0.10500484};

// CN scaling constant
inline const std::array<double, max_elem> kcn{
    -1.0,        0.04916110,  0.10937243,  -0.12349591, -0.02665108,
    -0.02631658, 0.06005196,  0.09279548,  0.11689703,  0.15704746,
    0.07987901,  -0.10002962, -0.07712863, -0.02170561, -0.04964052,
    0.14250599,  0.07126660,  0.13682750,  0.14877121,  -0.10219289,
    -0.08979338, -0.08273597, -0.01754829, -0.02765460, -0.02558926,
    -0.08010286, -0.04163215, -0.09369631, -0.03774117, -0.05759708,
    0.02431998,  -0.01056270, -0.02692862, 0.07657769,  0.06561608,
    0.08006749,  0.14139200,  -0.05351029, -0.06701705, -0.07377246,
    -0.02927768, -0.03867291, -0.06929825, -0.04485293, -0.04800824,
    -0.01484022, 0.07917502,  0.06619243,  0.02434095,  -0.01505548,
    -0.03030768, 0.01418235,  0.08953411,  0.08967527,  0.07277771,
    -0.02129476, -0.06188828, -0.06568203, -0.11000000, -0.11000000,
    -0.11000000, -0.11000000, -0.11000000, -0.11000000, -0.11000000,
    -0.11000000, -0.11000000, -0.11000000, -0.11000000, -0.11000000,
    -0.11000000, -0.11000000, -0.03585873, -0.03132400, -0.05902379,
    -0.02827592, -0.07606260, -0.02123839, 0.03814822,  0.02146834,
    0.01580538,  -0.00894298, -0.05864876, -0.01817842, 0.07721851,
    0.07936083,  0.05849285};

// charge widths
inline const std::array<double, max_elem> width{
    -1.0,       0.55159092, 0.66205886, 0.90529132, 1.51710827, 2.86070364,
    1.88862966, 1.32250290, 1.23166285, 1.77503721, 1.11955204, 1.28263182,
    1.22344336, 1.70936266, 1.54075036, 1.38200579, 2.18849322, 1.36779065,
    1.27039703, 1.64466502, 1.58859404, 1.65357953, 1.50021521, 1.30104175,
    1.46301827, 1.32928147, 1.02766713, 1.02291377, 0.94343886, 1.14881311,
    1.47080755, 1.76901636, 1.98724061, 2.41244711, 2.26739524, 2.95378999,
    1.20807752, 1.65941046, 1.62733880, 1.61344972, 1.63220728, 1.60899928,
    1.43501286, 1.54559205, 1.32663678, 1.37644152, 1.36051851, 1.23395526,
    1.65734544, 1.53895240, 1.97542736, 1.97636542, 2.05432381, 3.80138135,
    1.43893803, 1.75505957, 1.59815118, 1.76401732, 1.63999999, 1.63999999,
    1.63999999, 1.63999999, 1.63999999, 1.63999999, 1.63999999, 1.63999999,
    1.63999999, 1.63999999, 1.63999999, 1.63999999, 1.63999999, 1.63999999,
    1.47055223, 1.81127084, 1.40189963, 1.54015481, 1.33721475, 1.57165422,
    1.04815857, 1.78342098, 2.79106396, 1.78160840, 2.47588882, 2.37670734,
    1.76613217, 2.66172302, 2.82773085};

// Covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009,
// 188-197) values for metals decreased by 10%
inline const std::array<double, 119> covalent{
    -1.0, 0.32, 0.46, 1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67, 1.40,
    1.25, 1.13, 1.04, 1.10, 1.02, 0.99, 0.96, 1.76, 1.54, 1.33, 1.22, 1.21,
    1.10, 1.07, 1.04, 1.00, 0.99, 1.01, 1.09, 1.12, 1.09, 1.15, 1.10, 1.14,
    1.17, 1.89, 1.67, 1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15,
    1.23, 1.28, 1.26, 1.26, 1.23, 1.32, 1.31, 2.09, 1.76, 1.62, 1.47, 1.58,
    1.57, 1.56, 1.55, 1.51, 1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53, 1.46,
    1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32, 1.30, 1.30, 1.36,
    1.31, 1.38, 1.42, 2.01, 1.81, 1.67, 1.58, 1.52, 1.53, 1.54, 1.55, 1.49,
    1.49, 1.51, 1.51, 1.48, 1.50, 1.56, 1.58, 1.45, 1.41, 1.34, 1.29, 1.27,
    1.21, 1.16, 1.15, 1.09, 1.22, 1.36, 1.43, 1.46, 1.58, 1.48, 1.57};

} // namespace impl

Vec eeq_coordination_numbers(const IVec &atomic_numbers,
                             const Mat3N &positions) {
  const int N = atomic_numbers.rows();
  constexpr double kcn = 7.5;
  constexpr double D3_SCALE_FACTOR = 4.0 / 3.0;

  Vec cn = Vec::Zero(N);
  constexpr double cutoff = 25.0;
  const double cutoff2 = cutoff * cutoff;

  for (int i = 0; i < N; i++) {
    const int ni = atomic_numbers(i);
    for (int j = 0; j < i; j++) {
      const int nj = atomic_numbers(j);

      const double r2 = (positions.col(i) - positions.col(j)).squaredNorm();
      if (r2 > cutoff2)
        continue;
      const double r = std::sqrt(r2);

      const double rc =
          (impl::covalent[ni] + impl::covalent[nj]) * D3_SCALE_FACTOR;
      double count = 0.5 * (1.0 + std::erf(-kcn * (r - rc) / rc));
      cn(i) += count;
      if (i != j)
        cn(j) += count;
    }
  }
  return cn;
}

Mat build_A_matrix(const IVec &atomic_numbers, const Mat3N &positions) {
  const int N = atomic_numbers.rows();
  const double sqrt_pi_fac = std::sqrt(2.0 / M_PI);
  Mat A = Mat::Zero(N + 1, N + 1);
  for (int i = 0; i < N; i++) {
    const int ni = atomic_numbers(i);
    const double ri = impl::width[ni];
    Vec3 pos_i = positions.col(i) * occ::units::ANGSTROM_TO_BOHR;
    for (int j = 0; j < i; j++) {
      const int nj = atomic_numbers(j);
      Vec3 pos_j = positions.col(j) * occ::units::ANGSTROM_TO_BOHR;
      double rj = impl::width[nj];
      double r2 = (pos_i - pos_j).squaredNorm();
      double gamma = 1.0 / (ri * ri + rj * rj);
      double value = std::erf(std::sqrt(r2 * gamma)) / std::sqrt(r2);
      A(j, i) += value;
      A(i, j) += value;
    }
    double value = impl::eta[ni] + sqrt_pi_fac / impl::width[ni];
    A(i, i) += value;
  }

  A.row(N).array() = 1.0;
  A.col(N).array() = 1.0;
  A(N, N) = 0.0;
  return A;
}

Vec build_X_vector(const IVec &atomic_numbers, const Vec &cn,
                   double charge = 0.0) {
  const int N = atomic_numbers.rows();
  constexpr double eps = 1e-14; // avoid singularity with 0
  Vec X(N + 1);
  for (int i = 0; i < N; i++) {
    const int ni = atomic_numbers(i);
    double tmp = impl::kcn[ni] / std::sqrt(cn(i) + eps);
    X(i) = -impl::chi[ni] + tmp * cn(i);
  }
  X(N) = charge;
  return X;
}

Vec eeq_partial_charges(const IVec &atomic_numbers, const Mat3N &positions,
                        double charge) {

  auto cn = eeq_coordination_numbers(atomic_numbers, positions);
  size_t N = atomic_numbers.rows();

  Mat A = build_A_matrix(atomic_numbers, positions);
  Vec X = build_X_vector(atomic_numbers, cn, charge);

  return A.householderQr().solve(X).topRows(N);
}

// ============================================================================
// Periodic (3D) EEQ — Ewald-summed A matrix + lattice-summed CN.
// Mirrors multicharge/tblite's `get_amat_3d`. The same Ewald split is used in
// our periodic γ matrix (`occ::xtb::periodic_klopman_ohno_gamma`) — the only
// difference is the short-range kernel (erf vs Klopman-Ohno).
// ============================================================================

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrtPi = 1.7724538509055160273;

// Perpendicular distance from origin to the face spanned by (aj, ak).
inline double perp_distance(const Vec3 &ai, const Vec3 &aj, const Vec3 &ak) {
  const double V = std::abs(ai.dot(aj.cross(ak)));
  const double face = aj.cross(ak).norm();
  return V / face;
}

// Triple-loop bounds for a direct-lattice cutoff: integer indices in each
// lattice direction that suffice to enumerate every translation within
// `cutoff` of the origin. Stored as a small POD so the enumerator can
// be reused without recomputing per pair.
struct LatticeBounds {
  Vec3 a, b, c;
  int na, nb, nc;
  double cutoff;
  double cutoff_plus_slack; // tight reject inside the loop
};

inline LatticeBounds compute_lattice_bounds(const Mat3 &lattice_bohr,
                                             double cutoff_bohr) {
  LatticeBounds lb;
  lb.a = lattice_bohr.col(0);
  lb.b = lattice_bohr.col(1);
  lb.c = lattice_bohr.col(2);
  const double da = perp_distance(lb.a, lb.b, lb.c);
  const double db = perp_distance(lb.b, lb.a, lb.c);
  const double dc = perp_distance(lb.c, lb.a, lb.b);
  lb.na = static_cast<int>(std::ceil(cutoff_bohr / da));
  lb.nb = static_cast<int>(std::ceil(cutoff_bohr / db));
  lb.nc = static_cast<int>(std::ceil(cutoff_bohr / dc));
  lb.cutoff = cutoff_bohr;
  lb.cutoff_plus_slack = cutoff_bohr + std::max({da, db, dc});
  return lb;
}

// Inline-able iterator over direct-lattice translations within `lb.cutoff`.
// `f(const Vec3 &T)` is invoked for each translation including T=0; the
// only allocation is the caller's accumulator.
template <typename F>
inline void for_each_lattice_translation(const LatticeBounds &lb, F &&f) {
  const double slack = lb.cutoff_plus_slack;
  for (int i = -lb.na; i <= lb.na; ++i) {
    for (int j = -lb.nb; j <= lb.nb; ++j) {
      for (int k = -lb.nc; k <= lb.nc; ++k) {
        const Vec3 t = i * lb.a + j * lb.b + k * lb.c;
        if (t.norm() - 1e-12 > slack) continue;
        f(t);
      }
    }
  }
}

// Inline-able iterator over G with 0 < |G| ≤ recip_cutoff. `f(const Vec3 &G,
// double precomputed_coeff)` runs once per qualifying G; `precomputed_coeff`
// is the (4π/V)·exp(−G²/4α²)/G² factor that appears in the reciprocal Ewald
// sum (saves a per-call exp() in tight inner loops).
template <typename F>
inline void for_each_g_vector(const Mat3 &reciprocal_bohr,
                               double recip_cutoff, double four_pi_over_V,
                               double inv4a2, F &&f) {
  const Vec3 b1 = reciprocal_bohr.col(0);
  const Vec3 b2 = reciprocal_bohr.col(1);
  const Vec3 b3 = reciprocal_bohr.col(2);
  auto bound = [&](const Vec3 &v) {
    return static_cast<int>(std::ceil(recip_cutoff / v.norm())) + 1;
  };
  const int n1 = bound(b1);
  const int n2 = bound(b2);
  const int n3 = bound(b3);
  const double cutoff2 = recip_cutoff * recip_cutoff;
  for (int i = -n1; i <= n1; ++i) {
    for (int j = -n2; j <= n2; ++j) {
      for (int k = -n3; k <= n3; ++k) {
        const Vec3 G = i * b1 + j * b2 + k * b3;
        const double g2 = G.squaredNorm();
        if (g2 < 1e-20 || g2 > cutoff2) continue;
        const double coeff = four_pi_over_V * std::exp(-g2 * inv4a2) / g2;
        f(G, coeff);
      }
    }
  }
}

// Pair Ewald sum at vector R between Gaussian charges with combined width
// γ_ij = 1/√(w_i² + w_j²). Returns the A_ij contribution
// (direct + reciprocal + background, plus self if `same_pair`).
inline double eeq_pair_ewald(const Vec3 &R, double gamma_pair,
                              const EeqEwaldData &d, bool same_pair) {
  const double alpha = d.alpha;
  // Direct: Σ_T [erf(γ·r) − erf(α·r)] / r
  double s_dir = 0.0;
  const LatticeBounds lb =
      compute_lattice_bounds(d.lattice_bohr, d.erfc_cutoff);
  const double erfc_cutoff = d.erfc_cutoff;
  for_each_lattice_translation(lb, [&](const Vec3 &T) {
    const Vec3 dR = R + T;
    const double r2 = dR.squaredNorm();
    if (r2 < 1e-20) return; // self position handled by self_term
    const double r = std::sqrt(r2);
    if (r > erfc_cutoff) return;
    s_dir += (std::erf(gamma_pair * r) - std::erf(alpha * r)) / r;
  });
  // Reciprocal: (4π/V) Σ_{G≠0} (1/G²) exp(−G²/4α²) cos(G·R)
  double s_rec = 0.0;
  for_each_g_vector(d.reciprocal_bohr, d.recip_cutoff, d.four_pi_over_V,
                     d.inv4a2,
                     [&](const Vec3 &G, double coeff) {
                       s_rec += coeff * std::cos(G.dot(R));
                     });
  double s = s_dir + s_rec + d.background;
  if (same_pair) s += d.self_term;
  return s;
}

} // namespace

EeqEwaldData build_eeq_ewald_data(const Mat3 &lattice_bohr, double tol,
                                  double alpha_user) {
  const Vec3 a = lattice_bohr.col(0);
  const Vec3 b = lattice_bohr.col(1);
  const Vec3 c = lattice_bohr.col(2);
  const double V = std::abs(a.dot(b.cross(c)));
  const double alpha = (alpha_user > 0.0) ? alpha_user
                                          : kSqrtPi / std::cbrt(V);
  // erfc(x) ≤ exp(-x²)/(x√π); set x = sqrt(-ln tol).
  const double x = std::sqrt(-std::log(tol));
  EeqEwaldData d;
  d.lattice_bohr = lattice_bohr;
  d.reciprocal_bohr = 2.0 * kPi * lattice_bohr.inverse().transpose();
  d.alpha = alpha;
  d.erfc_cutoff = x / alpha + 1.0;     // Bohr, +1 safety
  d.recip_cutoff = 2.0 * alpha * x;    // 1/Bohr
  d.four_pi_over_V = 4.0 * kPi / V;
  d.inv4a2 = 1.0 / (4.0 * alpha * alpha);
  d.background = -kPi / (V * alpha * alpha);
  d.self_term = -2.0 * alpha / kSqrtPi;
  return d;
}

Vec eeq_coordination_numbers_periodic(const IVec &atomic_numbers,
                                      const Mat3N &positions_angstrom,
                                      const Mat3 &lattice_angstrom,
                                      double cutoff_angstrom) {
  using occ::units::ANGSTROM_TO_BOHR;
  const int N = atomic_numbers.rows();
  constexpr double kcn_steepness = 7.5;
  constexpr double D3_SCALE = 4.0 / 3.0;
  // CN kernel works in Å (matches the molecular version): rc and r both in Å.
  // Bounds on the lattice translation enumeration are computed once in Å.
  const LatticeBounds lb =
      compute_lattice_bounds(lattice_angstrom, cutoff_angstrom);
  const double cutoff2 = cutoff_angstrom * cutoff_angstrom;
  Vec cn = Vec::Zero(N);
  for (int i = 0; i < N; ++i) {
    const int ni = atomic_numbers(i);
    const double rci = impl::covalent[ni];
    for (int j = 0; j < N; ++j) {
      const int nj = atomic_numbers(j);
      const double r0 = (rci + impl::covalent[nj]) * D3_SCALE;
      const Vec3 dij_ang =
          positions_angstrom.col(i) - positions_angstrom.col(j);
      for_each_lattice_translation(lb, [&](const Vec3 &T_ang) {
        // Skip the i == j self-pair at T = 0.
        const bool central = T_ang.squaredNorm() < 1e-20;
        if (central && i == j) return;
        const Vec3 dR_ang = dij_ang + T_ang;
        const double r2 = dR_ang.squaredNorm();
        if (r2 > cutoff2 || r2 < 1e-20) return;
        const double r = std::sqrt(r2);
        cn(i) += 0.5 * (1.0 + std::erf(-kcn_steepness * (r - r0) / r0));
      });
    }
  }
  return cn;
}

Vec eeq_partial_charges_periodic(const IVec &atomic_numbers,
                                 const Mat3N &positions_angstrom,
                                 const Mat3 &lattice_angstrom,
                                 double total_charge, double tol) {
  using occ::units::ANGSTROM_TO_BOHR;
  const int N = atomic_numbers.rows();
  const double sqrt2_pi = std::sqrt(2.0 / kPi);

  // Bohr conversions for the Ewald math.
  const Mat3 lattice_bohr = lattice_angstrom * ANGSTROM_TO_BOHR;
  Mat3N positions_bohr = positions_angstrom * ANGSTROM_TO_BOHR;

  const EeqEwaldData ew = build_eeq_ewald_data(lattice_bohr, tol);

  Mat A = Mat::Zero(N + 1, N + 1);
  for (int i = 0; i < N; ++i) {
    const int ni = atomic_numbers(i);
    const double wi = impl::width[ni];
    for (int j = i; j < N; ++j) {
      const int nj = atomic_numbers(j);
      const double wj = impl::width[nj];
      const double gamma_pair = 1.0 / std::sqrt(wi * wi + wj * wj);
      const Vec3 R = positions_bohr.col(i) - positions_bohr.col(j);
      const bool same_pair = (i == j);
      const double v = eeq_pair_ewald(R, gamma_pair, ew, same_pair);
      A(i, j) += v;
      if (i != j) A(j, i) += v;
    }
    // Standard EEQ on-site terms.
    A(i, i) += impl::eta[ni] + sqrt2_pi / impl::width[ni];
  }
  A.row(N).array() = 1.0;
  A.col(N).array() = 1.0;
  A(N, N) = 0.0;

  const Vec cn =
      eeq_coordination_numbers_periodic(atomic_numbers, positions_angstrom,
                                         lattice_angstrom);
  Vec X(N + 1);
  constexpr double eps = 1e-14;
  for (int i = 0; i < N; ++i) {
    const int ni = atomic_numbers(i);
    const double tmp = impl::kcn[ni] / std::sqrt(cn(i) + eps);
    X(i) = -impl::chi[ni] + tmp * cn(i);
  }
  X(N) = total_charge;

  return A.householderQr().solve(X).topRows(N);
}

EeqWithGradient eeq_partial_charges_and_gradient(
    const IVec &atomic_numbers, const Mat3N &positions_angstrom,
    double charge) {
  using occ::units::ANGSTROM_TO_BOHR;
  const int N = atomic_numbers.rows();
  EeqWithGradient out;

  // ------ Coordination number + dCN/dR (in Bohr) ------
  // The base eeq_coordination_numbers uses Å for the (r - rc) erf argument
  // (rc is in Å too via the Pyykko table). We follow the same convention
  // here so cn matches; gradients are converted to per-Bohr at the end.
  Vec cn = Vec::Zero(N);
  std::vector<Mat3N> dcn(N, Mat3N::Zero(3, N));
  constexpr double kcn = 7.5;
  constexpr double D3_SCALE = 4.0 / 3.0;
  constexpr double cutoff = 25.0;
  constexpr double cutoff2 = cutoff * cutoff;
  constexpr double inv_sqrt_pi = 0.5641895835477563;
  for (int i = 0; i < N; ++i) {
    const int ni = atomic_numbers(i);
    for (int j = 0; j < i; ++j) {
      const int nj = atomic_numbers(j);
      const Vec3 rij = positions_angstrom.col(i) - positions_angstrom.col(j);
      const double r2 = rij.squaredNorm();
      if (r2 > cutoff2) continue;
      const double r = std::sqrt(r2);
      const double rc = (impl::covalent[ni] + impl::covalent[nj]) * D3_SCALE;
      const double t = -kcn * (r - rc) / rc;
      const double count = 0.5 * (1.0 + std::erf(t));
      cn(i) += count;
      cn(j) += count;
      // dcount/dr = 0.5 · (2/√π) · e^{−t²} · dt/dr = -k/rc · e^{−t²} / √π
      const double dcount_dr = -kcn / rc * std::exp(-t * t) * inv_sqrt_pi;
      const Vec3 dcount_dRi = dcount_dr * rij / r; // dr/dRi = (Ri-Rj)/r
      dcn[i].col(i) += dcount_dRi;
      dcn[i].col(j) -= dcount_dRi;
      dcn[j].col(i) += dcount_dRi;
      dcn[j].col(j) -= dcount_dRi;
    }
  }
  // dCN derivatives are wrt Å positions; convert to Bohr by dividing by
  // ANGSTROM_TO_BOHR (since dR_Å = dR_Bohr / ANGSTROM_TO_BOHR).
  for (auto &m : dcn) m *= (1.0 / ANGSTROM_TO_BOHR);

  // ------ Build A matrix (Bohr inside, as the existing impl does) ------
  Mat A = build_A_matrix(atomic_numbers, positions_angstrom);

  // ------ Build X vector (CN-modulated electronegativity) ------
  Vec X = build_X_vector(atomic_numbers, cn, charge);

  // ------ Solve A · q_full = X (q_full has the Lagrange multiplier appended) ------
  Eigen::PartialPivLU<Mat> lu(A);
  Vec q_full = lu.solve(X);
  out.charges = q_full.topRows(N);

  // ------ ∂q/∂R via the implicit-derivative system ------
  //   A · (∂q/∂R) = ∂X/∂R - (∂A/∂R) · q
  // Each (∂X/∂R)_i and (∂A/∂R)_{ij} only depends on a few atoms, so we
  // assemble the RHS sparsely. We solve all 3N right-hand sides with one LU
  // (already factorised above).
  //
  // ∂X_i/∂R_a: only via cn(i). dX_i/dCN_i = (-0.5 · kcn(ni) / sqrt(cn+ε)).
  // (Wait: X_i = -chi(ni) + kcn(ni)/sqrt(cn+ε) · cn = -chi(ni) + kcn(ni) ·
  // sqrt(cn+ε); for cn ≪ ε this is fine. Actually X_i = -chi + kcn·cn/√(cn+ε).
  // dX_i/dcn = kcn · [√(cn+ε) − cn/(2√(cn+ε))] / (cn+ε)
  //         = kcn · [(cn+ε) − cn/2] / (cn+ε)^(3/2)
  //         = kcn · [(cn/2 + ε)] / (cn+ε)^(3/2)
  //         ≈ kcn / (2 √(cn+ε))   for ε → 0).
  constexpr double eps = 1e-14;
  // Build RHS = (∂X/∂R - ∂A/∂R · q) as (N+1, 3N) matrix.
  Mat rhs = Mat::Zero(N + 1, 3 * N);
  // ∂X/∂R contribution: dX_i/dR_a^α = (dX_i/dcn_i) · dcn_i/dR_a^α
  for (int i = 0; i < N; ++i) {
    const int ni = atomic_numbers(i);
    const double cn_eps = cn(i) + eps;
    const double dXi_dcn =
        impl::kcn[ni] * (cn(i) / 2.0 + eps) / std::pow(cn_eps, 1.5);
    if (dXi_dcn == 0.0) continue;
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < 3; ++k) {
        rhs(i, 3 * j + k) += dXi_dcn * dcn[i](k, j);
      }
    }
  }
  // ∂A/∂R · q contribution. A_{ij} = erf(√(γ_ij·r²)) / r  for i ≠ j (in
  // Bohr). ∂A_{ij}/∂R_a^α nonzero only for a ∈ {i, j}. Diagonal A_{ii} is
  // R-independent.
  // Let g_ij = γ_ij = 1 / (w_i² + w_j²). Define u = r·√γ.
  //   d(erf(u))/dr  = (2/√π) e^{-u²} · √γ
  //   A_{ij} = erf(u) / r
  //   dA_{ij}/dr = [d(erf(u))/dr · r − erf(u)] / r²
  //             = [(2/√π) e^{-u²} √γ - A_{ij}] / r
  // Then dA_{ij}/dR_i^α = dA_{ij}/dr · (R_i^α − R_j^α)/r  (for the i atom).
  // Subtract dA · q from RHS for each gradient component.
  for (int i = 0; i < N; ++i) {
    const int ni = atomic_numbers(i);
    const double wi = impl::width[ni];
    const Vec3 ri_bohr = positions_angstrom.col(i) * ANGSTROM_TO_BOHR;
    for (int j = 0; j < i; ++j) {
      const int nj = atomic_numbers(j);
      const double wj = impl::width[nj];
      const Vec3 rj_bohr = positions_angstrom.col(j) * ANGSTROM_TO_BOHR;
      const Vec3 rij = ri_bohr - rj_bohr;
      const double r2 = rij.squaredNorm();
      if (r2 < 1e-12) continue;
      const double r = std::sqrt(r2);
      const double gamma = 1.0 / (wi * wi + wj * wj);
      const double sqrt_g = std::sqrt(gamma);
      const double u = r * sqrt_g;
      const double erf_u = std::erf(u);
      const double Aij = erf_u / r;
      const double dAij_dr =
          (2.0 * inv_sqrt_pi * std::exp(-u * u) * sqrt_g - Aij) / r;
      // dA_{ij}/dR_i^α = dA_{ij}/dr · (R_i^α - R_j^α) / r
      const Vec3 dAij_dRi = (dAij_dr / r) * rij;
      // Contribution to rhs(*, 3*i + α) and rhs(*, 3*j + α):
      // rhs_k -= ∂A_{kl}/∂R_a · q_l for all (k, l) with derivative nonzero.
      // The pair (i, j) contributes:
      //   dA_{ij}/dR_a · q_j  goes into rhs(i, ·)
      //   dA_{ji}/dR_a · q_i  goes into rhs(j, ·)
      // (A is symmetric ⇒ dA_{ij} = dA_{ji}; sign flips for j atom.)
      for (int k = 0; k < 3; ++k) {
        rhs(i, 3 * i + k) -= dAij_dRi(k) * q_full(j);
        rhs(j, 3 * i + k) -= dAij_dRi(k) * q_full(i);
        rhs(i, 3 * j + k) += dAij_dRi(k) * q_full(j); // ∂A/∂R_j = -∂A/∂R_i
        rhs(j, 3 * j + k) += dAij_dRi(k) * q_full(i);
      }
    }
  }
  // Solve A · dq_full = rhs.
  Mat dq_full = lu.solve(rhs);
  // Pack into per-atom Mat3N tensors.
  out.dcharges_dR.assign(N, Mat3N::Zero(3, N));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      for (int k = 0; k < 3; ++k) {
        out.dcharges_dR[i](k, j) = dq_full(i, 3 * j + k);
      }
    }
  }
  return out;
}

} // namespace occ::core::charges
