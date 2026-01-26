#include <algorithm>
#include <cassert>
#include <cmath>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/numint/grid_utils.h>
#include <occ/numint/lebedev.h>
#include <stdexcept>

namespace occ::dft {

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
      double a_ij = u_ij / (u_ij * u_ij - 1.0);
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
      double a_ij = u_ij / (u_ij * u_ij - 1.0);
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
  result.points.array() *= M_PI / (2 * num_points);
  result.points.array() = result.points.array().cos();
  result.weights.setConstant(M_PI / num_points);
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
    double theta = M_PI * k / (num_points + 1);
    double x = std::cos(theta);

    // M3 mapping: r = (xi / ln2) * ln(2 / (1-x))
    double r = (xi / ln2) * std::log(2.0 / (1.0 - x));

    // Weight transformation: w(r) = w(x) * dr/dx
    // For M3: dr/dx = xi / (ln2 * (1-x))
    // Gauss-Chebyshev second kind weight: w_k = pi/(n+1) * sin^2(theta)
    double sin_theta = std::sin(theta);
    double gc_weight = M_PI / (num_points + 1) * sin_theta * sin_theta;
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
    c0 = 4.0 * std::sqrt(2.0) * M_PI;
    cm = tgamma(3.0 / 2.0) / tgamma((m + 3.0) / 2.0);
    p0 = 1.0 / h;
    e0 = std::exp(-M_PI * M_PI / (2.0 * h));
    pm = std::pow(M_PI / h, m / 2.0);
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
  double far = 5.2;

  // Adjust far parameter based on atomic number
  switch (charge) {
  case 3:
  case 4:
  case 11:
  case 12:
  case 19:
  case 20:
    far = 7;
    break;
  }

  for (size_t i = 0; i < num_points; i++) {
    double x = (i + 0.5) / num_points;
    double x2 = x * x;
    double x3 = x2 * x;
    result.points(i) = -far * std::log(1 - x3);
    result.weights(i) = far * 3 * x2 / ((1 - x3) * num_points);
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
  double step = M_PI / (num_points + 1);
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

// In grid_utils.cpp, implement the function with a switch statement:

AtomGrid generate_atom_grid(size_t atomic_number, const GridSettings &settings,
                            RadialGridMethod method, double alpha_max,
                            int l_max, const Vec &alpha_min) {

  // Start timer
  occ::timing::start(occ::timing::category::grid_init);

  // Create the appropriate radial grid based on the method
  RadialGrid radial;

  // Determine number of radial points
  size_t n_radial;
  if (settings.int_acc > 0) {
    // ORCA-style IntAcc-based radial point count
    n_radial = occ::io::calculate_radial_points_orca(settings.int_acc, atomic_number);
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
  radial.weights.array() *= 4 * M_PI;

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
} // namespace occ::dft
