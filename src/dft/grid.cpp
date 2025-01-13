#include <occ/core/atom.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/dft/grid.h>
#include <occ/dft/lebedev.h>

namespace occ::dft {
using occ::qm::AOBasis;

const std::array<uint_fast16_t, 33> lebedev_grid_levels{
    1,    6,    14,   26,   38,   50,   74,   86,   110,  146,  170,
    194,  230,  266,  302,  350,  434,  590,  770,  974,  1202, 1454,
    1730, 2030, 2354, 2702, 3074, 3470, 3890, 4334, 4802, 5294, 5810};

uint_fast16_t angular_order(uint_fast16_t n) {
  for (int i = 0; i < 33; i++) {
    auto m = lebedev_grid_levels[i];
    if (m >= n)
      return m;
  }

  throw std::runtime_error("Request number of angular points too exceeds 5810");
  // unreachable
  return 1;
}

// Need to multiple by (1.0 / BOHR) to get the real RADII_BRAGG
constexpr std::array<double, 131> bragg_radii = {
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

occ::Vec becke_partition(const occ::Vec &w) {
  occ::Vec result = w;
  for (size_t i = 0; i < 3; i++) {
    result.array() =
        (3 - result.array() * result.array()) * result.array() * 0.5;
  }
  return result;
}

occ::Vec stratmann_scuseria_partition(const occ::Vec &w) {
  occ::Vec result(w.rows());
  constexpr double a = 0.64;
  for (size_t i = 0; i < w.rows(); i++) {
    double ma = w(i) / a;
    double ma2 = ma * ma;
    double det = ma / 16 * (35 + ma2 * (-35 + ma2 * (21 - 5 * ma2)));
    result(i) = (det <= a) ? -1 : 1;
  }
  return result;
}

occ::Mat interatomic_distances(const std::vector<occ::core::Atom> &atoms) {
  size_t natoms = atoms.size();
  occ::Mat dists(natoms, natoms);
  for (size_t i = 0; i < natoms; i++) {
    dists(i, i) = 0;
    for (size_t j = i + 1; j < natoms; j++) {
      double dx = atoms[i].x - atoms[j].x;
      double dy = atoms[i].y - atoms[j].y;
      double dz = atoms[i].z - atoms[j].z;
      dists(i, j) = sqrt(dx * dx + dy * dy + dz * dz);
      dists(j, i) = dists(i, j);
    }
  }
  return dists;
}

occ::IVec prune_nwchem_scheme(size_t nuclear_charge, size_t max_angular,
                              size_t num_radial, const occ::Vec &radii) {
  std::array<int, 5> lebedev_level;
  occ::IVec angular_grids(num_radial);
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

  constexpr double bohr{0.52917721092};
  double radius = bragg_radii[nuclear_charge - 1] / bohr;
  for (size_t i = 0; i < num_radial; i++) {
    double scale = radii(i) / radius;
    size_t place = std::distance(
        alphas.begin(), std::find_if(alphas.begin(), alphas.end(),
                                     [scale](double x) { return x > scale; }));
    angular_grids(i) = lebedev_grid_levels[lebedev_level[place]];
  }
  return angular_grids;
}

occ::IVec prune_numgrid_scheme(size_t atomic_number, size_t max_angular,
                               size_t min_angular, const occ::Vec &radii) {
  occ::IVec result(radii.rows());
  constexpr double bohr{0.52917721092};
  double rb = bragg_radii[atomic_number - 1] / (5 * bohr);
  for (int i = 0; i < radii.rows(); i++) {
    double r = radii(i);
    size_t num_angular = max_angular;
    if (r < rb) {
      num_angular = static_cast<size_t>(max_angular * (r / rb));
      num_angular = angular_order(num_angular);
      if (num_angular < min_angular)
        num_angular = min_angular;
    }
    result(i) = num_angular;
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

// TCA 106, 178 (2001), eq. 25
// we evaluate r_inner for s functions
double lmg_inner(const double max_error, const double alpha_inner) {
  int m = 0;
  double d = 1.9;

  double r = d - std::log(1.0 / max_error);
  r = r * 2.0 / (m + 3.0);
  r = std::exp(r) / (alpha_inner);
  r = std::sqrt(r);

  return r;
}

// TCA 106, 178 (2001), eq. 19
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

// TCA 106, 178 (2001), eqs. 17 and 18
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
                                    size_t l_max, const occ::Vec &alpha_min) {
  double r_inner = lmg_inner(radial_precision, 2 * alpha_max);
  double h = std::numeric_limits<float>::max();
  double r_outer = 0.0;
  double br = bragg_radii[atomic_number - 1];
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

AtomGrid generate_atom_grid(size_t atomic_number, size_t max_angular_points,
                            size_t radial_points) {
  AtomGrid result(radial_points * max_angular_points);

  size_t num_points = 0;
  size_t n_radial = radial_points;
  RadialGrid radial = generate_treutler_alrichs_radial_grid(n_radial);
  radial.weights.array() *=
      4 * M_PI * radial.points.array() * radial.points.array();
  occ::IVec n_angular = prune_nwchem_scheme(atomic_number, max_angular_points,
                                            n_radial, radial.points);
  for (size_t i = 0; i < n_radial; i++) {
    auto lebedev = grid::lebedev(n_angular(i));
    double r = radial.points(i);
    double w = radial.weights(i);
    result.points.block(0, num_points, 3, lebedev.rows()) =
        lebedev.leftCols(3).transpose() * r;
    result.weights.block(num_points, 0, lebedev.rows(), 1) = lebedev.col(3) * w;
    num_points += lebedev.rows();
  }
  result.atomic_number = atomic_number;
  result.points.conservativeResize(3, num_points);
  result.weights.conservativeResize(num_points);
  return result;
}

void MolecularGrid::ensure_settings() {

  if (m_settings.max_angular_points < m_settings.min_angular_points) {
    m_settings.max_angular_points = m_settings.min_angular_points + 1;
    occ::log::warn(
        "Invalid maximum angular grid points < minimum angular grid points "
        "- will be set equal to the minimum + 1 ({} points)",
        m_settings.max_angular_points);
  }

  // make sure lebedev grid levels exist or are clamped etc.
  {
    int new_maximum = occ::dft::grid::nearest_grid_level_at_or_above(
        m_settings.max_angular_points);
    if (new_maximum != m_settings.max_angular_points) {
      occ::log::debug("Clamping max angular grid points to next grid "
                      "level ({} -> {})",
                      new_maximum, m_settings.max_angular_points);
      m_settings.max_angular_points = new_maximum;
    }
  }

  {
    int new_minimum = occ::dft::grid::nearest_grid_level_at_or_above(
        m_settings.min_angular_points);
    if (new_minimum != m_settings.min_angular_points) {
      occ::log::debug("Clamping min angular grid points to next grid "
                      "level ({} -> {})",
                      new_minimum, m_settings.min_angular_points);
      m_settings.min_angular_points = new_minimum;
    }
  }

  occ::log::debug("DFT molecular grid settings:");
  occ::log::debug("max_angular_points        = {}",
                  m_settings.max_angular_points);
  occ::log::debug("min_angular_points        = {}",
                  m_settings.min_angular_points);
  occ::log::debug("radial_precision          = {:.3g}",
                  m_settings.radial_precision);
  occ::log::debug("reduced grid size (H, He) = {}",
                  m_settings.reduced_first_row_element_grid);
}

MolecularGrid::MolecularGrid(const AOBasis &basis,
                             const BeckeGridSettings &settings)
    : m_atomic_numbers(basis.atoms().size()),
      m_positions(3, basis.atoms().size()), m_settings(settings),
      m_l_max(basis.atoms().size()), m_alpha_max(basis.atoms().size()),
      m_alpha_min(basis.l_max() + 1, basis.atoms().size()) {
  occ::timing::start(occ::timing::category::grid_init);
  ensure_settings();
  size_t natom = basis.atoms().size();
  std::vector<int> unique_atoms;
  const auto atom_map = basis.atom_to_shell();
  for (size_t i = 0; i < natom; i++) {
    m_atomic_numbers(i) = basis.atoms()[i].atomic_number;
    unique_atoms.push_back(basis.atoms()[i].atomic_number);
    m_positions(0, i) = basis.atoms()[i].x;
    m_positions(1, i) = basis.atoms()[i].y;
    m_positions(2, i) = basis.atoms()[i].z;
    std::vector<double> atom_min_alpha;
    double max_alpha = 0.0;
    int max_l = basis.l_max();
    for (const auto &shell_idx : atom_map[i]) {
      const auto &shell = basis[shell_idx];
      for (int i = atom_min_alpha.size(); i < max_l + 1; i++) {
        atom_min_alpha.push_back(std::numeric_limits<double>::max());
      }

      atom_min_alpha[shell.l] =
          std::min(shell.exponents.minCoeff(), atom_min_alpha[shell.l]);
      max_alpha = std::max(max_alpha, shell.exponents.maxCoeff());
    }
    for (int l = 0; l <= max_l; l++)
      m_alpha_min(l, i) = atom_min_alpha[l];
    m_alpha_max(i) = max_alpha;
    m_l_max(i) = max_l;
  }
  std::sort(unique_atoms.begin(), unique_atoms.end());
  unique_atoms.erase(std::unique(unique_atoms.begin(), unique_atoms.end()),
                     unique_atoms.end());
  for (const auto &atom : unique_atoms) {
    m_unique_atom_grids.emplace_back(generate_lmg_atom_grid(atom));
  }
  m_dists = interatomic_distances(basis.atoms());
  occ::timing::stop(occ::timing::category::grid_init);
}

AtomGrid MolecularGrid::generate_partitioned_atom_grid(size_t atom_idx) const {
  occ::timing::start(occ::timing::category::grid_points);
  size_t natoms = n_atoms();
  occ::Vec3 center = m_positions.col(atom_idx);
  const size_t atomic_number = m_atomic_numbers(atom_idx);
  AtomGrid grid;
  grid.atomic_number = -1;
  for (const auto &ugrid : m_unique_atom_grids) {
    if (ugrid.atomic_number == atomic_number) {
      grid = ugrid;
    }
  }
  if (grid.atomic_number < 0)
    throw std::runtime_error("Unique atom grids not calculated");

  grid.points.colwise() += center;
  occ::Mat grid_dists(grid.num_points(), natoms);
  for (size_t i = 0; i < natoms; i++) {
    occ::Vec3 xyz = m_positions.col(i);
    grid_dists.col(i) = (grid.points.colwise() - xyz).colwise().norm();
  }
  occ::Mat becke_weights = occ::Mat::Ones(grid.num_points(), natoms);
  constexpr double bohr{0.52917721092};
  for (size_t i = 0; i < natoms; i++) {
    double r_i = bragg_radii[m_atomic_numbers(i) - 1] / bohr;
    for (size_t j = 0; j < i; j++) {
      double r_j = bragg_radii[m_atomic_numbers(j) - 1] / bohr;
      occ::Vec w = (grid_dists.col(i).array() - grid_dists.col(j).array()) /
                   m_dists(i, j);

      // treutler alrichs adjustment to bragg radii
      if (std::fabs(r_i - r_j) > 1e-14) {
        double xi = sqrt(r_i / r_j);
        double u_ij = (xi - 1) / (xi + 1);
        double a_ij = u_ij / (u_ij * u_ij - 1.0);
        w.array() += a_ij * (1 - w.array() * w.array());
      }
      w = becke_partition(w);
      for (size_t idx = 0; idx < w.rows(); idx++) {
        double v = w(idx);
        if (std::fabs(1.0 - v) < 1e-14) {
          becke_weights(idx, i) = 0.0;
        } else {
          becke_weights(idx, i) *= 0.5 * (1.0 - v);
          becke_weights(idx, j) *= 0.5 * (1.0 + v);
        }
      }
    }
  }
  grid.weights.array() *= becke_weights.col(atom_idx).array() /
                          becke_weights.array().rowwise().sum();
  occ::timing::stop(occ::timing::category::grid_points);
  return grid;
}

AtomGrid MolecularGrid::generate_lmg_atom_grid(size_t atomic_number) {
  size_t num_points = 0;
  size_t atom_idx;
  for (atom_idx = 0; atom_idx < n_atoms(); ++atom_idx) {
    if (m_atomic_numbers(atom_idx) == atomic_number)
      break;
  }
  assert(atom_idx < n_atoms());
  double alpha_max = m_alpha_max(atom_idx);
  size_t l_max = m_l_max(atom_idx);
  const occ::Vec &alpha_min = m_alpha_min.col(atom_idx);

  int num_angular = m_settings.max_angular_points;
  if (m_settings.reduced_first_row_element_grid && atomic_number < 3) {
    num_angular = occ::dft::grid::nearest_grid_level_below(num_angular);
    occ::log::debug("Reduced grid size for element {} = {}", atomic_number,
                    num_angular);
  } else {
    occ::log::debug("Max angular points for element {} = {}", atomic_number,
                    num_angular);
  }

  RadialGrid radial = generate_lmg_radial_grid(
      atomic_number, m_settings.radial_precision, alpha_max, l_max, alpha_min);
  size_t n_radial = radial.points.rows();
  radial.weights.array() *= 4 * M_PI;
  occ::IVec n_angular = prune_nwchem_scheme(atomic_number, num_angular,
                                            radial.num_points(), radial.points);

  AtomGrid result(n_angular.sum());
  for (size_t i = 0; i < n_radial; i++) {
    auto lebedev = grid::lebedev(n_angular(i));
    double r = radial.points(i);
    double w = radial.weights(i);
    result.points.block(0, num_points, 3, lebedev.rows()) =
        lebedev.leftCols(3).transpose() * r;
    result.weights.block(num_points, 0, lebedev.rows(), 1) = lebedev.col(3) * w;
    num_points += lebedev.rows();
  }
  result.atomic_number = atomic_number;
  occ::log::debug("{} total grid points for element {} ({} radial)",
                  result.points.cols(), atomic_number, n_angular.rows());
  return result;
}

} // namespace occ::dft
