#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <numbers>
#include <occ/core/bondgraph.h>
#include <occ/core/element.h>
#include <occ/core/kdtree.h>
#include <occ/core/parallel.h>
#include <occ/core/units.h>
#include <occ/solvent/sigma_profile.h>
#include <stdexcept>

namespace occ::solvent::sigma {

namespace {

constexpr double BOHR2_TO_ANGS2 =
    occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;

bool is_hbond_heavy_atom(int z) { return z == 7 || z == 8 || z == 9; }

/// Deposit weights for a value on a node grid: the two bracketing node
/// indices and the fraction belonging to the upper one. Values outside the
/// grid clamp onto the end node.
struct BinWeights {
  int lower;
  int upper;
  double frac;
  bool out_of_range;
};

BinWeights bin_weights(const Grid &grid, double value) {
  if (value <= grid.lo)
    return {0, 0, 0.0, value < grid.lo};
  if (value >= grid.hi)
    return {grid.n - 1, grid.n - 1, 0.0, value > grid.hi};
  const double t = (value - grid.lo) / grid.spacing();
  int lower = static_cast<int>(std::floor(t));
  lower = std::clamp(lower, 0, grid.n - 2);
  return {lower, lower + 1, t - lower, false};
}

void require_same_grid(const Grid &a, const Grid &b, const char *what) {
  if (a != b)
    throw std::runtime_error(fmt::format("{}: grids do not match", what));
}

} // namespace

Vec Grid::centers() const {
  if (n < 2)
    throw std::runtime_error("sigma::Grid needs at least 2 bins");
  Vec c(n);
  const double h = spacing();
  for (int i = 0; i < n; i++)
    c(i) = lo + i * h;
  return c;
}

double Grid::spacing() const {
  if (n < 2)
    throw std::runtime_error("sigma::Grid needs at least 2 bins");
  return (hi - lo) / (n - 1);
}

bool Grid::operator==(const Grid &other) const {
  return n == other.n && std::abs(lo - other.lo) < 1e-12 &&
         std::abs(hi - other.hi) < 1e-12;
}

double Segments::total_charge() const { return areas.dot(sigma); }

double Segments::total_charge_averaged() const {
  if (sigma_averaged.size() != areas.size())
    return 0.0;
  return areas.dot(sigma_averaged);
}

Segments segments_from_cavity(const surface::Surface &cavity,
                              const Vec &charges) {
  const Eigen::Index n = cavity.areas.size();
  if (charges.size() != n)
    throw std::runtime_error(
        fmt::format("segments_from_cavity: {} charges for {} cavity elements",
                    charges.size(), n));

  Segments s;
  s.positions = cavity.vertices;
  s.areas = cavity.areas * BOHR2_TO_ANGS2;
  s.atom_index = cavity.atom_index;
  s.sigma = charges.array() / s.areas.array();
  s.sigma_averaged = Vec();
  s.hbond_class = IVec::Constant(n, static_cast<int>(HBondClass::None));
  return s;
}

void average_sigma(Segments &segments, double r_av_angs) {
  const Eigen::Index n = segments.size();
  segments.sigma_averaged = Vec::Zero(n);
  if (n == 0)
    return;

  const double r_av2 = r_av_angs * r_av_angs;

  // Equal-area disc radius per segment, and the Gaussian denominator it
  // implies.
  Vec denom(n);
  for (Eigen::Index j = 0; j < n; j++) {
    const double r2 = segments.areas(j) / std::numbers::pi_v<double>;
    denom(j) = r2 + r_av2;
  }
  Vec prefactor(n);
  for (Eigen::Index j = 0; j < n; j++) {
    const double r2 = segments.areas(j) / std::numbers::pi_v<double>;
    prefactor(j) = r2 * r_av2 / denom(j);
  }

  // exp(-d^2/denom) < 1e-12 beyond ~5.3 sqrt(denom).
  const double cutoff = 5.5 * std::sqrt(denom.maxCoeff());

  Mat3N positions_angs = segments.positions * occ::units::BOHR_TO_ANGSTROM;
  occ::core::KDTree<double> tree(3, positions_angs, occ::core::max_leaf);
  tree.index->buildIndex();

  occ::parallel::parallel_for(size_t(0), size_t(n), [&](size_t i) {
    std::vector<std::pair<Eigen::Index, double>> neighbors;
    nanoflann::RadiusResultSet<double, Eigen::Index> results(cutoff * cutoff,
                                                             neighbors);
    Vec3 p = positions_angs.col(i);
    tree.index->findNeighbors(results, p.data(), nanoflann::SearchParams());

    double numerator = 0.0, weight_sum = 0.0;
    for (const auto &[j, d2] : neighbors) {
      const double w = prefactor(j) * std::exp(-d2 / denom(j));
      numerator += segments.sigma(j) * w;
      weight_sum += w;
    }
    segments.sigma_averaged(i) =
        (weight_sum > 0.0) ? numerator / weight_sum : segments.sigma(i);
  });
}

void classify_hbond_segments(Segments &segments, const IVec &atomic_numbers,
                             const Mat3N &atom_positions_bohr) {
  const Eigen::Index natoms = atomic_numbers.size();
  IVec atom_class = IVec::Constant(natoms, static_cast<int>(HBondClass::None));

  Vec cov_radii(natoms);
  for (Eigen::Index a = 0; a < natoms; a++) {
    cov_radii(a) = occ::core::Element(atomic_numbers(a)).covalent_radius();
    if (is_hbond_heavy_atom(atomic_numbers(a)))
      atom_class(a) = static_cast<int>(HBondClass::OT);
  }

  Mat3N positions_angs = atom_positions_bohr * occ::units::BOHR_TO_ANGSTROM;
  for (Eigen::Index h = 0; h < natoms; h++) {
    if (atomic_numbers(h) != 1)
      continue;
    for (Eigen::Index a = 0; a < natoms; a++) {
      if (!is_hbond_heavy_atom(atomic_numbers(a)))
        continue;
      const double threshold =
          cov_radii(h) + cov_radii(a) + occ::core::covalent_bond_tolerance;
      const double d =
          (positions_angs.col(h) - positions_angs.col(a)).squaredNorm();
      if (d < threshold * threshold) {
        atom_class(h) = static_cast<int>(HBondClass::OH);
        break;
      }
    }
  }

  segments.hbond_class = IVec(segments.size());
  for (Eigen::Index i = 0; i < segments.size(); i++)
    segments.hbond_class(i) = atom_class(segments.atom_index(i));
}

Vec Profile::total() const { return values.rowwise().sum(); }

Mat Profile::normalized() const {
  const double a = total_area();
  return (a > 0.0) ? Mat(values / a) : Mat(Mat::Zero(values.rows(), values.cols()));
}

Profile bin_segments(const Segments &segments, const Grid &grid,
                     bool resolve_hbond_classes, double *out_of_range_area) {
  const int ncols = resolve_hbond_classes ? num_hbond_classes : 1;
  Profile profile;
  profile.grid = grid;
  profile.values = Mat::Zero(grid.n, ncols);

  const Vec &sigma = (segments.sigma_averaged.size() == segments.size())
                         ? segments.sigma_averaged
                         : segments.sigma;
  double outside = 0.0;

  for (Eigen::Index i = 0; i < segments.size(); i++) {
    const auto w = bin_weights(grid, sigma(i));
    if (w.out_of_range)
      outside += segments.areas(i);
    int col = 0;
    if (resolve_hbond_classes) {
      col = segments.hbond_class.size() == segments.size()
                ? segments.hbond_class(i)
                : 0;
      col = std::clamp(col, 0, ncols - 1);
    }
    profile.values(w.lower, col) += segments.areas(i) * (1.0 - w.frac);
    if (w.upper != w.lower)
      profile.values(w.upper, col) += segments.areas(i) * w.frac;
  }

  if (out_of_range_area)
    *out_of_range_area = outside;
  return profile;
}

Profile mix_profiles(const std::vector<Profile> &components,
                     const Vec &mole_fractions) {
  if (components.empty())
    throw std::runtime_error("mix_profiles: no components");
  if (mole_fractions.size() != static_cast<Eigen::Index>(components.size()))
    throw std::runtime_error(fmt::format(
        "mix_profiles: {} mole fractions for {} components",
        mole_fractions.size(), components.size()));

  Profile mixture;
  mixture.grid = components.front().grid;
  mixture.values =
      Mat::Zero(mixture.grid.n, components.front().values.cols());
  for (size_t k = 0; k < components.size(); k++) {
    require_same_grid(mixture.grid, components[k].grid, "mix_profiles");
    if (components[k].values.cols() != mixture.values.cols())
      throw std::runtime_error("mix_profiles: class count mismatch");
    mixture.values += mole_fractions(k) * components[k].values;
  }
  return mixture;
}

double contract(const Profile &profile, const Mat &field) {
  if (field.rows() != profile.values.rows() ||
      field.cols() != profile.values.cols())
    throw std::runtime_error(fmt::format(
        "contract: field is {}x{}, profile is {}x{}", field.rows(),
        field.cols(), profile.values.rows(), profile.values.cols()));
  return (profile.values.array() * field.array()).sum();
}

Vec contract_segments(const Segments &segments, const Grid &grid,
                      const Mat &field) {
  if (field.rows() != grid.n)
    throw std::runtime_error(fmt::format(
        "contract_segments: field has {} rows, grid has {} bins", field.rows(),
        grid.n));

  const Vec &sigma = (segments.sigma_averaged.size() == segments.size())
                         ? segments.sigma_averaged
                         : segments.sigma;
  const int ncols = static_cast<int>(field.cols());
  Vec out(segments.size());

  for (Eigen::Index i = 0; i < segments.size(); i++) {
    const auto w = bin_weights(grid, sigma(i));
    int col = 0;
    if (ncols > 1) {
      col = segments.hbond_class.size() == segments.size()
                ? segments.hbond_class(i)
                : 0;
      col = std::clamp(col, 0, ncols - 1);
    }
    const double interpolated =
        field(w.lower, col) * (1.0 - w.frac) +
        ((w.upper != w.lower) ? field(w.upper, col) * w.frac : 0.0);
    out(i) = segments.areas(i) * interpolated;
  }
  return out;
}

} // namespace occ::solvent::sigma
