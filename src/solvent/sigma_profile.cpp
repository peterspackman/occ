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

/// Which column a segment deposits into before the P_hb weighting: only the
/// acceptor lobe of a heavy atom and the donor lobe of a hydrogen take part.
int hbond_column(int hb_class, int atomic_number, double sigma) {
  if (hb_class == static_cast<int>(HBondClass::None))
    return static_cast<int>(HBondClass::None);
  const bool acceptor_lobe = is_hbond_heavy_atom(atomic_number) && sigma > 0.0;
  const bool donor_lobe = atomic_number == 1 && sigma < 0.0;
  return (acceptor_lobe || donor_lobe) ? hb_class
                                       : static_cast<int>(HBondClass::None);
}

/// P_hb(σ) = 1 − exp(−σ²/2σ_0²), the fraction of a qualifying segment's area
/// that actually hydrogen bonds.
Vec hbond_weights(const Grid &grid, double sigma_0) {
  Vec centers = grid.centers();
  const double denom = 2.0 * sigma_0 * sigma_0;
  return 1.0 - (-centers.array().square() / denom).exp();
}

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
                              const Vec &charges, const IVec &atomic_numbers) {
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
  s.atomic_number = IVec(n);
  for (Eigen::Index i = 0; i < n; i++) {
    const int a = cavity.atom_index(i);
    if (a < 0 || a >= atomic_numbers.size())
      throw std::runtime_error(fmt::format(
          "segments_from_cavity: cavity element {} refers to atom {} of {}", i,
          a, atomic_numbers.size()));
    s.atomic_number(i) = atomic_numbers(a);
  }
  return s;
}

Vec averaged_sigma(const Segments &segments, double r_av_angs,
                   double f_decay) {
  const Eigen::Index n = segments.size();
  Vec result = Vec::Zero(n);
  if (n == 0)
    return result;

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

  // exp(-f_decay d^2/denom) < 1e-12 beyond ~5.3 sqrt(denom/f_decay).
  const double cutoff = 5.5 * std::sqrt(denom.maxCoeff() / f_decay);

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
      const double w = prefactor(j) * std::exp(-f_decay * d2 / denom(j));
      numerator += segments.sigma(j) * w;
      weight_sum += w;
    }
    result(i) = (weight_sum > 0.0) ? numerator / weight_sum : segments.sigma(i);
  });
  return result;
}

void average_sigma(Segments &segments, double r_av_angs, double f_decay) {
  segments.sigma_averaged = averaged_sigma(segments, r_av_angs, f_decay);
}

void average_sigma_orth(Segments &segments, double r_av_angs,
                        double r_corr_angs, double factor) {
  if (segments.sigma_averaged.size() != segments.size())
    throw std::runtime_error(
        "average_sigma_orth: call average_sigma on the same radius first");
  segments.sigma_orth =
      averaged_sigma(segments, r_corr_angs) - factor * segments.sigma_averaged;
}

void classify_hbond_segments(Segments &segments, const IVec &atomic_numbers,
                             const Mat3N &atom_positions_bohr) {
  const Eigen::Index natoms = atomic_numbers.size();
  IVec atom_class = IVec::Constant(natoms, static_cast<int>(HBondClass::None));

  Vec cov_radii(natoms);
  for (Eigen::Index a = 0; a < natoms; a++)
    cov_radii(a) = occ::core::Element(atomic_numbers(a)).covalent_radius();

  Mat3N positions_angs = atom_positions_bohr * occ::units::BOHR_TO_ANGSTROM;
  auto bonded = [&](Eigen::Index i, Eigen::Index j) {
    const double threshold =
        cov_radii(i) + cov_radii(j) + occ::core::covalent_bond_tolerance;
    return (positions_angs.col(i) - positions_angs.col(j)).squaredNorm() <
           threshold * threshold;
  };

  // O with an attached H (and that H) is the OH class; every other N, O, F,
  // and any H on N or F, is OT.
  for (Eigen::Index a = 0; a < natoms; a++) {
    if (!is_hbond_heavy_atom(atomic_numbers(a)))
      continue;
    bool has_hydrogen = false;
    for (Eigen::Index h = 0; h < natoms; h++) {
      if (atomic_numbers(h) != 1 || !bonded(a, h))
        continue;
      has_hydrogen = true;
      atom_class(h) = (atomic_numbers(a) == 8)
                          ? static_cast<int>(HBondClass::OH)
                          : static_cast<int>(HBondClass::OT);
    }
    atom_class(a) = (has_hydrogen && atomic_numbers(a) == 8)
                        ? static_cast<int>(HBondClass::OH)
                        : static_cast<int>(HBondClass::OT);
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
                     HBondSplit split, double *out_of_range_area) {
  const int ncols = split.enabled ? num_hbond_classes : 1;
  Profile profile;
  profile.grid = grid;
  profile.values = Mat::Zero(grid.n, ncols);

  const Vec &sigma = (segments.sigma_averaged.size() == segments.size())
                         ? segments.sigma_averaged
                         : segments.sigma;
  const bool have_classes = segments.hbond_class.size() == segments.size() &&
                            segments.atomic_number.size() == segments.size();
  double outside = 0.0;

  for (Eigen::Index i = 0; i < segments.size(); i++) {
    const auto w = bin_weights(grid, sigma(i));
    if (w.out_of_range)
      outside += segments.areas(i);
    int col = 0;
    if (split.enabled && have_classes)
      col = hbond_column(segments.hbond_class(i), segments.atomic_number(i),
                         sigma(i));
    profile.values(w.lower, col) += segments.areas(i) * (1.0 - w.frac);
    if (w.upper != w.lower)
      profile.values(w.upper, col) += segments.areas(i) * w.frac;
  }

  if (split.enabled) {
    Vec p_hb = hbond_weights(grid, split.sigma_0);
    const int nhb = static_cast<int>(HBondClass::None);
    const int oh = static_cast<int>(HBondClass::OH);
    const int ot = static_cast<int>(HBondClass::OT);
    for (int b = 0; b < grid.n; b++) {
      const double returned =
          (1.0 - p_hb(b)) * (profile.values(b, oh) + profile.values(b, ot));
      profile.values(b, oh) *= p_hb(b);
      profile.values(b, ot) *= p_hb(b);
      profile.values(b, nhb) += returned;
    }
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
                      const Mat &field, HBondSplit split) {
  if (field.rows() != grid.n)
    throw std::runtime_error(fmt::format(
        "contract_segments: field has {} rows, grid has {} bins", field.rows(),
        grid.n));

  const Vec &sigma = (segments.sigma_averaged.size() == segments.size())
                         ? segments.sigma_averaged
                         : segments.sigma;
  const bool resolve = split.enabled && field.cols() > 1 &&
                       segments.hbond_class.size() == segments.size() &&
                       segments.atomic_number.size() == segments.size();
  const int nhb = static_cast<int>(HBondClass::None);
  Vec p_hb = resolve ? hbond_weights(grid, split.sigma_0) : Vec();

  // Mirrors the deposit in bin_segments, including the per-bin P_hb split,
  // so the summed result matches the binned contraction exactly.
  auto value_at = [&](int bin, int col) {
    if (!resolve || col == nhb)
      return field(bin, resolve ? nhb : 0);
    return p_hb(bin) * field(bin, col) + (1.0 - p_hb(bin)) * field(bin, nhb);
  };

  Vec out(segments.size());
  for (Eigen::Index i = 0; i < segments.size(); i++) {
    const auto w = bin_weights(grid, sigma(i));
    const int col =
        resolve ? hbond_column(segments.hbond_class(i),
                               segments.atomic_number(i), sigma(i))
                : 0;
    double interpolated = value_at(w.lower, col) * (1.0 - w.frac);
    if (w.upper != w.lower)
      interpolated += value_at(w.upper, col) * w.frac;
    out(i) = segments.areas(i) * interpolated;
  }
  return out;
}

} // namespace occ::solvent::sigma
