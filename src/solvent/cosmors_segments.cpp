#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <numbers>
#include <occ/core/kdtree.h>
#include <occ/core/parallel.h>
#include <occ/core/units.h>
#include <occ/solvent/cosmors_segments.h>
#include <stdexcept>

namespace occ::solvent::cosmors {

namespace {

constexpr double BOHR2_TO_ANGS2 =
    occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;

} // namespace

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

Vec averaged_sigma(const Segments &segments, double r_av_angs, double f_decay) {
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

} // namespace occ::solvent::cosmors
