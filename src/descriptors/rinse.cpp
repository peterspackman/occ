#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <occ/core/parallel.h>
#include <occ/descriptors/rinse.h>
#include <stdexcept>

namespace occ::descriptors {

namespace {

constexpr double pi = std::numbers::pi_v<double>;
const double sqrt2 = std::sqrt(2.0);
constexpr double tiny = std::numeric_limits<double>::min();

/// Reflections per spherical-harmonic block. The harmonics for a block are
/// materialised before being contracted, so this trades working-set size
/// against how much of the contraction lands in a BLAS-3 shaped product.
constexpr Eigen::Index harmonic_block = 256;

/// Upper bound on how many independent accumulators the reduction is split
/// across. Fixing this by reflection count rather than by thread count is what
/// makes the result independent of how many threads run.
constexpr int max_reduction_groups = 256;

} // namespace

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

void RinseParams::validate() const {
  if (n_max <= 0)
    throw std::invalid_argument(
        fmt::format("RINSE n_max must be positive, got {}", n_max));
  if (radial_scale <= 0.0)
    throw std::invalid_argument(fmt::format(
        "RINSE radial_scale must be positive, got {}", radial_scale));
  if (l_min < 0 || l_min % 2 != 0)
    throw std::invalid_argument(fmt::format(
        "RINSE l_min must be even and non-negative, got {}", l_min));
  if (l_max % 2 != 0)
    throw std::invalid_argument(
        fmt::format("RINSE l_max must be even, got {}", l_max));
  if (l_min >= l_max)
    throw std::invalid_argument(fmt::format(
        "RINSE l_min ({}) must be less than l_max ({})", l_min, l_max));
}

double RinseParams::q_max() const {
  if (n_max <= 0)
    return 0.0;
  switch (radial_basis) {
  case RinseRadialBasis::SmoothShellsCW:
    return radial_scale * std::max(n_max - 1, 1);
  case RinseRadialBasis::SmoothShellsNL:
  default:
    return radial_scale * std::cbrt(static_cast<double>(n_max));
  }
}

double RinseParams::sin_theta_over_lambda_max() const { return 0.5 * q_max(); }

std::vector<int> RinseParams::l_values() const {
  std::vector<int> result;
  for (int l = l_min; l < l_max; l += 2)
    result.push_back(l);
  return result;
}

int RinseParams::num_l_levels() const { return (l_max - l_min) / 2; }

int RinseParams::size() const { return n_max * num_l_levels(); }

// ---------------------------------------------------------------------------
// Radial basis
// ---------------------------------------------------------------------------

Mat rinse_radial_basis(Eigen::Ref<const Vec> q, const RinseParams &params) {
  const Eigen::Index num_points = q.size();
  const int n_max = params.n_max;
  Mat result(num_points, n_max);
  if (n_max <= 0)
    return result;
  if (n_max == 1) {
    result.setOnes();
    return result;
  }

  const Array q_clipped = q.array().max(0.0);
  const double scale = params.radial_scale;

  if (params.radial_basis == RinseRadialBasis::SmoothShellsCW) {
    // Linearly spaced Gaussians of width `scale`, renormalised to a partition
    // of unity so the basis sums to one at every q.
    for (int n = 0; n < n_max; n++) {
      const Array scaled = (q_clipped - scale * n) / scale;
      result.col(n) = (-0.5 * scaled.square()).exp();
    }
    const Array row_sum = result.rowwise().sum().array().max(tiny);
    result.array().colwise() /= row_sum;
    return result;
  }

  // Volume-uniform shells: edge n sits at scale * n^(1/3), so every shell spans
  // the same volume of reciprocal space no matter how many there are. Each
  // Gaussian is scaled by 1/(sigma_n c_n^2) so its q^2-weighted integral is the
  // same for every shell -- equal-volume shells then contribute equally to a
  // flat intensity field, and the monopole tracks only the intensity envelope.
  for (int n = 0; n < n_max; n++) {
    const double inner = scale * std::cbrt(static_cast<double>(n));
    const double outer = scale * std::cbrt(static_cast<double>(n + 1));
    const double centre = scale * std::cbrt(n + 0.5);
    const double sigma = std::max(outer - inner, tiny);
    const double norm = std::max(sigma * centre * centre, tiny);
    const Array scaled = (q_clipped - centre) / sigma;
    result.col(n) = (-0.5 * scaled.square()).exp() / norm;
  }
  return result;
}

// ---------------------------------------------------------------------------
// Real spherical harmonics, evaluated a block of directions at a time
// ---------------------------------------------------------------------------

namespace {

/// Scratch for one block of directions.
struct HarmonicWorkspace {
  HarmonicWorkspace(Eigen::Index block, int l_max_degree, int num_coefficients,
                    int n_max)
      : ylm(block, num_coefficients),
        cos_mphi(block, std::max(l_max_degree, 1)),
        sin_mphi(block, std::max(l_max_degree, 1)), weights(block, n_max),
        direction(block, 3), legendre(block, 3), cos_theta(block),
        sin_theta(block), sector(block) {}
  Mat ylm;
  Mat cos_mphi;
  Mat sin_mphi;
  Mat weights;
  Mat direction;
  /// The three rolling levels of the Legendre recurrence, cycled by index --
  /// swapping the columns themselves would copy the whole block twice a step,
  /// which costs more than the arithmetic it carries.
  Mat legendre;
  Vec cos_theta;
  Vec sin_theta;
  Vec sector;
};

/**
 * Real orthonormal spherical harmonics for a block of directions.
 *
 * Y_l^0 = P~_l^0, Y_l^m = sqrt(2) P~_l^m cos(m phi) and
 * Y_l^-m = sqrt(2) P~_l^m sin(m phi), where P~ is the fully normalised
 * associated Legendre function -- the sqrt((2l+1)(l-m)!/(4 pi (l+m)!)) is
 * carried through the recurrence rather than applied at the end, which keeps
 * the values order one instead of running through 10^47 on the way, and saves a
 * multiply per coefficient. The Condon-Shortley phase is dropped: it flips the
 * sign of odd-m coefficients, and the power spectrum squares them.
 *
 * The recurrence runs outermost over order m and inner over degree l, so every
 * P~_l^m is computed exactly once, vectorised over the block, with its two
 * coefficients precomputed -- a division here would cost more than the rest of
 * the loop put together. The trig table is a Chebyshev recurrence in cos(phi),
 * which avoids an arctangent and l_max transcendental calls per direction, and
 * carries the sqrt(2) so the store is a single multiply.
 */
void evaluate_harmonics(const Mat3N &q_vectors, const Vec &q_magnitudes,
                        Eigen::Index begin, Eigen::Index count,
                        int l_max_degree,
                        const std::vector<int> &degree_columns, const Mat &a,
                        const Mat &b, const Vec &sector_ratio,
                        HarmonicWorkspace &work) {
  auto direction = work.direction.topRows(count);
  const Array inverse_length =
      q_magnitudes.segment(begin, count).array().max(1e-12).inverse();
  for (int axis = 0; axis < 3; axis++)
    direction.col(axis) =
        q_vectors.row(axis).segment(begin, count).transpose().array() *
        inverse_length;

  auto cos_theta = work.cos_theta.head(count);
  auto sin_theta = work.sin_theta.head(count);
  cos_theta = direction.col(2).cwiseMax(-1.0).cwiseMin(1.0);
  sin_theta = (1.0 - cos_theta.array().square()).max(0.0).sqrt();

  if (l_max_degree > 0) {
    // cos(phi) = x / r_xy and sin(phi) = y / r_xy, no arctangent needed. At a
    // pole r_xy is zero, but there every Y_l^(m != 0) vanishes with P~_l^m, so
    // any finite value will do.
    const Array r_xy =
        (direction.col(0).array().square() + direction.col(1).array().square())
            .sqrt();
    const Array scale = sqrt2 / (r_xy > 0.0).select(r_xy, 1.0);
    auto cos_phi = work.cos_mphi.col(0).head(count);
    auto sin_phi = work.sin_mphi.col(0).head(count);
    cos_phi = direction.col(0).array() * scale;
    sin_phi = direction.col(1).array() * scale;

    if (l_max_degree >= 2) {
      // The tables carry a factor of sqrt(2), so the recurrence in cos(phi)
      // needs it divided back out of the multiplier and the constant.
      const Array two_cos = (2.0 / sqrt2) * cos_phi.array();
      work.cos_mphi.col(1).head(count) = two_cos * cos_phi.array() - sqrt2;
      work.sin_mphi.col(1).head(count) = two_cos * sin_phi.array();
      for (int m = 3; m <= l_max_degree; m++) {
        work.cos_mphi.col(m - 1).head(count) =
            two_cos * work.cos_mphi.col(m - 2).head(count).array() -
            work.cos_mphi.col(m - 3).head(count).array();
        work.sin_mphi.col(m - 1).head(count) =
            two_cos * work.sin_mphi.col(m - 2).head(count).array() -
            work.sin_mphi.col(m - 3).head(count).array();
      }
    }
  }

  auto sector = work.sector.head(count);
  int previous = 0, current = 1, next = 2;

  const auto store = [&](int l, int m) {
    const int column = degree_columns[l];
    const auto value = work.legendre.col(current).head(count).array();
    if (m == 0) {
      work.ylm.col(column + l).head(count) = value;
      return;
    }
    work.ylm.col(column + l + m).head(count) =
        value * work.cos_mphi.col(m - 1).head(count).array();
    work.ylm.col(column + l - m).head(count) =
        value * work.sin_mphi.col(m - 1).head(count).array();
  };

  sector.setConstant(sector_ratio(0)); // P~_0^0 = 1 / sqrt(4 pi)
  for (int m = 0; m <= l_max_degree; m++) {
    if (m > 0)
      sector = sector_ratio(m) * sector.array() * sin_theta.array();

    work.legendre.col(previous).head(count).setZero();
    work.legendre.col(current).head(count) = sector;
    if (degree_columns[m] >= 0)
      store(m, m);

    for (int l = m + 1; l <= l_max_degree; l++) {
      work.legendre.col(next).head(count) =
          a(l, m) * cos_theta.array() *
              work.legendre.col(current).head(count).array() +
          b(l, m) * work.legendre.col(previous).head(count).array();
      const int spent = previous;
      previous = current;
      current = next;
      next = spent;
      if (degree_columns[l] >= 0)
        store(l, m);
    }
  }
}

} // namespace

Rinse::Rinse(const RinseParams &params)
    : m_params(params), m_l_values(params.l_values()) {
  m_params.validate();
  m_l_max_degree = m_l_values.empty() ? 0 : m_l_values.back();

  m_degree_columns.assign(m_l_max_degree + 1, -1);
  m_num_coefficients = 0;
  for (const int l : m_l_values) {
    m_degree_columns[l] = m_num_coefficients;
    m_level_offsets.push_back(m_num_coefficients);
    m_num_coefficients += 2 * l + 1;
  }

  // The normalised Legendre recurrence,
  //   P~_l^m = a(l,m) cos(theta) P~_(l-1)^m + b(l,m) P~_(l-2)^m
  // with the l = m+1 step falling out of it because P~_(m-1)^m is zero, and the
  // sector step P~_m^m = sector(m) sin(theta) P~_(m-1)^(m-1).
  m_legendre_a = Mat::Zero(m_l_max_degree + 1, m_l_max_degree + 1);
  m_legendre_b = Mat::Zero(m_l_max_degree + 1, m_l_max_degree + 1);
  for (int m = 0; m <= m_l_max_degree; m++) {
    for (int l = m + 1; l <= m_l_max_degree; l++) {
      const double denominator = static_cast<double>(l) * l - m * m;
      m_legendre_a(l, m) = std::sqrt((4.0 * l * l - 1.0) / denominator);
      m_legendre_b(l, m) =
          -std::sqrt((2.0 * l + 1.0) * ((l - 1.0) * (l - 1.0) - m * m) /
                     ((2.0 * l - 3.0) * denominator));
    }
  }
  m_legendre_sector = Vec::Zero(m_l_max_degree + 1);
  m_legendre_sector(0) = 1.0 / std::sqrt(4.0 * pi);
  for (int m = 1; m <= m_l_max_degree; m++)
    m_legendre_sector(m) = std::sqrt((2.0 * m + 1.0) / (2.0 * m));
}

Mat Rinse::power_spectrum(const ReflectionList &reflections) const {
  const int n_max = m_params.n_max;
  const int num_levels = static_cast<int>(m_l_values.size());
  Mat spectrum = Mat::Zero(n_max, num_levels);

  const Eigen::Index num_reflections = reflections.size();
  if (num_reflections == 0)
    return spectrum;

  const Mat radial = rinse_radial_basis(reflections.q_magnitudes, m_params);

  const Eigen::Index num_blocks =
      (num_reflections + harmonic_block - 1) / harmonic_block;
  const Eigen::Index num_groups =
      std::min<Eigen::Index>(num_blocks, max_reduction_groups);

  // A(n, lm) summed independently per group, then folded together in group
  // order: the answer does not depend on how the work was scheduled.
  std::vector<Mat> group_coefficients(num_groups);
  std::vector<Vec> group_radial_sums(num_groups);

  occ::parallel::parallel_for(
      size_t{0}, static_cast<size_t>(num_groups), [&](size_t g) {
        const Eigen::Index first_block =
            static_cast<Eigen::Index>(g) * num_blocks / num_groups;
        const Eigen::Index last_block =
            (static_cast<Eigen::Index>(g) + 1) * num_blocks / num_groups;

        Mat coefficients = Mat::Zero(m_num_coefficients, n_max);
        Vec radial_sum = Vec::Zero(n_max);
        // Only as wide as this group actually needs: for a small structure the
        // whole reflection list is one short block, and the harmonic array is
        // by far the largest thing allocated here.
        HarmonicWorkspace work(
            std::min(harmonic_block,
                     num_reflections - first_block * harmonic_block),
            m_l_max_degree, m_num_coefficients, n_max);

        for (Eigen::Index b = first_block; b < last_block; b++) {
          const Eigen::Index begin = b * harmonic_block;
          const Eigen::Index count =
              std::min(harmonic_block, num_reflections - begin);

          evaluate_harmonics(reflections.q_vectors, reflections.q_magnitudes,
                             begin, count, m_l_max_degree, m_degree_columns,
                             m_legendre_a, m_legendre_b, m_legendre_sector,
                             work);

          auto weights = work.weights.topRows(count);
          weights = radial.middleRows(begin, count).array().colwise() *
                    reflections.intensities.segment(begin, count).array();

          coefficients.noalias() +=
              work.ylm.topRows(count).transpose() * weights;
          radial_sum += weights.colwise().sum().transpose();
        }
        group_coefficients[g] = std::move(coefficients);
        group_radial_sums[g] = std::move(radial_sum);
      });

  Mat coefficients = Mat::Zero(m_num_coefficients, n_max);
  Vec radial_sum = Vec::Zero(n_max);
  for (Eigen::Index g = 0; g < num_groups; g++) {
    coefficients += group_coefficients[g];
    radial_sum += group_radial_sums[g];
  }

  for (int k = 0; k < num_levels; k++) {
    const int width = 2 * m_l_values[k] + 1;
    spectrum.col(k) = coefficients.middleRows(m_level_offsets[k], width)
                          .colwise()
                          .squaredNorm()
                          .transpose();
  }

  if (m_params.monopole_normalisation) {
    // The l = 0 projection is A_n00 = Y_00 * sum_G I(G) R_n(|G|), the
    // spherically averaged scattering power in shell n. Dividing by its power
    // takes out the resolution-dependent intensity envelope shell by shell, and
    // because systematically absent reflections have I = 0 they neither
    // contribute to it nor disturb it.
    const double y00 = 0.5 / std::sqrt(pi);
    const Vec monopole = (radial_sum * y00).array().square();
    const double floor =
        std::numeric_limits<double>::epsilon() * monopole.maxCoeff();
    for (int n = 0; n < n_max; n++) {
      if (monopole(n) > floor)
        spectrum.row(n) /= monopole(n);
      else
        spectrum.row(n).setZero();
    }
  }

  spectrum = spectrum.cwiseMax(0.0);
  if (m_params.log1p)
    spectrum = spectrum.array().log1p();
  if (m_params.l2) {
    const double norm = spectrum.norm();
    if (norm > 0.0)
      spectrum /= norm;
  }
  return spectrum;
}

Mat Rinse::operator()(const crystal::Crystal &crystal) const {
  return power_spectrum(rinse_reflections(crystal, m_params));
}

Vec Rinse::compute(const crystal::Crystal &crystal) const {
  return flatten((*this)(crystal));
}

Vec Rinse::flatten(Eigen::Ref<const Mat> spectrum) {
  Vec result(spectrum.size());
  Eigen::Index index = 0;
  for (Eigen::Index n = 0; n < spectrum.rows(); n++)
    for (Eigen::Index k = 0; k < spectrum.cols(); k++)
      result(index++) = spectrum(n, k);
  return result;
}

} // namespace occ::descriptors
