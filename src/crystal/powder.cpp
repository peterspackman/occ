#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cmath>
#include <gemmi/elem.hpp>
#include <gemmi/it92.hpp>
#include <occ/core/parallel.h>
#include <occ/core/units.h>
#include <occ/crystal/powder.h>
#include <stdexcept>

namespace occ::crystal {

namespace {

using Rot3i = Eigen::Matrix3i;

// The rotations acting on reciprocal space: a face/reflection h maps under
// (R|t) to (R^T)^-1 h, and the group is closed under inversion, so the orbit of
// h is its orbit under {R^T}. Friedel's law adds -R^T.
std::vector<Rot3i> laue_rotations(const SpaceGroup &sg) {
  std::vector<Rot3i> result;
  auto already_present = [&result](const Rot3i &r) {
    return std::any_of(result.begin(), result.end(),
                       [&r](const Rot3i &x) { return x == r; });
  };
  for (const auto &symop : sg.symmetry_operations()) {
    Rot3i rt = symop.rotation().transpose().cast<int>();
    if (!already_present(rt))
      result.push_back(rt);
    Rot3i minus_rt = -rt;
    if (!already_present(minus_rt))
      result.push_back(minus_rt);
  }
  return result;
}

inline IVec3 apply(const Rot3i &r, const IVec3 &h) { return r * h; }

// Lexicographic ordering on (h, k, l), used to pick a canonical orbit
// representative.
inline bool lex_greater(const IVec3 &a, const IVec3 &b) {
  return std::tie(a(0), a(1), a(2)) > std::tie(b(0), b(1), b(2));
}

} // namespace

double d_spacing(const HKL &hkl, const UnitCell &uc) {
  // HKL::d returns |G| = 1/d when handed the reciprocal matrix.
  return 1.0 / hkl.d(uc.reciprocal());
}

double lorentz_polarization(double two_theta) {
  const double theta = 0.5 * two_theta;
  const double s = std::sin(theta);
  const double c = std::cos(theta);
  const double ct = std::cos(two_theta);
  return (1.0 + ct * ct) / (s * s * c);
}

std::vector<PowderPeak> unique_reflections(const Crystal &crystal,
                                           double d_min) {
  const UnitCell &uc = crystal.unit_cell();
  const std::vector<Rot3i> rotations = laue_rotations(crystal.space_group());
  const HKL limits = uc.hkl_limits(d_min);

  std::vector<PowderPeak> result;
  std::vector<IVec3> orbit;
  orbit.reserve(rotations.size());

  for (int h = -limits.h; h <= limits.h; h++) {
    for (int k = -limits.k; k <= limits.k; k++) {
      for (int l = -limits.l; l <= limits.l; l++) {
        if (h == 0 && k == 0 && l == 0)
          continue;
        HKL hkl{h, k, l};
        double d = d_spacing(hkl, uc);
        if (d < d_min)
          continue;

        // Rotations preserve d, so the whole orbit lies inside the same
        // resolution sphere and hence inside the enumerated box: it is safe to
        // keep only the lexicographically greatest member as representative.
        const IVec3 v(h, k, l);
        orbit.clear();
        bool is_representative = true;
        for (const auto &r : rotations) {
          IVec3 image = apply(r, v);
          if (lex_greater(image, v)) {
            is_representative = false;
            break;
          }
          if (std::find(orbit.begin(), orbit.end(), image) == orbit.end())
            orbit.push_back(image);
        }
        if (!is_representative)
          continue;

        PowderPeak peak;
        peak.hkl = hkl;
        peak.d = d;
        peak.multiplicity = static_cast<int>(orbit.size());
        result.push_back(peak);
      }
    }
  }
  return result;
}

CVec structure_factors(const Crystal &crystal,
                       const std::vector<PowderPeak> &reflections) {
  const CrystalAtomRegion &atoms = crystal.unit_cell_atoms();
  const Eigen::Index num_atoms = atoms.size();
  const size_t num_refl = reflections.size();

  // Group the unit cell atoms by element: the form factor f(s) then only needs
  // evaluating once per element per reflection, rather than once per atom.
  ankerl::unordered_dense::map<int, std::vector<Eigen::Index>> by_element;
  for (Eigen::Index i = 0; i < num_atoms; i++)
    by_element[atoms.atomic_numbers(i)].push_back(i);

  for (const auto &[z, _] : by_element) {
    if (!gemmi::IT92<double>::has(static_cast<gemmi::El>(z)))
      throw std::runtime_error(
          fmt::format("No X-ray scattering factors available for element Z={}",
                      z));
  }

  CVec result = CVec::Zero(num_refl);

  occ::parallel::parallel_for(size_t{0}, num_refl, [&](size_t idx) {
    const auto &refl = reflections[idx];
    const Vec3 h(refl.hkl.h, refl.hkl.k, refl.hkl.l);
    // (sin(theta)/lambda)^2, with sin(theta)/lambda = 1/(2d)
    const double stol2 = 1.0 / (4.0 * refl.d * refl.d);

    std::complex<double> f_total{0.0, 0.0};
    for (const auto &[z, indices] : by_element) {
      const double f = gemmi::IT92<double>::get(static_cast<gemmi::El>(z), 0)
                           .calculate_sf(stol2);
      std::complex<double> phase_sum{0.0, 0.0};
      for (Eigen::Index i : indices) {
        const double phase =
            2.0 * M_PI * h.dot(atoms.frac_pos.col(i));
        phase_sum += atoms.occupation(i) *
                     std::complex<double>(std::cos(phase), std::sin(phase));
      }
      f_total += f * phase_sum;
    }
    result(idx) = f_total;
  });

  return result;
}

PowderPattern::PowderPattern(std::vector<PowderPeak> peaks, double wavelength)
    : m_peaks(std::move(peaks)), m_wavelength(wavelength) {}

Vec PowderPattern::normalized_intensities() const {
  Vec result(m_peaks.size());
  for (size_t i = 0; i < m_peaks.size(); i++)
    result(i) = m_peaks[i].intensity;
  const double max = result.size() > 0 ? result.maxCoeff() : 0.0;
  if (max > 0.0)
    result *= 100.0 / max;
  return result;
}

std::pair<Vec, Vec> PowderPattern::profile(double two_theta_min,
                                           double two_theta_max, int num_bins,
                                           double fwhm) const {
  if (num_bins < 1)
    throw std::invalid_argument("profile requires at least one bin");
  if (two_theta_max <= two_theta_min)
    throw std::invalid_argument("profile requires two_theta_max > two_theta_min");

  const double width = (two_theta_max - two_theta_min) / num_bins;
  Vec centres(num_bins), counts = Vec::Zero(num_bins);
  for (int i = 0; i < num_bins; i++)
    centres(i) = two_theta_min + (i + 0.5) * width;

  for (const auto &peak : m_peaks) {
    int bin = static_cast<int>((peak.two_theta - two_theta_min) / width);
    if (bin < 0 || bin >= num_bins)
      continue;
    counts(bin) += peak.intensity;
  }

  if (fwhm <= 0.0)
    return {centres, counts};

  // convolve with a Gaussian truncated at +/- 4 sigma
  const double sigma = fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0)));
  const int half = std::max(1, static_cast<int>(std::ceil(4.0 * sigma / width)));
  Vec kernel(2 * half + 1);
  for (int i = -half; i <= half; i++) {
    const double x = i * width;
    kernel(i + half) = std::exp(-0.5 * x * x / (sigma * sigma));
  }
  kernel /= kernel.sum();

  Vec smoothed = Vec::Zero(num_bins);
  for (int i = 0; i < num_bins; i++) {
    if (counts(i) == 0.0)
      continue;
    const int lo = std::max(0, i - half);
    const int hi = std::min(num_bins - 1, i + half);
    for (int j = lo; j <= hi; j++)
      smoothed(j) += counts(i) * kernel(j - i + half);
  }
  return {centres, smoothed};
}

PowderPattern compute_powder_pattern(const Crystal &crystal,
                                     const PowderPatternSettings &settings) {
  const double lambda = settings.wavelength;
  if (lambda <= 0.0)
    throw std::invalid_argument("wavelength must be positive");
  if (settings.two_theta_max <= settings.two_theta_min)
    throw std::invalid_argument("two_theta_max must exceed two_theta_min");
  if (settings.two_theta_max >= 180.0)
    throw std::invalid_argument("two_theta_max must be less than 180 degrees");

  // lambda = 2 d sin(theta)  =>  the largest 2*theta sets the smallest d
  const double theta_max = 0.5 * occ::units::radians(settings.two_theta_max);
  const double d_min = lambda / (2.0 * std::sin(theta_max));

  std::vector<PowderPeak> reflections = unique_reflections(crystal, d_min);
  const CVec f = structure_factors(crystal, reflections);

  std::vector<PowderPeak> peaks;
  peaks.reserve(reflections.size());
  for (size_t i = 0; i < reflections.size(); i++) {
    PowderPeak peak = reflections[i];
    const double sin_theta = lambda / (2.0 * peak.d);
    if (sin_theta > 1.0)
      continue;
    const double two_theta = 2.0 * std::asin(sin_theta); // radians
    const double two_theta_deg = occ::units::degrees(two_theta);
    if (two_theta_deg < settings.two_theta_min ||
        two_theta_deg > settings.two_theta_max)
      continue;

    peak.two_theta = two_theta_deg;
    peak.f_squared = std::norm(f(i));
    peak.intensity =
        peak.multiplicity * peak.f_squared * lorentz_polarization(two_theta);
    peaks.push_back(peak);
  }

  std::sort(peaks.begin(), peaks.end(),
            [](const PowderPeak &a, const PowderPeak &b) {
              return a.two_theta < b.two_theta;
            });

  return PowderPattern(std::move(peaks), lambda);
}

} // namespace occ::crystal
