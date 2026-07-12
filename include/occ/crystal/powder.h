#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/hkl.h>
#include <occ/crystal/unitcell.h>
#include <utility>
#include <vector>

namespace occ::crystal {

/**
 * \brief Characteristic K-alpha X-ray wavelengths, in Angstroms.
 */
namespace xray_wavelength {
inline constexpr double Cu_Ka = 1.5405980;
inline constexpr double Mo_Ka = 0.7107300;
inline constexpr double Co_Ka = 1.7889650;
inline constexpr double Cr_Ka = 2.2897000;
inline constexpr double Fe_Ka = 1.9373500;
inline constexpr double Ag_Ka = 0.5594180;
} // namespace xray_wavelength

/**
 * \brief A single symmetry-unique powder reflection.
 */
struct PowderPeak {
  /// The (symmetry-unique representative) Miller indices
  HKL hkl;
  /// d-spacing in Angstroms
  double d{0.0};
  /// Bragg angle 2*theta, in **degrees**
  double two_theta{0.0};
  /// Number of distinct reflections in this reflection's orbit
  int multiplicity{1};
  /// |F(hkl)|^2, in electrons^2
  double f_squared{0.0};
  /// multiplicity * |F|^2 * Lp
  double intensity{0.0};
};

struct PowderPatternSettings {
  /// Incident wavelength in Angstroms
  double wavelength{xray_wavelength::Cu_Ka};
  /// Lower limit of the 2*theta range, in degrees
  double two_theta_min{5.0};
  /// Upper limit of the 2*theta range, in degrees
  double two_theta_max{50.0};
};

/**
 * \brief A simulated powder X-ray diffraction pattern.
 *
 * Intensities use neutral-atom X-ray form factors (International Tables 1992)
 * and the Lorentz-polarization factor. No Debye-Waller (temperature) factor
 * and no anomalous dispersion (f', f'') are applied, so relative intensities
 * are most accurate for light atoms away from an absorption edge.
 *
 * Systematically absent reflections need no special handling: their structure
 * factor is identically zero.
 */
class PowderPattern {
public:
  PowderPattern(std::vector<PowderPeak> peaks, double wavelength);

  /// The symmetry-unique peaks, sorted by increasing 2*theta
  inline const std::vector<PowderPeak> &peaks() const { return m_peaks; }
  inline double wavelength() const { return m_wavelength; }
  inline size_t size() const { return m_peaks.size(); }

  /// Peak intensities rescaled so that the strongest peak is 100.
  Vec normalized_intensities() const;

  /**
   * \brief Bin the peaks onto a 2*theta grid and broaden them.
   *
   * \param two_theta_min lower edge of the grid, in degrees
   * \param two_theta_max upper edge of the grid, in degrees
   * \param num_bins number of grid points
   * \param fwhm full width at half maximum of the Gaussian, in degrees.
   *        A value <= 0 disables broadening (a raw histogram).
   *
   * \returns a pair of (bin centres, intensities)
   */
  std::pair<Vec, Vec> profile(double two_theta_min, double two_theta_max,
                              int num_bins = 4500, double fwhm = 0.1) const;

private:
  std::vector<PowderPeak> m_peaks;
  double m_wavelength;
};

/**
 * \brief The d-spacing of a reflection, in Angstroms.
 *
 * \note This exists because `HKL::d(cell.reciprocal())` returns the *reciprocal*
 * lattice vector length |G| = 1/d, in Angstrom^-1, despite its name. Prefer this
 * function whenever an actual d-spacing is wanted.
 */
double d_spacing(const HKL &hkl, const UnitCell &uc);

/**
 * \brief Enumerate the symmetry-unique reflections with d >= d_min.
 *
 * Reflections are reduced under the Laue group (the rotation parts of the space
 * group, plus inversion by Friedel's law). Each returned peak carries the
 * multiplicity of its orbit. Only `hkl`, `d` and `multiplicity` are set.
 */
std::vector<PowderPeak> unique_reflections(const Crystal &crystal,
                                           double d_min);

/**
 * \brief Structure factors F(hkl) for the given reflections.
 *
 * \f[ F(h) = \sum_j occ_j \, f_j(s) \, \exp(2 \pi i \, h \cdot r_j) \f]
 *
 * summed over the unit cell atoms, with \f$ s = \sin\theta/\lambda = 1/(2d) \f$.
 */
CVec structure_factors(const Crystal &crystal,
                       const std::vector<PowderPeak> &reflections);

/// Lorentz-polarization factor for an unpolarized source; two_theta in radians.
double lorentz_polarization(double two_theta);

/// Compute a powder pattern for the given crystal.
PowderPattern compute_powder_pattern(const Crystal &crystal,
                                     const PowderPatternSettings &settings = {});

} // namespace occ::crystal
