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
/// K-alpha_1 lines
inline constexpr double Cu_Ka1 = 1.5405980;
inline constexpr double Mo_Ka1 = 0.7093000;
inline constexpr double Co_Ka1 = 1.7889650;
inline constexpr double Cr_Ka1 = 2.2897000;
inline constexpr double Fe_Ka1 = 1.9360400;
inline constexpr double Ag_Ka1 = 0.5594180;

/// Intensity-weighted mean of the K-alpha_1 / K-alpha_2 doublet. A real
/// laboratory source emits both; the doublet is not modelled, so these are the
/// right choice when comparing against an unstripped measured pattern.
inline constexpr double Cu_Ka = 1.5418740;
inline constexpr double Mo_Ka = 0.7107300;
inline constexpr double Co_Ka = 1.7902600;
inline constexpr double Cr_Ka = 2.2909100;
inline constexpr double Fe_Ka = 1.9373500;
inline constexpr double Ag_Ka = 0.5608380;
} // namespace xray_wavelength

/**
 * \brief A single powder peak.
 *
 * Reflections from different orbits that share a d-spacing are indistinguishable
 * in a powder experiment and are merged into one peak: silicon's (333) and (511)
 * both sit at 2*theta = 94.95 degrees, and what is observed is their sum. `hkl`
 * is then the strongest contributing reflection and `multiplicity` the total.
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
  double wavelength{xray_wavelength::Cu_Ka1};
  /// Lower limit of the 2*theta range, in degrees. Must be greater than zero:
  /// the Lorentz factor diverges as 1/theta^2.
  double two_theta_min{5.0};
  /// Upper limit of the 2*theta range, in degrees
  double two_theta_max{50.0};
  /// Apply Debye-Waller (temperature) factors from the asymmetric unit's ADPs.
  /// A no-op when the ADPs are zero, so it is safe to leave on. Neglecting it
  /// overestimates |F|^2 badly at higher angle -- for a structure with
  /// U ~ 0.07 A^2 the error is already tens of percent by 2*theta = 30 degrees.
  bool debye_waller{true};
  /// Reflections whose |F|^2 falls below this are dropped from the peak list.
  /// Systematically absent reflections have |F|^2 identically zero, and a peak
  /// list full of them is not a useful product.
  double min_f_squared{1e-6};
  /// Refuse to enumerate more than this many points of reciprocal space. Guards
  /// against a wavelength typo (nanometres for Angstroms, say) asking for a
  /// hundred billion reflections.
  long max_reflection_box{50000000};
};

/**
 * \brief A simulated powder X-ray diffraction pattern.
 *
 * Intensities use neutral-atom X-ray form factors (International Tables 1992),
 * Debye-Waller factors from the structure's ADPs, and the Lorentz-polarization
 * factor.
 *
 * Not modelled: anomalous dispersion (f', f''), the K-alpha_1 / K-alpha_2
 * doublet, preferred orientation, peak asymmetry, and 2*theta-dependent
 * broadening. Intensities are therefore most reliable for light atoms away from
 * an absorption edge.
 */
class PowderPattern {
public:
  PowderPattern(std::vector<PowderPeak> peaks, double wavelength);

  /// The peaks, sorted by increasing 2*theta. Systematically absent reflections
  /// are not included, and coincident reflections have been merged.
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
                       const std::vector<PowderPeak> &reflections,
                       bool debye_waller = true);

/// Lorentz-polarization factor for an unpolarized source.
///
/// \param two_theta the Bragg angle in **radians** -- note that PowderPeak and
///        PowderPatternSettings both carry degrees.
///
/// Diverges as 1/theta^2 at low angle, which is physical: the Lorentz factor
/// really does blow up as the Bragg condition degenerates.
double lorentz_polarization(double two_theta);

/// Compute a powder pattern for the given crystal.
PowderPattern compute_powder_pattern(const Crystal &crystal,
                                     const PowderPatternSettings &settings = {});

} // namespace occ::crystal
