#pragma once
#include <optional>
#include <string_view>

namespace occ::crystal {

/**
 * \brief A neutral-atom or ion X-ray form factor from Waasmaier & Kirfel
 * (1995).
 *
 * Acta Cryst. A51, 416-431, doi:10.1107/S0108767394013292: five Gaussians plus
 * a constant, valid to sin(theta)/lambda = 6 A^-1 (International Tables' own
 * fit only reaches 2).
 *
 * \note The coefficients are held in single precision. That is not an
 * oversight: cctbx stores this table as `float`, and matching its structure
 * factors bit for bit means rounding the same way. The published values carry
 * more digits than a float holds, so a handful of the literals below differ
 * from the paper in the eighth significant figure.
 */
struct XrayFormFactor {
  float a[5];
  float b[5];
  float c;

  /**
   * \brief f(s), in electrons.
   * \param stol_sq (sin(theta)/lambda)^2 in A^-2, i.e. |G|^2/4.
   */
  double at_stol_sq(double stol_sq) const;
};

/**
 * \brief Look up a form factor by scattering-type label, e.g. "C", "Na1+",
 * "O2-".
 *
 * Labels follow the cctbx/International Tables spelling: the element symbol,
 * optionally followed by the charge as digit-then-sign. The table also holds
 * the valence-state entries "Cval" and "Sival".
 *
 * "H" is not Waasmaier & Kirfel's hydrogen but cctbx's fit to the more
 * contracted Stewart, Davidson & Simpson (1965) bonded hydrogen, which is what
 * cctbx and rinse-descriptor scatter hydrogen with. The free atom is "Hiso".
 *
 * \returns nullopt if the label is not in the table.
 */
std::optional<XrayFormFactor> xray_form_factor(std::string_view label);

/// Look up the neutral atom for an atomic number (1-98); nullopt if out of
/// range. Hydrogen is the bonded "H".
std::optional<XrayFormFactor> xray_form_factor(int atomic_number);

} // namespace occ::crystal
