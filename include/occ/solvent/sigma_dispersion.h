#pragma once
#include <occ/core/linear_algebra.h>
#include <string>
#include <string_view>

namespace occ::solvent::sigma {

/// Hydrogen-bonding class of a molecule in COSMO-SAC-dsp. The pair of classes
/// selects the sign of the interaction constant.
enum class DispersionClass {
  None,          ///< no O, N or F
  Acceptor,      ///< O/N/F present, none of them carrying a hydrogen
  DonorAcceptor, ///< O/N/F present, at least one with a hydrogen
  Carboxyl,      ///< contains a -C(=O)OH group
  Water,
};

std::string_view dispersion_class_name(DispersionClass);
DispersionClass dispersion_class_from_name(std::string_view);

/// Molecular dispersion parameter for the COSMO-SAC-dsp term of Hsieh, Lin
/// and Vrabec, Fluid Phase Equilib. 367 (2014) 72.
struct Dispersion {
  double epsilon{0.0}; ///< ε/k_B, K
  DispersionClass klass{DispersionClass::None};
  bool known{false}; ///< false when the molecule holds an untabulated element

  explicit operator bool() const { return known; }
};

/// Type the atoms by perceived connectivity and average their parameters.
///
/// Connectivity uses the covalent-radius criterion of
/// `classify_hbond_segments`. Hydrogens bound to carbon carry no parameter
/// and are left out of the average rather than entered as zero, so the
/// denominator counts only typed atoms.
///
/// Parameters exist for H, C, N, O, F and Cl; any other element leaves
/// `known` false, as does a nitrogen or oxygen whose bond count is outside
/// the tabulated range. A carbon with a bond count outside 2–4 is skipped,
/// matching the reference implementation.
Dispersion dispersion_parameters(const IVec &atomic_numbers,
                                 const Mat3N &positions_bohr);

/// One-constant Margules coefficient
/// `A = w[½(ε_a+ε_b) − √(ε_a ε_b)] = ½w(√ε_a − √ε_b)²`.
///
/// `w` is 0.27027, negated for water paired with an acceptor-only or
/// carboxyl molecule and for a carboxyl paired with a non-bonding or
/// donor-acceptor one. Returns zero when either parameter is unknown or
/// non-positive, since the geometric mean is then undefined.
double dispersion_coefficient(const Dispersion &a, const Dispersion &b);

/// `ln γ^dsp` for a binary mixture: `(A x_b², A x_a²)`.
///
/// The published model is defined for two components only; `mole_fractions`
/// must have length two.
Vec dispersion_ln_gamma(const Dispersion &a, const Dispersion &b,
                        const Vec &mole_fractions);

} // namespace occ::solvent::sigma
