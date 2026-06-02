#pragma once
#include <cstddef>
#include <occ/qm/cc/thc.h>
#include <occ/qm/mo.h>

namespace occ::qm::cc {

using occ::gto::AOBasis;
using occ::qm::MolecularOrbitals;

struct ThcMP2Options {
  ThcOptions thc{};  ///< ISDF / LS-THC factorisation options (shared with CCSD)
  int n_laplace{14}; ///< Laplace quadrature points for the energy denominator
  int n_frozen{0};   ///< lowest occupied orbitals dropped from the correlation
  std::size_t memory_budget{std::size_t(1) << 30}; ///< DF B-tensor build budget
  // Skip the same-spin exchange (the O(o^2 v^2 P) part). The opposite-spin /
  // Coulomb energy is then pure O(P^3) per Laplace point -- this is the
  // genuinely fast, large-system THC win (SOS-MP2). `same_spin` is left at 0.
  bool opposite_spin_only{false};
};

struct ThcMP2Result {
  double same_spin{0.0};             ///< same-spin (αα+ββ) correlation energy
  double opposite_spin{0.0};         ///< opposite-spin (αβ) correlation energy
  double total{0.0};                 ///< same_spin + opposite_spin
  int n_isdf{0};                     ///< THC interpolation points used
  int n_laplace{0};                  ///< Laplace points used
  double laplace_max_rel_error{0.0}; ///< quadrature error over the gap range
};

/// LS-THC-MP2 correlation energy (restricted or unrestricted, dispatched on
/// mo.kind). The denominator is made separable by a Laplace quadrature so the
/// THC factors collapse the orbital sums into n_isdf x n_isdf contractions:
/// opposite-spin (Coulomb) is O(n_isdf^3) per Laplace point, same-spin exchange
/// O(n_isdf^4). `aux_basis` is the density-fitting basis used both for the
/// cheap DF reference and the LS-THC core fit (as in thc_eris).
ThcMP2Result thc_mp2(const AOBasis &basis, const AOBasis &aux_basis,
                     const MolecularOrbitals &mo,
                     const ThcMP2Options &opts = {});

} // namespace occ::qm::cc
