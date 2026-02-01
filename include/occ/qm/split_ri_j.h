#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/gto/gto.h>
#include <occ/qm/mo.h>
#include <memory>
#include <vector>

namespace occ::qm {

using ShellPairList = std::vector<std::vector<size_t>>;

/// Split-RI-J: Coulomb matrix via Hermite Gaussian basis (Neese 2003)
///
/// This implementation uses the McMurchie-Davidson (MMD) scheme to work
/// in Hermite Gaussian basis, computing Coulomb matrices without explicitly
/// forming the 3-center integrals (μν|P).
///
/// Algorithm (Scheme 4 from Neese 2003):
///   Forward pass:
///     1. Transform density to Hermite basis: X = E_ab^T × D
///     2. Contract with R-integrals: Y = R^T × X
///     3. Transform to aux basis: g = E_c × Y
///   Solve: d = V^{-1} × g
///   Backward pass (reusing same R-integrals):
///     1. Transform d to Hermite basis: T = E_c^T × d
///     2. Contract with R-integrals: U = R × T
///     3. Transform to orbital basis: J = E_ab × U
///
/// Reference: Neese, F. (2003). An improvement of the resolution of the
/// identity approximation for the formation of the Coulomb matrix.
/// J. Comput. Chem. 24, 1740-1747.
class SplitRIJ {
public:
  /// Construct Split-RI-J engine
  /// @param ao_basis     Atomic orbital basis
  /// @param aux_basis    Auxiliary (density fitting) basis
  /// @param shellpairs   Significant shell pair list from IntegralEngine
  /// @param schwarz      Optional Schwarz screening matrix (shell-pair indexed)
  SplitRIJ(const gto::AOBasis& ao_basis, const gto::AOBasis& aux_basis,
           const ShellPairList& shellpairs, const Mat& schwarz = Mat());

  ~SplitRIJ();

  // Non-copyable, movable
  SplitRIJ(const SplitRIJ&) = delete;
  SplitRIJ& operator=(const SplitRIJ&) = delete;
  SplitRIJ(SplitRIJ&&) noexcept;
  SplitRIJ& operator=(SplitRIJ&&) noexcept;

  /// Compute Coulomb matrix from molecular orbitals
  /// @param mo  Molecular orbitals containing density matrix
  /// @return J matrix (nbf × nbf)
  Mat coulomb(const MolecularOrbitals& mo) const;

  /// Compute Coulomb matrix from density matrix directly
  /// @param D  Density matrix (nbf × nbf)
  /// @return J matrix (nbf × nbf)
  Mat coulomb_from_density(const Mat& D) const;

  /// Get the auxiliary basis
  const gto::AOBasis& aux_basis() const;

  /// Get the AO basis
  const gto::AOBasis& ao_basis() const;

  /// Get number of auxiliary basis functions
  size_t naux() const;

  /// Get number of AO basis functions
  size_t nbf() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace occ::qm
