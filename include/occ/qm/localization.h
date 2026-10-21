#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/gto/shell.h>

namespace occ::qm {

struct Wavefunction;

/// Atomic population entering the Pipek-Mezey functional.
enum class PopulationKind {
  Mulliken, ///< q_A = sum_{mu in A} C_mu (S C)_mu, the original PM choice
  Lowdin,   ///< q_A = sum_{mu in A} (S^1/2 C)_mu^2, never negative
};

struct PipekMezeySettings {
  PopulationKind population{PopulationKind::Mulliken};
  int max_sweeps{500};
  /// converged when a Jacobi sweep raises the functional by less than this
  double tolerance{1e-10};
};

/// Orbitals localized by an orthogonal rotation of the input set.
struct LocalizedOrbitals {
  Mat C;           ///< localized orbitals (nbf x n), C_in * U
  Mat U;           ///< the orthogonal rotation (n x n)
  Mat populations; ///< atomic populations q_A of each orbital (natoms x n)
  double functional{0.0}; ///< sum_k sum_A (q_A^k)^2, maximized
  int sweeps{0};
  bool converged{false};
};

/// Pipek-Mezey localization (Pipek & Mezey, JCP 90, 4916 (1989)) of the
/// columns of C by Jacobi sweeps: maximizes sum_k sum_A (q_A^k)^2, the atomic
/// populations of each orbital, over rotations among the columns of C. The
/// span of C is unchanged; S is the AO overlap.
LocalizedOrbitals pipek_mezey(const AOBasis &basis, const Mat &S, const Mat &C,
                              const PipekMezeySettings &settings = {});

/// Pipek-Mezey localization of the occupied orbitals of a closed-shell
/// restricted wavefunction, the n_frozen lowest left out.
LocalizedOrbitals pipek_mezey(const Wavefunction &wfn, int n_frozen = 0,
                              const PipekMezeySettings &settings = {});

} // namespace occ::qm
