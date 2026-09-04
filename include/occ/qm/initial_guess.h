#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/gto/shell.h>
#include <occ/qm/guess_kind.h>

namespace occ::qm {

/// What a guess needs to know about the system.
///
/// Deliberately not the SCF procedure: none of these guesses needs a Fock
/// build, so none of them should have to name one. That keeps the whole
/// module out of the SCF template.
struct GuessRequest {
  /// The orbital basis, carrying the atoms and any ECP.
  const gto::AOBasis &basis;
  /// Net charge of the system.
  int charge{0};
  /// Active electrons, i.e. with any ECP-replaced core already removed.
  int num_electrons{0};
};

/// The starting point a guess produces.
///
/// A guess perturbs the core Hamiltonian in one of two ways, so it hands back
/// either a density to build a Fock matrix from or a one-electron potential to
/// add. `Core` supplies neither, leaving `F = H`.
struct Guess {
  GuessKind kind{GuessKind::Core};

  /// Spin-summed density normalised to half the electron count -- occ's usual
  /// convention -- expressed in `density_basis`. Empty for `Core` and `Sap`.
  Mat density;

  /// The basis `density` is expressed in. `Soad` builds in the minimal basis,
  /// so the caller has to project; every other kind uses the orbital basis,
  /// where the projection is the identity.
  gto::AOBasis density_basis;

  /// Whether `density` is block-diagonal over the shells of `density_basis`,
  /// which the mixed-basis Fock build uses to skip integrals. True only for
  /// `Soad`, whose occupations are diagonal by construction.
  bool density_is_shell_diagonal{false};

  /// One-electron potential to add to the core Hamiltonian, in the orbital
  /// basis. Only `Sap` sets it, and it sets no density.
  Mat potential;
};

/// Whether the shipped minimal basis has shells for every atom in `basis`.
///
/// `build_sto3g_shells` drops elements it has no entry for rather than
/// failing, so loading the minimal basis is not itself a coverage test: an
/// uncovered atom simply comes back carrying no shells.
bool minimal_basis_covers(const gto::AOBasis &basis);

/// Resolve `Auto` into a concrete kind; anything else passes through
/// unchanged, so an explicit request is honoured even where it is a poor
/// choice. Logs the reason whenever `Auto` picks something other than SOAD.
GuessKind select_guess(GuessKind requested, const gto::AOBasis &basis);

/// Build `kind` for `request`.
///
/// Falls back to `Core` -- an empty `Guess` -- when the guess cannot be built
/// for this system, having said why in the log. Every caller therefore gets a
/// usable starting point.
Guess build_guess(GuessKind kind, const GuessRequest &request);

} // namespace occ::qm
