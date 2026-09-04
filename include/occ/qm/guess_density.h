#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/gto/shell.h>
#include <vector>

namespace occ::qm::guess {

using gto::AOBasis;

/// One subshell of a neutral atom's minimal-basis configuration.
struct SubshellOccupation {
  int n{0};            ///< principal quantum number
  int l{0};            ///< angular momentum
  double electrons{0}; ///< electrons in the whole subshell
};

/// A neutral atom's minimal-basis configuration, in filling order.
///
/// Labelled by (n, l) rather than left as a flat vector, because the minimal
/// basis does not list its shells in filling order: STO-3G gives iron
/// 1s 2s 2p 3s 3p 4s 4p 3d, with the d shell last. Anything that places these
/// occupations has to match them up by subshell, not by position.
std::vector<SubshellOccupation> minimal_basis_subshell_occupations(int Z);

/// Spin multiplicity 2S+1 of the neutral atom's ground state, for Z in
/// 1..118; 0 for anything outside that range.
///
/// These are the experimental ground-state terms, so they include the cases
/// Hund's rule alone gets wrong -- chromium and copper take an electron from
/// the s shell to half-fill or fill the d shell, palladium empties its s
/// shell entirely. An effective core potential removes only closed shells, so
/// the same multiplicity applies to the valence-only atom: gold keeps its one
/// unpaired 6s electron whether or not its core is replaced.
int ground_state_multiplicity(int Z);

/// Superposition of atomic potentials, as a one-electron matrix in `basis`.
///
/// Uses whichever fit `AOBasis::load_sap_basis` provides; there is no choice
/// to make here, and the parameter this once took named a basis it did not
/// actually load.
Mat compute_sap_matrix(const std::vector<occ::core::Atom> &atoms,
                       const AOBasis &basis);

} // namespace occ::qm::guess
