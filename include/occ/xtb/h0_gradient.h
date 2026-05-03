#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/gto/shell.h>
#include <occ/qm/integral_engine.h>
#include <occ/xtb/gamma.h> // for ShellTable

namespace occ::xtb {

class Gfn2Parameters;

// Analytical gradient of the SCC's `Tr(P · H0)` term plus the Pulay
// `−Tr(W · ∂S/∂R)` term. Together these are the H0 + orbital-response
// contribution to dE_SCC/dR for a converged density.
//
// Three chain-rule contributions enter:
//   (a) Combined ∂S-derivative term: Σ_μν (P_μν · X_μν − W_μν) · ∂S_μν/∂R,
//       where X_μν = H0_μν / S_μν is the (constant in S) shell-pair scaling.
//   (b) ∂Π(R_AB)/∂R: derivative of the distance polynomial in H0_off-diag.
//   (c) ∂CN/∂R chain: H0's diagonal contains -kCN·CN, and its off-diagonal
//       contains 0.5·(h_A + h_B); both depend on CN through the shell self
//       energies.
//
// Inputs (all in atomic units):
//   atoms, params, shells, basis      — geometry / parameters / AO layout
//   engine                            — IntegralEngine wrapping `basis`,
//                                       used to compute overlap derivatives
//   S      (nbf × nbf)                — overlap matrix
//   P      (nbf × nbf)                — closed-shell density (Σ over both spins)
//   W      (nbf × nbf)                — energy-weighted density
//                                       (mo.energy_weighted_density_matrix())
//   cn     (n_atoms)                  — coordination numbers
//   dcn    (n_atoms × Mat3N each)     — ∂CN_i/∂R from
//                                       gfn_coordination_numbers_with_gradient
// Output: 3 × N_atoms gradient in Hartree/Bohr.
Mat3N h0_scc_gradient(const std::vector<core::Atom> &atoms,
                      const Gfn2Parameters &params, const ShellTable &shells,
                      const gto::AOBasis &basis, qm::IntegralEngine &engine,
                      const Mat &S, const Mat &P, const Mat &W, const Vec &cn,
                      const std::vector<Mat3N> &dcn);

} // namespace occ::xtb
