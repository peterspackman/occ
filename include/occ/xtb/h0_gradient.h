#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/gto/shell.h>
#include <occ/qm/integral_engine.h>
#include <occ/xtb/gamma.h> // for ShellTable

namespace occ::xtb {

class Gfn2Parameters;

// Analytical gradient of the SCC's H0 + Pulay + V_q-via-S contributions.
// Together these are the integral-derivative contributions to dE_SCC/dR for a
// converged density (charge-only SCC; multipole pieces handled separately).
//
// The S-derivative assembly uses Z = P·X − W − ½·P·(V_s+V_t):
//   • Tr(P·∂H0_off/∂R) via S       = Tr((P·X) · ∂S/∂R)
//   • Pulay                         = −Tr(W · ∂S/∂R)
//   • Tr(P·∂V_q/∂R) via S          = ½·Σ P · ∂S/∂R · (V_s+V_t)
// xtb's `peeq_module` uses the same combination — see ~/git/xtb/src/xtb/
// hamiltonian.f90 build_dSDQH0.
//
// Two further chain-rule contributions:
//   • ∂Π(R_AB)/∂R: derivative of the distance polynomial in H0_off-diag.
//   • ∂CN/∂R: H0 diagonal has −kCN·CN; off-diagonal has 0.5·(h_A + h_B).
//
// Inputs (all in atomic units):
//   atoms, params, shells, basis      — geometry / parameters / AO layout
//   engine                            — IntegralEngine wrapping `basis`,
//                                       used to compute overlap derivatives
//   S      (nbf × nbf)                — overlap matrix
//   P      (nbf × nbf)                — closed-shell density (Σ over both spins)
//   W      (nbf × nbf)                — energy-weighted density
//                                       (mo.energy_weighted_density_matrix())
//   V_shell (n_shells)                — converged SCC Coulomb shift potential,
//                                       V_s = Σ_t J_{st} q_t (units: Hartree)
//   cn     (n_atoms)                  — coordination numbers
//   dcn    (n_atoms × Mat3N each)     — ∂CN_i/∂R from
//                                       gfn_coordination_numbers_with_gradient
// Output: 3 × N_atoms gradient in Hartree/Bohr.
Mat3N h0_scc_gradient(const std::vector<core::Atom> &atoms,
                      const Gfn2Parameters &params, const ShellTable &shells,
                      const gto::AOBasis &basis, qm::IntegralEngine &engine,
                      const Mat &S, const Mat &P, const Mat &W,
                      const Vec &V_shell, const Vec &cn,
                      const std::vector<Mat3N> &dcn);

} // namespace occ::xtb
