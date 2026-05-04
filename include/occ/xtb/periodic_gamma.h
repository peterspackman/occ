#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/xtb/periodic.h>
#include <vector>

namespace occ::xtb {

class Gfn2Parameters;
struct ShellTable;

// Pre-computed Ewald data for a periodic system.
//
// The Klopman-Ohno γ used in GFN2 behaves like 1/R at large distances, so the
// shell-resolved γ matrix Σ_T γ(R+T) is conditionally convergent. We split
//   Σ_T γ(R+T) = Σ_T [γ(R+T) - 1/|R+T|]    (residual, decays as 1/R^3)
//              + Σ_T 1/|R+T|                 (Coulomb lattice sum, Ewald-summed)
//
// The two pieces have very different decay rates, so they need separate
// real-space cutoffs:
//   - `erfc_cutoff` controls the Ewald erfc(αR)/R real-space sum
//     (exponential, α-dependent — small).
//   - `residual_cutoff` controls Σ_T [γ(R+T) - 1/|R+T|] (algebraic 1/R^3,
//     α-independent — large).
//
// For a charge-neutral cell (Σ q = 0), the residual sum benefits from a
// dipole-cancellation that makes it converge absolutely as 1/R^4, so a
// 50-80 Bohr residual cutoff gives ~1e-7 accuracy in energies.
//
// `alpha` is the Ewald screening parameter (1/Bohr). Auto-pick uses
// alpha = sqrt(pi) / V^{1/3}, which roughly balances real-space erfc work
// against reciprocal work for typical cell sizes.
struct EwaldGammaData {
  double alpha;                       // 1/Bohr
  double erfc_cutoff;                 // Bohr — for erfc(αR)/R sum
  double residual_cutoff;             // Bohr — for γ(R) - 1/R sum
  double recip_cutoff;                // 1/Bohr
  std::vector<LatticeImage> images;   // direct-lattice translations within max(erfc_cutoff, residual_cutoff)
  std::vector<Vec3> g_vectors;        // G with 0 < |G| <= recip_cutoff
  std::vector<double> g_coeffs;       // (4π/V) (1/G²) exp(-G²/4α²)
  double background;                  // -π/(V α²)  (per-pair G=0 contribution)
  double self_term;                   // -2α/√π  (per-pair self contribution at R=0)
};

// Build Ewald data with default heuristics. `tol` controls erfc / reciprocal
// cutoffs (exponential decay). `residual_cutoff` is set independently — pass
// 0 to use the default (60 Bohr, gives ~1e-7 truncation for charge-neutral
// densities). Pass alpha_user > 0 to override the α auto-pick.
EwaldGammaData build_ewald_data(const PeriodicSystem &sys, double tol = 1e-10,
                                double alpha_user = 0.0,
                                double residual_cutoff = 0.0);

// Lattice-summed shell-resolved γ matrix at the Γ point.
//
//   γ^per_{ij} = Σ_T γ_{l_i l_j}(R_i - R_j - T)
//
// For same-shell-on-same-atom (i==j), the on-site contribution is the KO limit
// γ(0) = η_i (and ½(η_i+η_j) for different shells on the same atom).
//
// Result is real, symmetric, n_shells × n_shells where shells are those of the
// central cell.
Mat periodic_klopman_ohno_gamma(const PeriodicSystem &sys,
                                const ShellTable &shells,
                                const Gfn2Parameters &params,
                                const EwaldGammaData &ewald);

// Direct (no-Ewald) lattice sum, intended only as an oracle for tests. Uses
// the residual subtraction trick to make 1/R convergent: subtracts the leading
// 1/R from γ and adds a brute-force `Σ_T 1/|R+T|` over a large explicit
// translation list. NOT suitable for production (slow, conditionally
// convergent).
Mat periodic_klopman_ohno_gamma_direct(const PeriodicSystem &sys,
                                        const ShellTable &shells,
                                        const Gfn2Parameters &params,
                                        double cutoff_bohr);

} // namespace occ::xtb
