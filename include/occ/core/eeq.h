#pragma once

#include <occ/core/linear_algebra.h>

namespace occ::core::charges {

/**
 * Determine the atomic coordination numbers of atoms at provided positions,
 * for the EEQ method described in https://dx.doi.org/10.1063/1.5090222
 *
 * \param atomic_numbers const IVec& of length N atomic numbers
 * \param positions const Mat3N& of dimensions (3, N) positions in Angstroms
 *
 * \returns cn Vec representing (fractional) coordination numbers for each of
 * the N atoms
 */
Vec eeq_coordination_numbers(const occ::IVec &atomic_numbers,
                             const occ::Mat3N &positions);

/**
 * Determine the atomic partial charges of atoms at provided positions,
 * constraining the net charge, using the EEQ method described in
 * https://dx.doi.org/10.1063/1.5090222
 *
 * \param atomic_numbers const IVec& of length N atomic numbers
 * \param positions const Mat3N& of dimensions (3, N) positions in Angstroms
 * \param charge double of the system net charge (default 0 i.e. neutral)
 *
 * \returns charges Vec representing partial charges at each of the N atomic
 * sites
 */
Vec eeq_partial_charges(const occ::IVec &atomic_numbers,
                        const occ::Mat3N &positions, double charge = 0.0);

/**
 * EEQ partial charges + their derivative wrt nuclear positions.
 *
 * Returns:
 *   .charges       Vec of length N — same as eeq_partial_charges
 *   .dcharges_dR   std::vector<Mat3N> of length N. dcharges_dR[i](α, j) is
 *                  ∂q_i/∂R_j^α (Bohr⁻¹). Note R_j here is in Bohr — the
 *                  derivative respects the Bohr convention used inside the
 *                  Coulomb-like A matrix, not the input Angstrom positions.
 *
 * Implementation: given A·q = -χ + (constraint), differentiate to get
 *   A · ∂q/∂R = -∂χ/∂R - ∂A/∂R · q
 * and solve once with the LU factorisation of A.
 */
struct EeqWithGradient {
  Vec charges;
  std::vector<Mat3N> dcharges_dR;
};

EeqWithGradient eeq_partial_charges_and_gradient(
    const occ::IVec &atomic_numbers, const occ::Mat3N &positions_angstrom,
    double charge = 0.0);

// =============================================================================
// Periodic (3D) variants
// =============================================================================
//
// These mirror the molecular EEQ but apply the proper lattice sum:
//   * coordination number — direct real-space lattice sum of erf-counts
//   * A matrix — Ewald-summed Coulomb between Gaussian charges
//
// The Ewald split follows multicharge / tblite (see `model/eeq.f90`):
//
//   A_ij^periodic = Σ_T [ erf(γ_ij·|R_ij + T|) − erf(α·|R_ij + T|) ] / |R_ij + T|
//                 + (4π/V) Σ_{G≠0} (1/G²) exp(−G²/4α²) cos(G·R_ij)
//                 − π/(V α²)                       (background, per-pair)
//
// The diagonal A_ii adds the standard EEQ on-site terms η_i + √(2/π)/w_i and
// the Ewald self correction −2α/√π. Σ_T and Σ_G use direct/reciprocal cutoffs
// auto-picked from `tol`. Pass `alpha_user > 0` to override the optimal α.

// Self-contained Ewald parameters for the periodic EEQ A matrix. No cached
// translation / G-vector lists — callers iterate inline via the template
// helpers in `eeq.cpp` (no per-evaluation allocation).
struct EeqEwaldData {
  occ::Mat3 lattice_bohr;       // columns = a, b, c
  occ::Mat3 reciprocal_bohr;    // 2π · (A^-1)^T
  double alpha;                 // 1/Bohr
  double erfc_cutoff;           // Bohr — direct-space cutoff
  double recip_cutoff;          // 1/Bohr — reciprocal-space cutoff
  double four_pi_over_V;        // precomputed for the recip-sum coeff
  double inv4a2;                // 1 / (4 α²)
  double background;            // −π/(V α²)
  double self_term;             // −2α/√π
};

EeqEwaldData build_eeq_ewald_data(const occ::Mat3 &lattice_bohr,
                                  double tol = 1e-10,
                                  double alpha_user = 0.0);

// Periodic EEQ coordination number. Same erf-count kernel as the molecular
// version, summed over lattice translations within `cutoff_angstrom`.
occ::Vec eeq_coordination_numbers_periodic(
    const occ::IVec &atomic_numbers, const occ::Mat3N &positions_angstrom,
    const occ::Mat3 &lattice_angstrom, double cutoff_angstrom = 25.0);

// Periodic EEQ partial charges. Builds the Ewald-summed A matrix (per
// `EeqEwaldData`), assembles X using the periodic CN, and solves the
// constrained linear system to enforce Σq = `total_charge`.
occ::Vec eeq_partial_charges_periodic(
    const occ::IVec &atomic_numbers, const occ::Mat3N &positions_angstrom,
    const occ::Mat3 &lattice_angstrom, double total_charge = 0.0,
    double tol = 1e-10);

} // namespace occ::core::charges
