#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/units.h>
#include <occ/dft/molecular_grid.h>
#include <occ/dma/add_qlm.h>
#include <occ/dma/binomial.h>
#include <occ/dma/dma.h>
#include <occ/dma/gauss_hermite.h>
#include <occ/dma/moveq.h>
#include <occ/dma/mult.h>
#include <occ/dma/shiftq.h>
#include <occ/gto/density.h>
#include <occ/gto/shell_order.h>
#include <occ/io/conversion.h>
#include <occ/qm/hf.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace occ::dma {

// Helper to get cartesian powers for basis functions
inline IVec3 get_powers(int bf_idx, int l) {

  IVec3 result(0, 0, 0);

  int count = 0;
  auto f = [&result, &bf_idx, &count](int i, int j, int k, int l) {
    if (count == bf_idx)
      result = IVec3(i, j, k);
    count++;
  };
  occ::gto::iterate_over_shell<true>(f, l);
  return result;
}

inline double get_normalization_factor(int l, int m, int n) {
  int angular_momenta = l + m + n;
  if (angular_momenta == 2 && ((l == 1) || (m == 1) || (n == 1))) {
    return std::sqrt(3.0);
  }
  if (angular_momenta == 3 && ((l == 2) || (m == 2) || (n == 2))) {
    return std::sqrt(5.0);
  }
  if (angular_momenta == 3 && ((l == 1) && (m == 1) && (n == 1))) {
    return std::sqrt(15.0);
  }

  return 1.0;
}

// Apply scaling factors for angular momentum functions

/**
 * @brief Calculate multipole moments and shift them to the nearest site.
 *
 * This is a direct implementation of the dmaqlm subroutine from the GDMA
 * Fortran code. It calculates multipole moments for all pairs of basis
 * functions and shifts them to appropriate expansion sites.
 *
 * @param wfn Wavefunction containing basis set and density matrix
 * @param max_rank Maximum multipole rank to calculate
 * @param verbose Controls output verbosity
 * @param include_nuclei Whether to include nuclear contributions in the
 * calculation
 * @param bigexp Threshold for switching between analytical and grid-based
 * methods
 * @return std::vector<Mult> Multipole moments for each site
 */
std::vector<Mult> dmaqlm(const qm::AOBasis &basis,
                         const qm::MolecularOrbitals &mo, const DMASites &sites,
                         const DMASettings &settings) {

  log::debug("Starting Distributed Multipole Analysis");
  log::debug("Site radii : {}\n", format_matrix(sites.radii));
  log::debug("Site limits: {}\n", format_matrix(sites.limits, "{}"));

  qm::HartreeFock hf(basis);
  Mat bf_norm =
      hf.compute_overlap_matrix().diagonal().array().sqrt().matrix().asDiagonal();
  Mat D = 2 * mo.D;
  // Transform the density matrix
  D = bf_norm * D * bf_norm;

  // Get basis set information
  const auto &shells = basis.shells();
  const auto n_shells = shells.size();
  const auto &shell_to_atom = basis.shell_to_atom();
  const auto &atom_to_shells = basis.atom_to_shell();
  const auto &first_bf = basis.first_bf();

  // Setup site data (initially just atoms)
  std::vector<Mult> site_multipoles(sites.size());

  // Threshold for numerical significance
  const double tol = 2.30258 * 18;

  // Flag for quadrature-based calculation
  bool do_quadrature = false;
  Vec rho;

  dft::GridSettings grid_settings;
  grid_settings.treutler_alrichs_adjustment = false;

  auto grid_gen = dft::MolecularGrid(basis, grid_settings);
  grid_gen.set_atomic_radii(sites.radii);
  grid_gen.populate_molecular_grid_points();

  const auto &grid = grid_gen.get_molecular_grid_points();
  const auto &grid_points = grid.points();

  double etol = 36.0; // Threshold for density calculation

  // Loop over atoms
  for (int atom_i = 0; atom_i < sites.atoms.size(); atom_i++) {
    const Vec3 pos_i = sites.atoms[atom_i].position();

    // Initialize a Mult object to store temporary multipoles
    Mult qt;
    qt.q = Vec::Zero(settings.max_rank * settings.max_rank +
                     2 * settings.max_rank + 1);

    // Include nuclear charges if requested
    if (settings.include_nuclei) {
      qt.Q00() = sites.atoms[atom_i].atomic_number;
      moveq(pos_i, qt, sites.positions, sites.radii, sites.limits,
            site_multipoles, settings.max_rank);
    }

    const auto &i_shells = atom_to_shells[atom_i];

    // Loop over shells for atom i
    for (int i_shell_idx : i_shells) {
      const auto &shell_i = shells[i_shell_idx];
      const int l_i = shell_i.l;
      const Vec3 &origin_i = shell_i.origin;

      // Get basis function ranges for shell i
      const int bf_i_start = first_bf[i_shell_idx];
      const int bf_i_end = (i_shell_idx + 1 < n_shells)
                               ? first_bf[i_shell_idx + 1]
                               : basis.nbf();

      // Loop over atoms j (up to i to avoid double counting)
      for (int atom_j = 0; atom_j <= atom_i; atom_j++) {
        const bool i_equals_j = (atom_i == atom_j);
        const Vec3 &pos_j = sites.atoms[atom_j].position();

        // Calculate atom separation
        const Vec3 r_ij = pos_i - pos_j;
        const double r_ij_squared = r_ij.squaredNorm();

        // Find shells for atom j
        const auto &j_shells = atom_to_shells[atom_j];

        // Loop over shells for atom j
        for (int j_shell_idx : j_shells) {
          // If atoms are the same, only process shell pairs i_shell >= j_shell
          // to avoid double counting
          if (i_equals_j && j_shell_idx > i_shell_idx)
            continue;

          const auto &shell_j = shells[j_shell_idx];
          const int l_j = shell_j.l;
          const Vec3 &origin_j = shell_j.origin;

          // Get basis function ranges for shell j
          const int bf_j_start = first_bf[j_shell_idx];
          const int bf_j_end = (j_shell_idx + 1 < n_shells)
                                   ? first_bf[j_shell_idx + 1]
                                   : basis.nbf();

          const bool i_shell_equals_j_shell = (i_shell_idx == j_shell_idx);

          Mat d_block = D.block(first_bf[i_shell_idx], first_bf[j_shell_idx],
                                shell_i.size(), shell_j.size());

          for (int bf_i_idx = 0; bf_i_idx < shell_i.size(); bf_i_idx++) {
            IVec3 i_powers = get_powers(bf_i_idx, l_i);
            double norm_i =
                get_normalization_factor(i_powers(0), i_powers(1), i_powers(2));

            // Get appropriate contraction coefficient
            for (int bf_j_idx = 0; bf_j_idx < shell_j.size(); bf_j_idx++) {
              IVec3 j_powers = get_powers(bf_j_idx, l_j);
              double norm_j = get_normalization_factor(j_powers(0), j_powers(1),
                                                       j_powers(2));
              d_block(bf_i_idx, bf_j_idx) *= norm_i * norm_j;
            }
          }

          // Loop over primitives in shell i
          for (int i_prim = 0; i_prim < shell_i.num_primitives(); i_prim++) {
            const double alpha_i = shell_i.exponents[i_prim];

            // If shells are the same, only loop up to i_prim to avoid double
            // counting
            int j_prim_max =
                i_shell_equals_j_shell ? i_prim + 1 : shell_j.num_primitives();

            for (int j_prim = 0; j_prim < j_prim_max; j_prim++) {
              const double alpha_j = shell_j.exponents[j_prim];
              const double alpha_sum = alpha_i + alpha_j;

              // Calculate shell separation
              const Vec3 r_shell_ij = origin_j - origin_i;
              const double shell_r2 = r_shell_ij.squaredNorm();

              // Skip if shell distance is too large or exponential term is
              // negligible
              const double dum = alpha_j * alpha_i * shell_r2 / alpha_sum;
              if (dum > tol)
                continue;

              // Factor for exponential term
              double fac = std::exp(-dum);

              // Double the factor if primitives or shells are different
              // (following dma.f90)
              if (i_prim != j_prim || !(i_shell_equals_j_shell)) {
                fac *= 2.0;
              }

              if ((alpha_i + alpha_j) > settings.big_exponent) {
                // Use analytical method for large exponents

                // P is the gaussian product center
                const double p = alpha_j / alpha_sum;
                const Vec3 P = origin_i + p * r_shell_ij;

                // Get center coordinates relative to product center
                const Vec3 A = origin_i - P;
                const Vec3 B = origin_j - P;

                // Maximum rank needed for multipole integrals
                const int lq = l_i + l_j;
                const int nq = lq + 1; // Number of quadrature points needed

                // Get Gauss-Hermite points and weights
                const auto points = gauss_hermite_points(nq);
                const auto weights = gauss_hermite_weights(nq);

                // Scaling factor for Gaussian-Hermite quadrature
                const double t = std::sqrt(1.0 / alpha_sum);

                // Initialize arrays for integrals with correct dimensions
                Eigen::Tensor<double, 3> gx(nq + 1, l_i + 1, l_j + 1);
                Eigen::Tensor<double, 3> gy(nq + 1, l_i + 1, l_j + 1);
                Eigen::Tensor<double, 3> gz(nq + 1, l_i + 1, l_j + 1);

                gx.setZero();
                gy.setZero();
                gz.setZero();

                // Perform Gauss-Hermite quadrature
                for (int k = 0; k < points.size(); k++) {
                  const double s = points(k) * t;
                  const double g = weights(k) * t;

                  // Displacements from s to centers
                  const double xas = s - A(0);
                  const double yas = s - A(1);
                  const double zas = s - A(2);

                  const double xbs = s - B(0);
                  const double ybs = s - B(1);
                  const double zbs = s - B(2);

                  // Following the Fortran implementation pattern to accumulate
                  // integrals
                  double pax = g, pay = g, paz = g;

                  for (int ia = 0; ia <= l_i; ia++) {
                    double px = pax, py = pay, pz = paz;

                    for (int ib = 0; ib <= l_j; ib++) {
                      double pq = 1.0;

                      for (int iq = 0; iq <= nq; iq++) {
                        gx(iq, ia, ib) += px * pq;
                        gy(iq, ia, ib) += py * pq;
                        gz(iq, ia, ib) += pz * pq;
                        pq *= s;
                      }

                      px *= xbs;
                      py *= ybs;
                      pz *= zbs;
                    }

                    pax *= xas;
                    pay *= yas;
                    paz *= zas;
                  }
                }

                // Initialize temporary multipole for this primitive pair
                qt.q.setZero();

                // Get contraction coefficients
                const double ci = shell_i.coeff_normalized_dma(0, i_prim);
                const double cj = shell_j.coeff_normalized_dma(0, j_prim);

                // Process all basis function combinations
                for (int bf_i_idx = 0; bf_i_idx < shell_i.size(); bf_i_idx++) {
                  IVec3 i_powers = get_powers(bf_i_idx, l_i);

                  // Get appropriate contraction coefficient
                  for (int bf_j_idx = 0; bf_j_idx < shell_j.size();
                       bf_j_idx++) {
                    IVec3 j_powers = get_powers(bf_j_idx, l_j);

                    // Skip if density matrix element is too small
                    if (std::abs(d_block(bf_i_idx, bf_j_idx)) < 1e-12)
                      continue;

                    // Calculate prefactor (negative sign as in dmaqlm)
                    double f = -fac * ci * cj * d_block(bf_i_idx, bf_j_idx);

                    // Create 1D vectors from tensor slices for addqlm
                    Vec gx_vec(nq + 1), gy_vec(nq + 1), gz_vec(nq + 1);
                    for (int n = 0; n <= nq; n++) {
                      gx_vec(n) = gx(n, i_powers(0), j_powers(0));
                      gy_vec(n) = gy(n, i_powers(1), j_powers(1));
                      gz_vec(n) = gz(n, i_powers(2), j_powers(2));
                    }

                    // Add contribution to multipoles
                    addqlm(lq, settings.max_rank, f, gx_vec, gy_vec, gz_vec,
                           qt);
                  }
                }

                // Move multipoles to nearest sites
                moveq(P, qt, sites.positions, sites.radii, sites.limits,
                      site_multipoles, settings.max_rank);

              } else {
                // For small exponents, use numerical integration
                if (!do_quadrature) {
                  // Initialize density array
                  rho = Vec::Zero(grid.num_points());

                  do_quadrature = true;
                }

                // Calculate center P of overlap distribution
                const double p = alpha_j / alpha_sum;
                const Vec3 P = origin_i + p * r_shell_ij;

                // Get contraction coefficients
                const double ci = shell_i.coeff_normalized_dma(0, i_prim);
                const double cj = shell_j.coeff_normalized_dma(0, j_prim);

                // Process grid points
                for (int p = 0; p < grid.num_points(); p++) {
                  const Vec3 &grid_pos = grid_points.col(p);

                  // Calculate distance from grid point to product center
                  const double dist2 = (grid_pos - P).squaredNorm();

                  // Skip if too far from product center (exponential too small)
                  const double e = alpha_sum * dist2;
                  if (e > etol)
                    continue;

                  // Exponential factor
                  const double exp_factor = std::exp(-e);

                  // Displacements from grid point to centers
                  const Vec3 r_i = grid_pos - origin_i;
                  const Vec3 r_j = grid_pos - origin_j;

                  // Precompute powers of coordinates for efficiency
                  std::vector<double> x_i_powers(l_i + 1, 1.0);
                  std::vector<double> y_i_powers(l_i + 1, 1.0);
                  std::vector<double> z_i_powers(l_i + 1, 1.0);

                  std::vector<double> x_j_powers(l_j + 1, 1.0);
                  std::vector<double> y_j_powers(l_j + 1, 1.0);
                  std::vector<double> z_j_powers(l_j + 1, 1.0);

                  // Calculate powers of coordinates
                  for (int l = 1; l <= l_i; l++) {
                    x_i_powers[l] = x_i_powers[l - 1] * r_i(0);
                    y_i_powers[l] = y_i_powers[l - 1] * r_i(1);
                    z_i_powers[l] = z_i_powers[l - 1] * r_i(2);
                  }

                  for (int l = 1; l <= l_j; l++) {
                    x_j_powers[l] = x_j_powers[l - 1] * r_j(0);
                    y_j_powers[l] = y_j_powers[l - 1] * r_j(1);
                    z_j_powers[l] = z_j_powers[l - 1] * r_j(2);
                  }

                  // Loop through all basis function combinations
                  for (int bf_i_idx = 0; bf_i_idx < shell_i.size();
                       bf_i_idx++) {
                    IVec3 i_powers = get_powers(bf_i_idx, l_i);

                    // Get appropriate contraction coefficient
                    for (int bf_j_idx = 0; bf_j_idx < shell_j.size();
                         bf_j_idx++) {
                      IVec3 j_powers = get_powers(bf_j_idx, l_j);

                      // Skip if density matrix element is too small
                      if (std::abs(d_block(bf_i_idx, bf_j_idx)) < 1e-12)
                        continue;

                      // Calculate prefactor (negative sign as in dmaqlm)
                      double f = -fac * ci * cj * d_block(bf_i_idx, bf_j_idx);

                      // Calculate basis function product at grid point
                      double bf_product =
                          exp_factor * x_i_powers[i_powers[0]] *
                          y_i_powers[i_powers[1]] * z_i_powers[i_powers[2]] *
                          x_j_powers[j_powers[0]] * y_j_powers[j_powers[1]] *
                          z_j_powers[j_powers[2]];

                      // Add contribution to density at this grid point
                      rho(p) += f * bf_product;
                    }
                  }
                }
              } // end j_prim loop
            } // end i_prim loop
          } // end j_shell loop
        } // end atom_j loop
      } // end i_shell loop
    } // end atom_i loop
  } // end atom_i loop

  // If we did numerical integration, process the grid density
  if (do_quadrature) {

    Mult qt;
    qt.q = Vec::Zero(settings.max_rank * settings.max_rank +
                     2 * settings.max_rank + 1);
    const auto &grid_weights = grid.weights();
    // Process the grid density contributions to multipoles
    std::vector<Mult> site_contributions(sites.size());

    // Initialize temporary arrays for coordinates and their powers
    Vec ggx = Vec::Zero(settings.max_rank + 1);
    Vec ggy = Vec::Zero(settings.max_rank + 1);
    Vec ggz = Vec::Zero(settings.max_rank + 1);

    const auto &atom_blocks = grid.atom_blocks();
    for (int site_index = 0; site_index < sites.size(); site_index++) {
      Vec3 pos_i = sites.positions.col(site_index);
      qt.q.setZero();
      const auto &[p_start, num_points] = atom_blocks[site_index];
      // Process each grid point
      for (int p = p_start; p < p_start + num_points; p++) {

        // Calculate position relative to site
        const Vec3 &grid_pos = grid_points.col(p);
        const Vec3 rel_pos = grid_pos - sites.positions.col(site_index);

        // Initialize power arrays
        ggx(0) = 1.0;
        ggy(0) = 1.0;
        ggz(0) = 1.0;

        // Calculate powers of coordinates
        for (int l = 1; l <= settings.max_rank; l++) {
          ggx(l) = ggx(l - 1) * rel_pos(0);
          ggy(l) = ggy(l - 1) * rel_pos(1);
          ggz(l) = ggz(l - 1) * rel_pos(2);
        }

        // Get grid weight
        double weight = grid_weights(p);

        // Add contribution to multipoles for this site
        addqlm(settings.max_rank, settings.max_rank, weight * rho(p), ggx, ggy,
               ggz, qt);
      }

      moveq(pos_i, qt, sites.positions, sites.radii, sites.limits,
            site_multipoles, settings.max_rank);
    }
  }

  // Return the calculated multipoles
  return site_multipoles;
}
} // namespace occ::dma
