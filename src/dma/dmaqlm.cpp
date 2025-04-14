#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <occ/dft/grid.h>
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
  occ::gto::iterate_over_shell<true, gto::ShellOrder::Gaussian>(f, l);
  return result;
}

inline double get_normalization_factor(int l, int m, int n) {
  int angular_momenta = l + m + n;
  using occ::util::double_factorial_2n_1;
  double result =
      std::sqrt(double_factorial_2n_1(angular_momenta) /
                (double_factorial_2n_1(l) * double_factorial_2n_1(m) *
                 double_factorial_2n_1(n)));
  return result;
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
std::vector<Mult> dmaqlm(const occ::qm::Wavefunction &wfn, int max_rank = 4,
                         bool include_nuclei = true, double bigexp = 4.0) {

  fmt::print("Starting Distributed Multipole Analysis\n");
  fmt::print("    Atoms   Shells Primitives            Position     Multipole "
             "contributions ...\n");

  // Get molecule data from wavefunction
  const auto &atoms = wfn.atoms;
  const auto &positions = wfn.positions();
  const size_t n_atoms = atoms.size();

  // Get density matrix
  auto mo = occ::io::conversion::orb::to_gaussian_order(wfn.basis, wfn.mo);
  mo.update_density_matrix();
  const Mat D = 2 * mo.D;
  const Vec S = wfn.overlap_matrix().diagonal().array().sqrt();
  fmt::print("Overlap\n{}\n", format_matrix(S));

  // Get basis set information
  const auto &basis = wfn.basis;
  const auto &shells = basis.shells();
  const auto n_shells = shells.size();
  const auto &shell_to_atom = basis.shell_to_atom();
  const auto &atom_to_shells = basis.atom_to_shell();
  const auto &first_bf = basis.first_bf();

  // Setup site data (initially just atoms)
  Mat3N sites = positions;
  std::vector<Mult> site_multipoles(n_atoms);
  Vec site_radii = Vec::Ones(n_atoms) * 0.5; // Default radius of 0.5 Å

  constexpr char shell_labels[] = "spdfghikmnoqrtuvwxyz";
  fmt::print("Total number of primitives required\n");
  // Adjust radii for hydrogens
  int prim = 1;
  for (size_t i = 0; i < n_atoms; i++) {
    if (atoms[i].atomic_number == 1) {
      site_radii(i) = 0.325; // Hydrogen atoms get smaller radius
    }

    // Initialize multipoles at each site
    site_multipoles[i].q = Vec::Zero(max_rank * max_rank + 2 * max_rank + 1);
    fmt::print("{}\n", core::Element(atoms[i].atomic_number).symbol());
    for (const auto &shell_idx : atom_to_shells[i]) {
      const auto &sh = shells[shell_idx];
      fmt::print("Shell {}   {}\n", shell_idx + 1, shell_labels[sh.l]);
      for (int i = 0; i < sh.num_primitives(); i++) {
        fmt::print("{:10d}{:16.8f}{:14.8f}\n", prim, sh.exponents(i),
                   sh.coeff_normalized_dma(0, i));
        prim++;
      }
    }
  }
    site_radii <<    1.2283219807815948, 0.61416099039079741, 0.61416099039079741;
    fmt::print("site_radii:{}\n", format_matrix(site_radii));

  IVec site_limits = IVec::Constant(n_atoms, max_rank);

  // Threshold for numerical significance
  const double tol = 2.30258 * 18;

  // Flag for quadrature-based calculation
  bool do_quadrature = false;
  Vec rho;

  auto grid_gen = dft::MolecularGrid(basis, {});
  grid_gen.populate_molecular_grid_points();
  const auto &grid = grid_gen.get_molecular_grid_points();
  const auto &grid_points = grid.points();
  std::vector<int> point_to_site_map;

  double etol = 36.0; // Threshold for density calculation

  // Loop over atoms
  for (int atom_i = 0; atom_i < n_atoms; atom_i++) {
    const Vec3 &pos_i = positions.col(atom_i);

    // Initialize a Mult object to store temporary multipoles
    Mult qt;
    qt.q = Vec::Zero(max_rank * max_rank + 2 * max_rank + 1);

    // Include nuclear charges if requested
    if (include_nuclei) {
      qt.Q00() = atoms[atom_i].atomic_number;
      moveq(pos_i, qt, sites, site_radii, site_limits, site_multipoles,
            max_rank);
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
        const Vec3 &pos_j = positions.col(atom_j);

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

          Vec S_block_i =
              S.block(first_bf[i_shell_idx], 0, shell_i.size(), 1).array();
          Vec S_block_j =
              S.block(first_bf[j_shell_idx], 0, shell_j.size(), 1).array();
          Mat prod = S_block_i * S_block_j.transpose();
          // Extract density matrix block for these shells
          Mat d_block = D.block(first_bf[i_shell_idx], first_bf[j_shell_idx],
                                shell_i.size(), shell_j.size());

          d_block = D.block(first_bf[i_shell_idx], first_bf[j_shell_idx],
                            shell_i.size(), shell_j.size())
                        .array();
                 //   S_block_i(0) * S_block_j(0);
          fmt::print(" d_block sum\n{:2d}{:2d}\n{:9.5f}\n", l_i, l_j, d_block.sum());

          fmt::print("{:5d}{:5d}{:5d}\n", shell_i.num_primitives(), i_shell_idx,
                     j_shell_idx);
          // Loop over primitives in shell i
          for (int i_prim = 0; i_prim < shell_i.num_primitives(); i_prim++) {
            const double alpha_i = shell_i.exponents[i_prim];

            // If shells are the same, only loop up to i_prim to avoid double
            // counting
            int j_prim_max =
                i_shell_equals_j_shell ? i_prim + 1 : shell_j.num_primitives();

            fmt::print("{:5d}\n", j_prim_max);
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

              if (alpha_i + alpha_j > bigexp) {
                // Use analytical method for large exponents

                // P is the gaussian product center
                const double p = alpha_j / alpha_sum;
                const Vec3 P = origin_i + p * r_shell_ij;

                fmt::print("{:5d}{:4d}{:5d}{:4d}{:5d}{:4d}   "
                           "{:10.5f}{:10.5f}{:10.5f}\n",
                           atom_i + 1, atom_j + 1, i_shell_idx + 1,
                           j_shell_idx + 1, i_prim + 1, j_prim + 1, P(0), P(1),
                           P(2));

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
                    fmt::print(
                        "{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:5d}{:5d}{:5d}{:5d}{:5d}{:5d}\n",
                        fac, f, d_block(bf_i_idx, bf_j_idx), gx_vec.sum(), gy_vec.sum(), gz_vec.sum(),
                        i_powers(0), i_powers(1), i_powers(2), j_powers(0), j_powers(1), j_powers(2));

                    // Add contribution to multipoles
                    addqlm(lq, max_rank, f, gx_vec, gy_vec, gz_vec, qt);
                  }
                }

                /*
                fmt::print("{:5d}{:5d} {:5d}{:5d} {:5d}{:5d} {:11.4f} {:11.4f} "
                          "{:11.4f} {}\n",
                          atom_i, atom_j, i_shell_idx, j_shell_idx, i_prim,
                          j_prim, P(0), P(1), P(2), format_matrix(qt.q));
                */

                fmt::print("{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}\n",
                           qt.q(0), qt.q(1), qt.q(2), qt.q(3), qt.q(4),
                           qt.q(5));
                // Move multipoles to nearest sites
                moveq(P, qt, sites, site_radii, site_limits, site_multipoles,
                      max_rank);

                for (int site_idx = 0; site_idx < site_multipoles.size();
                     site_idx++) {
                  const auto &qt = site_multipoles[site_idx];
                  fmt::print(
                      "{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}{:12.6f}\n",
                      qt.q(0), qt.q(1), qt.q(2), qt.q(3), qt.q(4), qt.q(5));
                }

              } else {
                // For small exponents, use numerical integration
                if (!do_quadrature) {
                  // Initialize grid for integration
                  point_to_site_map.resize(grid.num_points());

                  // Assign each grid point to nearest site
                  for (int p = 0; p < grid.num_points(); p++) {
                    const Vec3 grid_pos = grid_points.col(p);
                    double min_dist = std::numeric_limits<double>::max();
                    int nearest_site = 0;

                    for (int s = 0; s < n_atoms; s++) {
                      double dist = (grid_pos - sites.col(s)).squaredNorm() /
                                    (site_radii(s) * site_radii(s));
                      if (dist < min_dist) {
                        min_dist = dist;
                        nearest_site = s;
                      }
                    }

                    point_to_site_map[p] = nearest_site;
                  }

                  // Initialize density array
                  rho = Vec::Zero(grid.num_points());

                  do_quadrature = true;
                }

                // Calculate center P of overlap distribution
                const double p = alpha_j / alpha_sum;
                const Vec3 P = origin_i + p * r_shell_ij;

                fmt::print("{:5d} {:5d} {:10.5f} {:10.5f} {:10.5f}\n", i_prim,
                           j_prim, P(0), P(1), P(2));

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
    fmt::print("Numerical integration\n");
    const auto &grid_weights = grid.weights();
    // Process the grid density contributions to multipoles
    std::vector<Mult> site_contributions(n_atoms);
    for (int site = 0; site < n_atoms; site++) {
      site_contributions[site].q =
          Vec::Zero(max_rank * max_rank + 2 * max_rank + 1);
    }

    // Initialize temporary arrays for coordinates and their powers
    Vec ggx = Vec::Zero(max_rank + 1);
    Vec ggy = Vec::Zero(max_rank + 1);
    Vec ggz = Vec::Zero(max_rank + 1);

    // Process each grid point
    for (int p = 0; p < grid.num_points(); p++) {
      // Skip if density is negligible
      if (std::abs(rho(p)) < 1e-12)
        continue;

      // Get site this point is assigned to
      int site_idx = point_to_site_map[p];

      // Calculate position relative to site
      const Vec3 &grid_pos = grid_points.col(p);
      const Vec3 rel_pos = grid_pos - sites.col(site_idx);

      // Initialize power arrays
      ggx(0) = 1.0;
      ggy(0) = 1.0;
      ggz(0) = 1.0;

      // Calculate powers of coordinates
      for (int l = 1; l <= max_rank; l++) {
        ggx(l) = ggx(l - 1) * rel_pos(0);
        ggy(l) = ggy(l - 1) * rel_pos(1);
        ggz(l) = ggz(l - 1) * rel_pos(2);
      }

      // Get grid weight
      double weight = grid_weights(p);
      fmt::print("{:12.6f}\n", rho(p));

      // Add contribution to multipoles for this site
      addqlm(max_rank, max_rank, weight * rho(p), ggx, ggy, ggz,
             site_contributions[site_idx]);
    }

    // Add grid contributions to total site multipoles
    for (int site = 0; site < n_atoms; site++) {
      site_multipoles[site].q += site_contributions[site].q;
    }
  }

  fmt::print("mults\n");
  for (const auto &mult : site_multipoles) {
    fmt::print("{}\n", format_matrix(mult.q));
  }

  // Return the calculated multipoles
  return site_multipoles;
}
} // namespace occ::dma
