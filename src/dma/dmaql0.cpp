#include <cmath>
#include <occ/core/log.h>
#include <occ/dma/add_qlm.h>
#include <occ/dma/binomial.h>
#include <occ/dma/dmaql0.h>
#include <occ/dma/gauss_hermite.h>
#include <occ/dma/movez.h>
#include <occ/gto/shell_order.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace occ::dma {

/**
 * @brief Calculate error function integrals for slices of space
 *
 * This is equivalent to the DMAERF subroutine in the GDMA Fortran code.
 * It calculates integrals over a slice between z1 and z2 for linear
 * molecules.
 *
 * @param aa Sum of gaussian exponents
 * @param la Angular momentum of first function
 * @param lb Angular momentum of second function
 * @param za Z-coordinate of first center relative to product center
 * @param zb Z-coordinate of second center relative to product center
 * @param z1 Lower bound of slice
 * @param z2 Upper bound of slice
 * @param gz Output tensor for integrals
 * @param skip Flag set to true if all integrals are negligible
 */
void dmaerf(double aa, int la, int lb, double za, double zb, double z1,
            double z2, Eigen::Tensor<double, 3> &gz, bool &skip) {
  const double rtpi = std::sqrt(M_PI);

  // Calculate v = sqrt(aa) and scaling factor t = rtpi/v
  double v = std::sqrt(aa);
  double t = rtpi / v;

  // Calculate first integral - Gaussian integrated between z1 and z2
  double gz000 = 0.5 * t * (std::erf(v * z2) - std::erf(v * z1));
  gz(0, 0, 0) = gz000;

  // Skip if integral is negligible
  skip = (std::abs(gz000) < 1.0e-8);
  if (skip)
    return;

  // Calculate exponential factors
  double e1 = std::exp(-aa * z1 * z1) / (2.0 * aa);
  double e2 = std::exp(-aa * z2 * z2) / (2.0 * aa);

  // First order integral with power of z
  gz(1, 0, 0) = e1 - e2;

  // Check if still significant
  skip = skip && (std::abs(gz(1, 0, 0)) < 1.0e-8);
  if (skip)
    return;

  // Calculate higher order integrals recursively
  double p1 = e1;
  double p2 = e2;

  for (int n = 2; n <= 16; n++) {
    p1 *= z1;
    p2 *= z2;
    gz(n, 0, 0) = p1 - p2 + (n - 1) * gz(n - 2, 0, 0) / (2.0 * aa);

    skip = skip && (std::abs(gz(n, 0, 0)) < 1.0e-8);
    if (skip)
      return;
  }

  // Calculate remaining integrals using recursion relations
  for (int n = 15; n >= 0; n--) {
    for (int l = 1; l <= std::min(lb, 16 - n); l++) {
      gz(n, 0, l) = gz(n + 1, 0, l - 1) - zb * gz(n, 0, l - 1);
    }

    for (int k = 1; k <= std::min(la, 16 - n); k++) {
      gz(n, k, 0) = gz(n + 1, k - 1, 0) - za * gz(n, k - 1, 0);

      for (int l = 1; l <= std::min(lb, 16 - n - la); l++) {
        gz(n, k, l) = gz(n + 1, k, l - 1) - zb * gz(n, k, l - 1);
      }
    }
  }
}

/**
 * @brief Calculate multipole moments for linear molecules and shift them
 * to the nearest site.
 *
 * This is a direct implementation of the dmaql0 subroutine from the GDMA
 * Fortran code. It calculates multipole moments for linear molecules,
 * where only the Qlm with m=0 are non-zero and stored in the order Q0,
 * Q1, Q2, ...
 *
 * @param wfn Wavefunction containing basis set and density matrix
 * @param max_rank Maximum multipole rank to calculate
 * @param include_nuclei Whether to include nuclear contributions in the
 * calculation
 * @param use_slices Whether to use slices of space for calculating
 * multipoles
 * @return std::vector<Mult> Multipole moments for each site
 */
std::vector<Mult> dmaql0(const occ::qm::Wavefunction &wfn, int max_rank,
                         bool include_nuclei, bool use_slices) {

  // Initialize logging
  if (use_slices) {
    log::info("Starting Distributed Multipole Analysis for linear molecules "
              "using slices");
    log::info(" slice        site               separator");
  } else {
    log::info("Starting Distributed Multipole Analysis for linear molecules");
    log::info("    Atoms   Shells Primitives  Position     Multipole "
              "contributions ...");
  }

  // Get molecule data from wavefunction
  const auto &atoms = wfn.atoms;
  const auto &positions = wfn.positions();
  const size_t n_atoms = atoms.size();

  // Get density matrix
  const auto D = 2 * wfn.mo.D;

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

  // Adjust radii for hydrogens
  for (size_t i = 0; i < n_atoms; i++) {
    if (atoms[i].atomic_number == 1) {
      site_radii(i) = 0.325; // Hydrogen atoms get smaller radius
    }

    // Initialize multipoles at each site
    site_multipoles[i].q = Vec::Zero(max_rank + 1);
    fmt::print("{}\n", atoms[i].atomic_number);
    for(const auto &shell_idx: atom_to_shells[i]) {
      fmt::print("Shell {}\n", shell_idx);
       const auto &sh = shells[shell_idx];
      for(int i = 0; i < sh.num_primitives(); i++) {
        fmt::print("{:12d} {:12.8f} {:12.8f}\n", i, sh.exponents(i), sh.coeff_normalized_dma(0, i));
      }
    }
  }

  IVec site_limits = IVec::Constant(n_atoms, max_rank);

  // Threshold for numerical significance
  const double tol = 2.30258 * 18;

  // Storage for temporary multipoles
  Mult qt(max_rank + 1);

  // Sorting information for slices
  std::vector<int> sort(n_atoms);
  Vec sep = Vec::Zero(n_atoms + 1);

  if (use_slices) {
    // Limit max rank for slices
    max_rank = std::min(max_rank, 11);
    for (int i = 0; i < n_atoms; i++) {
      site_limits(i) = std::min(site_limits(i), max_rank);
    }

    // Sort DMA sites in order of increasing z
    for (int i = 0; i < n_atoms; i++) {
      sort[i] = i;
    }

    // Insertion sort for sites by z coordinate
    for (int i = 1; i < n_atoms; i++) {
      int j = i;
      while (j > 0 && sites(2, sort[j]) < sites(2, sort[j - 1])) {
        std::swap(sort[j], sort[j - 1]);
        j--;
      }
    }

    // Coordinates of separating planes
    sep(0) = -1.0e6; // Negative "infinity"
    for (int i = 0; i < n_atoms - 1; i++) {
      int i1 = sort[i];
      int i2 = sort[i + 1];
      sep(i + 1) =
          (site_radii(i1) * sites(2, i2) + site_radii(i2) * sites(2, i1)) /
          (site_radii(i1) + site_radii(i2));
    }
    sep(n_atoms) = 1.0e6; // Positive "infinity"

    // Log slice information if verbose
    log::info("Separator at z = {:.4f}", sep(0));
    for (int i = 0; i < n_atoms; i++) {
      log::info("{:4d} {:8d} {:10.4f} {:14.4f}", i, sort[i], sites(2, sort[i]),
                sep(i + 1));
    }

    log::info("   Atoms   Shells Primitives  Position");
    log::info("Slice     Multipole contributions ...");
  } else {
    log::info("    Atoms   Shells Primitives  Position     Multipole "
              "contributions ...");
  }


  // Loop over atoms
  for (int atom_i = 0; atom_i < n_atoms; atom_i++) {
    const double zi =
        positions(2, atom_i); // Using only z-coordinate for linear molecules

    // Clear temporary multipoles
    qt.q.setZero();

    // Include nuclear charges if requested
    if (include_nuclei) {
      qt.q(0) = atoms[atom_i].atomic_number;
      movez(qt, zi, sites, site_radii, site_limits, site_multipoles, max_rank);
    }

    // Find shells for atom i
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
        const double zj =
            positions(2, atom_j); // Only z-coordinate for linear molecules

        // Calculate atom separation (only z component)
        const double zji = zi - zj;
        const double rr = zji * zji; // Squared distance along z-axis

        // Find shells for atom j
        const auto &j_shells = atom_to_shells[atom_j];

        // Loop over shells for atom j
        for (int j_shell_idx : j_shells) {
          // If atoms are the same, only process up to i_shell to avoid double
          // counting
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

          // Extract density matrix block for these shells
          Mat d_block = Mat::Zero(shell_i.size(), shell_j.size());

          for (int bf_i = bf_i_start; bf_i < bf_i_end; bf_i++) {
            int i_idx = bf_i - bf_i_start;

            if (i_shell_equals_j_shell) {
              // For same shell, only need the triangular part
              for (int bf_j = bf_j_start; bf_j <= bf_i; bf_j++) {
                int j_idx = bf_j - bf_j_start;
                d_block(i_idx, j_idx) = D(bf_i, bf_j);
                if (bf_i != bf_j) {
                  d_block(j_idx, i_idx) = D(bf_i, bf_j);
                }
              }
            } else {
              // Different shells, fill entire block
              for (int bf_j = bf_j_start; bf_j < bf_j_end; bf_j++) {
                int j_idx = bf_j - bf_j_start;
                d_block(i_idx, j_idx) = D(bf_i, bf_j);
              }
            }
          }

          // Apply scaling factors for angular momentum functions
          auto apply_scaling = [&d_block](int l, int size, bool is_row) {
            const double rt3 = std::sqrt(3.0);
            const double rt5 = std::sqrt(5.0);
            const double rt7 = std::sqrt(7.0);
            const double rt15 = std::sqrt(15.0);

            if (l == 2) { // d-functions
              if (is_row) {
                // Scale rows 7,8,9 (dxy, dxz, dyz) by sqrt(3)
                for (int j = 0; j < d_block.cols(); j++) {
                  for (int i = 3; i < 6 && i < size; i++) {
                    d_block(i, j) *= rt3;
                  }
                }
              } else {
                // Scale columns
                for (int i = 0; i < d_block.rows(); i++) {
                  for (int j = 3; j < 6 && j < size; j++) {
                    d_block(i, j) *= rt3;
                  }
                }
              }
            } else if (l == 3) { // f-functions
              if (is_row) {
                // Scale rows for f functions
                for (int j = 0; j < d_block.cols(); j++) {
                  for (int i = 3; i < 9 && i < size; i++) {
                    d_block(i, j) *= rt5;
                  }
                  if (9 < size)
                    d_block(9, j) *= rt15;
                }
              } else {
                // Scale columns
                for (int i = 0; i < d_block.rows(); i++) {
                  for (int j = 3; j < 9 && j < size; j++) {
                    d_block(i, j) *= rt5;
                  }
                  if (9 < size)
                    d_block(i, 9) *= rt15;
                }
              }
            } else if (l == 4) { // g-functions
              if (is_row) {
                for (int j = 0; j < d_block.cols(); j++) {
                  // g scaling factors from dma.f90
                  for (int i = 4; i < 10 && i < size; i++) {
                    d_block(i, j) *= rt7;
                  }
                  for (int i = 10; i < 13 && i < size; i++) {
                    d_block(i, j) *= rt5 * rt7 / rt3;
                  }
                  for (int i = 13; i < 15 && i < size; i++) {
                    d_block(i, j) *= rt5 * rt7;
                  }
                }
              } else {
                for (int i = 0; i < d_block.rows(); i++) {
                  for (int j = 4; j < 10 && j < size; j++) {
                    d_block(i, j) *= rt7;
                  }
                  for (int j = 10; j < 13 && j < size; j++) {
                    d_block(i, j) *= rt5 * rt7 / rt3;
                  }
                  for (int j = 13; j < 15 && j < size; j++) {
                    d_block(i, j) *= rt5 * rt7;
                  }
                }
              }
            } else if (l == 5) { // h-functions
              if (is_row) {
                for (int j = 0; j < d_block.cols(); j++) {
                  // h scaling factors from dma.f90
                  for (int i = 5; i < 11 && i < size; i++) {
                    d_block(i, j) *= 3.0;
                  }
                  for (int i = 11; i < 17 && i < size; i++) {
                    d_block(i, j) *= rt3 * rt7;
                  }
                  for (int i = 17; i < 20 && i < size; i++) {
                    d_block(i, j) *= 3.0 * rt7;
                  }
                  for (int i = 20; i < 21 && i < size; i++) {
                    d_block(i, j) *= rt3 * rt5 * rt7;
                  }
                }
              } else {
                for (int i = 0; i < d_block.rows(); i++) {
                  for (int j = 5; j < 11 && j < size; j++) {
                    d_block(i, j) *= 3.0;
                  }
                  for (int j = 11; j < 17 && j < size; j++) {
                    d_block(i, j) *= rt3 * rt7;
                  }
                  for (int j = 17; j < 20 && j < size; j++) {
                    d_block(i, j) *= 3.0 * rt7;
                  }
                  for (int j = 20; j < 21 && j < size; j++) {
                    d_block(i, j) *= rt3 * rt5 * rt7;
                  }
                }
              }
            }
          };

          log::info("d_block: {}", format_matrix(d_block));
          // Apply scaling for both shells
          if (l_i >= 2)
            apply_scaling(l_i, shell_i.size(), true);
          if (l_j >= 2)
            apply_scaling(l_j, shell_j.size(), false);

          // Loop over primitives in shell i
          int i_prim_max = shell_i.num_primitives();
          for (int i_prim = 0; i_prim < i_prim_max; i_prim++) {
            const double alpha_i = shell_i.exponents[i_prim];

            // If shells are the same, only loop up to i_prim to avoid double
            // counting
            int j_prim_max =
                i_shell_equals_j_shell ? i_prim + 1 : shell_j.num_primitives();

            for (int j_prim = 0; j_prim < j_prim_max; j_prim++) {
              const double alpha_j = shell_j.exponents[j_prim];
              const double alpha_sum = alpha_i + alpha_j;

              // Skip if exponential term is negligible
              const double dum = alpha_j * alpha_i * rr / alpha_sum;
              if (dum > tol)
                continue;

              // Factor for exponential term
              double fac = std::exp(-dum);

              // Double the factor if primitives or shells are different
              if (i_prim != j_prim || !i_equals_j) {
                fac *= 2.0;
              }

              // Calculate position of overlap center
              const double p = alpha_j / alpha_sum;
              const double zp = zi - p * zji;
              const double za = zi - zp;
              const double zb = zj - zp;

              log::info("{:5d}{:5d} {:5d}{:5d} {:5d}{:5d} {:11.4f}", atom_i,
                        atom_j, i_shell_idx, j_shell_idx, i_prim, j_prim, zp);

              // Use numerical integration to evaluate multipole integrals
              const double t = std::sqrt(1.0 / alpha_sum);

              // Calculate order of quadrature needed
              int nq;
              if (use_slices) {
                nq = (l_i + l_j + max_rank) / 2;
              } else {
                nq = l_i + l_j;
              }

              // Get Gauss-Hermite quadrature points and weights
              const auto points = gauss_hermite_points(nq + 1);
              const auto weights = gauss_hermite_weights(nq + 1);

              // Initialize array for even powers of x and y
              Vec gx = Vec::Zero(21);

              // Accumulate sums of even powers for x (same used for y)
              const int point_count = points.size();
              for (int k = 0; k < point_count; k++) {
                const double s = points(k) * t;
                const double g = weights(k) * t;

                double ps = g;
                const int iq_min = std::min(l_i + l_j + max_rank, 20);

                for (int iq = 0; iq <= iq_min; iq += 2) {
                  gx(iq) += ps;
                  ps *= (s * s); // Square for even powers
                }
              }

              if (use_slices) {
                // Process each slice separately
                for (int slice_idx = 0; slice_idx < n_atoms; slice_idx++) {
                  // Get slice boundaries
                  const double z1 = sep(slice_idx) - zp;
                  const double z2 = sep(slice_idx + 1) - zp;

                  // Calculate integrals over z for this slice
                  Eigen::Tensor<double, 3> gz(21, l_i + 1, l_j + 1);
                  gz.setZero();

                  // Call dmaerf to get integrals
                  bool skip = false;
                  dmaerf(alpha_sum, l_i, l_j, za, zb, z1, z2, gz, skip);

                  // If all integrals are negligible, skip this slice
                  if (skip)
                    continue;

                  // Clear temporary multipoles for this slice
                  qt.q.setZero();

                  // Helper to get cartesian powers for basis functions
                  auto get_powers = [](int bf_idx, int l, int powers[3]) {
                    // Maps from basis function index within shell to x,y,z
                    // powers Following the order in occ::gto::shell_order
                    const int IX[56] = {0, 1, 0, 0, 2, 0, 0, 1, 1, 0,
                                        3, 0, 0, 2, 2, 1, 0, 1, 0, 1};
                    const int IY[56] = {0, 0, 1, 0, 0, 2, 0, 1, 0, 1,
                                        0, 3, 0, 1, 0, 2, 2, 0, 1, 1};
                    const int IZ[56] = {0, 0, 0, 1, 0, 0, 2, 0, 1, 1,
                                        0, 0, 3, 0, 1, 0, 1, 2, 2, 1};

                    if (bf_idx < 56) {
                      powers[0] = IX[bf_idx];
                      powers[1] = IY[bf_idx];
                      powers[2] = IZ[bf_idx];
                    } else {
                      // Default to zero if index out of range
                      powers[0] = powers[1] = powers[2] = 0;
                    }
                  };

                  // Get contraction coefficients
                  const double ci = shell_i.coeff_normalized_dma(0, i_prim);
                  const double cj = shell_j.coeff_normalized_dma(0, j_prim);

                  // Process all basis function combinations
                  for (int bf_i_idx = 0; bf_i_idx < shell_i.size();
                       bf_i_idx++) {
                    int i_powers[3] = {0, 0, 0};
                    get_powers(bf_i_idx, l_i, i_powers);

                    for (int bf_j_idx = 0; bf_j_idx < shell_j.size();
                         bf_j_idx++) {
                      int j_powers[3] = {0, 0, 0};
                      get_powers(bf_j_idx, l_j, j_powers);

                      // The integral is non-zero only if x and y powers are
                      // both even
                      int mx = i_powers[0] + j_powers[0];
                      int my = i_powers[1] + j_powers[1];

                      if (mx % 2 == 0 && my % 2 == 0) {
                        // Extract gz for these powers - convert to vector
                        Vec gz_vec = Vec::Zero(21);
                        for (int k = 0; k <= 20; k++) {
                          gz_vec(k) = gz(k, i_powers[2], j_powers[2]);
                        }

                        // Calculate prefactor with negative sign
                        double f = -fac * ci * cj * d_block(bf_i_idx, bf_j_idx);

                        // Add contribution to multipoles
                        // Note: First parameter is max_rank, not qt
                        addql0(max_rank, f, gx, gx, gz_vec, qt);
                      }
                    }
                  }


                  // Move multipoles to the site in this slice
                  int site_idx = sort[slice_idx];
                  double zs = sites(2, site_idx);
                  shiftz(qt, 0, max_rank, site_multipoles[site_idx], max_rank,
                         zp - zs);
                }
              } else {
                // Process entire molecule at once (not using slices)

                // Initialize 3D tensor for z integrals with dimensions [21,
                // l_i+1, l_j+1]
                Eigen::Tensor<double, 3> gz(21, l_i + 1, l_j + 1);
                gz.setZero();

                // Calculate integrals over z using quadrature
                for (int k = 0; k < point_count; k++) {
                  const double s = points(k) * t;
                  const double g = weights(k) * t;

                  const double zas = s - za;
                  const double zbs = s - zb;

                  double paz = g;
                  for (int ia = 0; ia <= l_i; ia++) {
                    double pz = paz;

                    for (int ib = 0; ib <= l_j; ib++) {
                      double pq = pz;

                      for (int iq = 0; iq <= l_i + l_j; iq++) {
                        gz(iq, ia, ib) += pq;
                        pq *= s;
                      }

                      pz *= zbs;
                    }

                    paz *= zas;
                  }
                }

                // Clear temporary multipoles
                qt.q.setZero();

                // Helper to get cartesian powers for basis functions
                auto get_powers = [](int bf_idx, int l, int powers[3]) {
                  // Maps from basis function index within shell to x,y,z powers
                  const int IX[56] = {0, 1, 0, 0, 2, 0, 0, 1, 1, 0,
                                      3, 0, 0, 2, 2, 1, 0, 1, 0, 1};
                  const int IY[56] = {0, 0, 1, 0, 0, 2, 0, 1, 0, 1,
                                      0, 3, 0, 1, 0, 2, 2, 0, 1, 1};
                  const int IZ[56] = {0, 0, 0, 1, 0, 0, 2, 0, 1, 1,
                                      0, 0, 3, 0, 1, 0, 1, 2, 2, 1};

                  if (bf_idx < 56) {
                    powers[0] = IX[bf_idx];
                    powers[1] = IY[bf_idx];
                    powers[2] = IZ[bf_idx];
                  } else {
                    // Default to zero if index out of range
                    powers[0] = powers[1] = powers[2] = 0;
                  }
                };

                const double ci = shell_i.coeff_normalized_dma(0, i_prim);
                const double cj = shell_j.coeff_normalized_dma(0, j_prim);

                // Process all basis function combinations
                for (int bf_i_idx = 0; bf_i_idx < shell_i.size(); bf_i_idx++) {
                  int i_powers[3] = {0, 0, 0};
                  get_powers(bf_i_idx, l_i, i_powers);

                  for (int bf_j_idx = 0; bf_j_idx < shell_j.size();
                       bf_j_idx++) {
                    int j_powers[3] = {0, 0, 0};
                    get_powers(bf_j_idx, l_j, j_powers);

                    // The integral is non-zero only if x and y powers are both
                    // even
                    int mx = i_powers[0] + j_powers[0];
                    int my = i_powers[1] + j_powers[1];

                    if (mx % 2 == 0 && my % 2 == 0) {
                      // Extract gz for these powers
                      Vec gz_vec = Vec::Zero(21);
                      for (int k = 0; k <= 20; k++) {
                        gz_vec(k) = gz(k, i_powers[2], j_powers[2]);
                      }

                      // Calculate prefactor with negative sign
                      double f = -fac * ci * cj * d_block(bf_i_idx, bf_j_idx);

                      // Add contribution to multipoles
                      // Note the first parameter is rank limit not the
                      // multipole vector
                      addql0(std::min(nq, max_rank), f, gx, gx, gz_vec, qt);

                      log::info("{:.2e} {:.2e} {:.2e} {:3d} {:3d} {:.2e} "
                                "{:.2e} {:.2e} {:.2e}",
                                fac, ci, cj, bf_i_idx, bf_j_idx,
                                d_block(bf_i_idx, bf_j_idx), gx(mx), gx(my),
                                gz_vec(0));
                    }
                  }
                }

                log::info("{} {} {} {} {} {} {}", atom_i, atom_j, i_shell_idx,
                          j_shell_idx, i_prim, j_prim, zp);
                log::info("{}", format_matrix(qt.q));
                // Move multipoles to nearest site
                movez(qt, zp, sites, site_radii, site_limits, site_multipoles,
                      max_rank);
              }

              // End of primitive pairs
            }
          }

          // Debug output for multipoles at each site
          log::info("mults");
          for (const auto &mult : site_multipoles) {
            log::info("{}", format_matrix(mult.q));
          }

          // End of shell pairs
        }
      }
      // End of atom pairs
    }
    // End of shells for atom i
  }
  // End of atoms

  return site_multipoles;
}
} // namespace occ::dma
