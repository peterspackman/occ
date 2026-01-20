#pragma once
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/dft/molecular_grid.h>
#include <occ/dma/gauss_hermite.h>
#include <occ/dma/multipole_calculator.h>
#include <occ/dma/multipole_shifter.h>
#include <occ/gto/shell_order.h>
#include <occ/qm/hf.h>

namespace occ::dma {

inline double multipole_get_normalization_factor(int l, int m, int n) {
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

// AnalyticalIntegrator implementation
AnalyticalIntegrator::AnalyticalIntegrator(const DMASettings &settings)
    : m_settings(settings) {}

void AnalyticalIntegrator::calculate_primitive_contribution(
    const qm::Shell &shell_i, const qm::Shell &shell_j, int i_prim, int j_prim,
    double fac, const Mat &d_block, const Vec3 &P, Mult &qt) const {

  const int l_i = shell_i.l;
  const int l_j = shell_j.l;
  const Vec3 &origin_i = shell_i.origin;
  const Vec3 &origin_j = shell_j.origin;
  const double alpha_i = shell_i.exponents[i_prim];
  const double alpha_j = shell_j.exponents[j_prim];
  const double alpha_sum = alpha_i + alpha_j;

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
  m_gx = Eigen::Tensor<double, 3>(nq + 1, l_i + 1, l_j + 1);
  m_gy = Eigen::Tensor<double, 3>(nq + 1, l_i + 1, l_j + 1);
  m_gz = Eigen::Tensor<double, 3>(nq + 1, l_i + 1, l_j + 1);

  m_gx.setZero();
  m_gy.setZero();
  m_gz.setZero();

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

    // Following the Fortran implementation pattern to accumulate integrals
    double pax = g, pay = g, paz = g;

    for (int ia = 0; ia <= l_i; ia++) {
      double px = pax, py = pay, pz = paz;

      for (int ib = 0; ib <= l_j; ib++) {
        double pq = 1.0;

        for (int iq = 0; iq <= nq; iq++) {
          m_gx(iq, ia, ib) += px * pq;
          m_gy(iq, ia, ib) += py * pq;
          m_gz(iq, ia, ib) += pz * pq;
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

  Vec gx_vec(nq + 1), gy_vec(nq + 1), gz_vec(nq + 1);
  int bf_i_idx = 0;
  gto::iterate_over_shell<true>(
      [&](int i1, int j1, int k1, int ll1) {
        int bf_j_idx = 0;
        gto::iterate_over_shell<true>(
            [&](int i2, int j2, int k2, int ll2) {
              // Skip if density matrix element is too small
              if (std::abs(d_block(bf_i_idx, bf_j_idx)) < 1e-12) {
                bf_j_idx++;
                return;
              }

              double f = -fac * ci * cj * d_block(bf_i_idx, bf_j_idx);

              // Create 1D vectors from tensor slices for addqlm
              for (int n = 0; n <= nq; n++) {
                gx_vec(n) = m_gx(n, i1, i2); // x powers from both shells
                gy_vec(n) = m_gy(n, j1, j2); // y powers from both shells
                gz_vec(n) = m_gz(n, k1, k2); // z powers from both shells
              }

              // Add contribution to multipoles
              addqlm(lq, m_settings.max_rank, f, gx_vec, gy_vec, gz_vec, qt);

              bf_j_idx++;
            },
            l_j);

        bf_i_idx++;
      },
      l_i);
}

// GridIntegrator implementation
GridIntegrator::GridIntegrator(const DMASettings &settings)
    : m_settings(settings) {}

template <int BlockSize = 64>
void add_primitive_to_grid_blocked(const qm::Shell &shell_i,
                                   const qm::Shell &shell_j, int i_prim,
                                   int j_prim, double fac, const Mat &d_block,
                                   const Vec3 &P, const Mat3N &grid_points,
                                   Vec &rho, double etol) {
  const int l_i = shell_i.l;
  const int l_j = shell_j.l;
  const Vec3 &origin_i = shell_i.origin;
  const Vec3 &origin_j = shell_j.origin;
  const double alpha_i = shell_i.exponents[i_prim];
  const double alpha_j = shell_j.exponents[j_prim];
  const double alpha_sum = alpha_i + alpha_j;

  // Get contraction coefficients
  const double ci = shell_i.coeff_normalized_dma(0, i_prim);
  const double cj = shell_j.coeff_normalized_dma(0, j_prim);

  constexpr int BS = BlockSize;
  const int max_l = std::max(l_i, l_j);

  // Pre-allocate block arrays
  Eigen::Array<double, BS, 1> block_exp_factors;
  Eigen::Array<double, BS, 1> block_rho_contrib;
  Eigen::Array<bool, BS, 1> block_mask;

  std::vector<Eigen::Array<double, BS, 3>> ri_powers_block(l_i + 1);
  std::vector<Eigen::Array<double, BS, 3>> rj_powers_block(l_j + 1);

  ri_powers_block[0].setConstant(1.0);
  rj_powers_block[0].setConstant(1.0);

  // Process grid in blocks
  for (int block_start = 0; block_start < grid_points.cols();
       block_start += BS) {
    int block_end = std::min(block_start + BS, (int)grid_points.cols());
    int actual_block_size = block_end - block_start;

    int valid_count = 0;
    for (int i = 0; i < actual_block_size; i++) {
      int p = block_start + i;
      const Vec3 grid_pos = grid_points.col(p);
      double dist2 = (grid_pos - P).squaredNorm();
      double e = alpha_sum * dist2;

      if (e <= etol) {
        block_mask(i) = true;
        block_exp_factors(i) = std::exp(-e);
        valid_count++;

        Vec3 r_i = grid_pos - origin_i;
        Vec3 r_j = grid_pos - origin_j;

        // Store l=1 powers (displacements)
        if (l_i >= 1) {
          ri_powers_block[1](i, 0) = r_i(0);
          ri_powers_block[1](i, 1) = r_i(1);
          ri_powers_block[1](i, 2) = r_i(2);
        }
        if (l_j >= 1) {
          rj_powers_block[1](i, 0) = r_j(0);
          rj_powers_block[1](i, 1) = r_j(1);
          rj_powers_block[1](i, 2) = r_j(2);
        }
      } else {
        block_mask(i) = false;
      }
    }

    // Skip this block if no valid points
    if (valid_count == 0)
      continue;

    for (int l = 2; l <= l_i; l++) {
      for (int i = 0; i < actual_block_size; i++) {
        if (block_mask(i)) {
          ri_powers_block[l](i, 0) =
              ri_powers_block[l - 1](i, 0) * ri_powers_block[1](i, 0);
          ri_powers_block[l](i, 1) =
              ri_powers_block[l - 1](i, 1) * ri_powers_block[1](i, 1);
          ri_powers_block[l](i, 2) =
              ri_powers_block[l - 1](i, 2) * ri_powers_block[1](i, 2);
        }
      }
    }

    for (int l = 2; l <= l_j; l++) {
      for (int i = 0; i < actual_block_size; i++) {
        if (block_mask(i)) {
          rj_powers_block[l](i, 0) =
              rj_powers_block[l - 1](i, 0) * rj_powers_block[1](i, 0);
          rj_powers_block[l](i, 1) =
              rj_powers_block[l - 1](i, 1) * rj_powers_block[1](i, 1);
          rj_powers_block[l](i, 2) =
              rj_powers_block[l - 1](i, 2) * rj_powers_block[1](i, 2);
        }
      }
    }

    block_rho_contrib.head(actual_block_size).setZero();

    int bf_i_idx = 0;
    gto::iterate_over_shell<true>(
        [&](int i1, int j1, int k1, int ll1) {
          int bf_j_idx = 0;
          gto::iterate_over_shell<true>(
              [&](int i2, int j2, int k2, int ll2) {
                if (std::abs(d_block(bf_i_idx, bf_j_idx)) >= 1e-12) {
                  double f = -fac * ci * cj * d_block(bf_i_idx, bf_j_idx);

                  for (int i = 0; i < actual_block_size; i++) {
                    if (block_mask(i)) {
                      double shell_i_contrib = ri_powers_block[i1](i, 0) *
                                               ri_powers_block[j1](i, 1) *
                                               ri_powers_block[k1](i, 2);
                      double shell_j_contrib = rj_powers_block[i2](i, 0) *
                                               rj_powers_block[j2](i, 1) *
                                               rj_powers_block[k2](i, 2);
                      block_rho_contrib(i) += f * block_exp_factors(i) *
                                              shell_i_contrib * shell_j_contrib;
                    }
                  }
                }
                bf_j_idx++;
              },
              l_j);
          bf_i_idx++;
        },
        l_i);

    for (int i = 0; i < actual_block_size; i++) {
      if (block_mask(i)) {
        rho(block_start + i) += block_rho_contrib(i);
      }
    }
  }
}

void GridIntegrator::add_primitive_to_grid(const qm::Shell &shell_i,
                                           const qm::Shell &shell_j, int i_prim,
                                           int j_prim, double fac,
                                           const Mat &d_block, const Vec3 &P,
                                           const Mat3N &grid_points, Vec &rho,
                                           double etol) const {
  add_primitive_to_grid_blocked<64>(shell_i, shell_j, i_prim, j_prim, fac,
                                    d_block, P, grid_points, rho, etol);
}

void GridIntegrator::process_grid_density(
    const Vec &rho, const Mat3N &grid_points, const Vec &grid_weights,
    const std::vector<std::pair<size_t, size_t>> &atom_blocks,
    const DMASites &sites, std::vector<Mult> &site_multipoles) const {

  Mult qt(m_settings.max_rank);
  // Initialize temporary arrays for coordinates and their powers
  Vec ggx = Vec::Zero(m_settings.max_rank + 1);
  Vec ggy = Vec::Zero(m_settings.max_rank + 1);
  Vec ggz = Vec::Zero(m_settings.max_rank + 1);

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
      for (int l = 1; l <= m_settings.max_rank; l++) {
        ggx(l) = ggx(l - 1) * rel_pos(0);
        ggy(l) = ggy(l - 1) * rel_pos(1);
        ggz(l) = ggz(l - 1) * rel_pos(2);
      }

      // Get grid weight
      double weight = grid_weights(p);

      // Add contribution to multipoles for this site
      addqlm(m_settings.max_rank, m_settings.max_rank, weight * rho(p), ggx,
             ggy, ggz, qt);
    }

    MultipoleShifter shifter(pos_i, qt, sites, site_multipoles,
                             m_settings.max_rank);
    shifter.shift();
  }
}

// MultipoleCalculator implementation
MultipoleCalculator::MultipoleCalculator(const qm::AOBasis &basis,
                                         const qm::MolecularOrbitals &mo,
                                         const DMASites &sites,
                                         const DMASettings &settings)
    : m_basis(basis), m_mo(mo), m_sites(sites), m_settings(settings),
      m_analytical(settings), m_grid(settings), m_tolerance(2.30258 * 18),
      m_use_quadrature(false) {
  setup_normalized_density_matrix();
}

void MultipoleCalculator::setup_normalized_density_matrix() {
  qm::HartreeFock hf(m_basis);
  Mat bf_norm = hf.compute_overlap_matrix()
                    .diagonal()
                    .array()
                    .sqrt()
                    .matrix()
                    .asDiagonal();
  m_normalized_density = 2 * m_mo.D;
  m_normalized_density = bf_norm * m_normalized_density * bf_norm;

  // Apply normalization factors to density matrix
  const auto &shells = m_basis.shells();
  const auto &first_bf = m_basis.first_bf();
  const auto n_shells = shells.size();

  for (int i_shell_idx = 0; i_shell_idx < n_shells; i_shell_idx++) {
    const auto &shell_i = shells[i_shell_idx];
    const int l_i = shell_i.l;

    for (int j_shell_idx = 0; j_shell_idx < n_shells; j_shell_idx++) {
      const auto &shell_j = shells[j_shell_idx];
      const int l_j = shell_j.l;

      int bf_i_idx = 0;
      gto::iterate_over_shell<true>(
          [&](int i1, int j1, int k1, int ll1) {
            double norm_i = multipole_get_normalization_factor(i1, j1, k1);

            int bf_j_idx = 0;
            gto::iterate_over_shell<true>(
                [&](int i2, int j2, int k2, int ll2) {
                  double norm_j = multipole_get_normalization_factor(i2, j2, k2);

                  m_normalized_density(first_bf[i_shell_idx] + bf_i_idx,
                                       first_bf[j_shell_idx] + bf_j_idx) *=
                      norm_i * norm_j;

                  bf_j_idx++;
                },
                l_j);

            bf_i_idx++;
          },
          l_i);
    }
  }
}

void MultipoleCalculator::process_nuclear_contributions(
    std::vector<Mult> &site_multipoles) {
  if (!m_settings.include_nuclei)
    return;

  for (int atom_i = 0; atom_i < m_sites.atoms.size(); atom_i++) {
    const Vec3 pos_i = m_sites.atoms[atom_i].position();

    Mult qt;
    qt.q = Vec::Zero(m_settings.max_rank * m_settings.max_rank +
                     2 * m_settings.max_rank + 1);
    qt.Q00() = m_sites.atoms[atom_i].atomic_number;

    MultipoleShifter shifter(pos_i, qt, m_sites, site_multipoles,
                             m_settings.max_rank);
    shifter.shift();
  }
}

std::vector<Mult> MultipoleCalculator::calculate() {
  occ::timing::start(occ::timing::category::dma_total);
  log::debug("Starting Distributed Multipole Analysis");
  log::debug("Site radii : {}\n", format_matrix(m_sites.radii));
  log::debug("Site limits: {}\n", format_matrix(m_sites.limits, "{}"));

  std::vector<Mult> site_multipoles;
  site_multipoles.reserve(m_sites.size());
  for (int i = 0; i < m_sites.size(); i++) {
    site_multipoles.push_back(Mult(m_sites.limits(i)));
  }

  // Handle nuclear contributions
  process_nuclear_contributions(site_multipoles);

  // Handle electronic contributions
  process_electronic_contributions(site_multipoles);

  occ::timing::stop(occ::timing::category::dma_total);
  return site_multipoles;
}

void MultipoleCalculator::process_electronic_contributions(
    std::vector<Mult> &site_multipoles) {
  const auto &shells = m_basis.shells();
  const auto n_shells = shells.size();
  const auto &shell_to_atom = m_basis.shell_to_atom();
  const auto &atom_to_shells = m_basis.atom_to_shell();
  const auto &first_bf = m_basis.first_bf();

  double etol = 36.0; // Threshold for density calculation

  dft::GridSettings grid_settings;
  grid_settings.max_angular_points = 590;
  grid_settings.radial_points = 80;
  grid_settings.treutler_alrichs_adjustment = false;

  auto grid_gen = dft::MolecularGrid(m_basis, grid_settings);
  grid_gen.set_atomic_radii(m_sites.radii);
  grid_gen.populate_molecular_grid_points();

  const auto &grid = grid_gen.get_molecular_grid_points();
  const auto &grid_points = grid.points();

  // Use TBB-based thread-local storage for multipoles and grid density
  occ::parallel::thread_local_storage<std::vector<Mult>> thread_site_multipoles(
    [&]() {
      std::vector<Mult> local_multipoles(site_multipoles.size());
      for (size_t i = 0; i < site_multipoles.size(); i++) {
        local_multipoles[i] = Mult(m_settings.max_rank);
      }
      return local_multipoles;
    }
  );
  occ::parallel::thread_local_storage<Vec> thread_grid_density(Vec::Zero(grid.num_points()));
  occ::parallel::thread_local_storage<bool> thread_use_quadrature(false);
  
  // Thread-local analytical and grid integrators
  occ::parallel::thread_local_storage<AnalyticalIntegrator> analytical_local(
    [&m_settings = m_settings]() { return AnalyticalIntegrator(m_settings); }
  );
  occ::parallel::thread_local_storage<GridIntegrator> grid_local(
    [&m_settings = m_settings]() { return GridIntegrator(m_settings); }
  );

  // Build list of unique shell pairs (i,j) with i >= j
  std::vector<std::pair<int, int>> shell_pairs_to_process;
  for (int i_shell_idx = 0; i_shell_idx < n_shells; i_shell_idx++) {
    for (int j_shell_idx = 0; j_shell_idx <= i_shell_idx; j_shell_idx++) {
      shell_pairs_to_process.emplace_back(i_shell_idx, j_shell_idx);
    }
  }
  
  occ::parallel::parallel_for(size_t(0), shell_pairs_to_process.size(), [&](size_t idx) {
    auto &local_site_multipoles = thread_site_multipoles.local();
    auto &local_grid_density = thread_grid_density.local();
    bool &local_use_quadrature = thread_use_quadrature.local();
    auto &local_analytical = analytical_local.local();
    auto &local_grid = grid_local.local();
    
    int i_shell_idx = shell_pairs_to_process[idx].first;
    int j_shell_idx = shell_pairs_to_process[idx].second;
    
    const auto &shell_i = shells[i_shell_idx];
    const auto &shell_j = shells[j_shell_idx];
    const bool i_shell_equals_j_shell = (i_shell_idx == j_shell_idx);

    Mat d_block = m_normalized_density.block(first_bf[i_shell_idx],
                                             first_bf[j_shell_idx],
                                             shell_i.size(), shell_j.size());

    // Calculate shell separation
    const Vec3 r_shell_ij = shell_j.origin - shell_i.origin;
    const double shell_r2 = r_shell_ij.squaredNorm();

    // Loop over primitives in shell i
    for (int i_prim = 0; i_prim < shell_i.num_primitives(); i_prim++) {
      const double alpha_i = shell_i.exponents[i_prim];

      int j_prim_max =
          i_shell_equals_j_shell ? i_prim + 1 : shell_j.num_primitives();

      for (int j_prim = 0; j_prim < j_prim_max; j_prim++) {
        const double alpha_j = shell_j.exponents[j_prim];
        const double alpha_sum = alpha_i + alpha_j;

        // Skip if shell distance is too large or exponential term is
        // negligible
        const double dum = alpha_j * alpha_i * shell_r2 / alpha_sum;
        if (dum > m_tolerance)
          continue;

        double fac = std::exp(-dum);

        if (i_prim != j_prim || !(i_shell_equals_j_shell)) {
          fac *= 2.0;
        }

        const double p = alpha_j / alpha_sum;
        const Vec3 P = shell_i.origin + p * r_shell_ij;

        if ((alpha_i + alpha_j) > m_settings.big_exponent) {
          // Use analytical method for large exponents
          Mult qt(m_settings.max_rank);

          local_analytical.calculate_primitive_contribution(
              shell_i, shell_j, i_prim, j_prim, fac, d_block, P, qt);

          MultipoleShifter shifter(P, qt, m_sites, local_site_multipoles,
                                   m_settings.max_rank);
          shifter.shift();

        } else {
          // For small exponents, use numerical integration
          if (!local_use_quadrature) {
            local_use_quadrature = true;
          }
          local_grid.add_primitive_to_grid(shell_i, shell_j, i_prim, j_prim, fac,
                                           d_block, P, grid_points,
                                           local_grid_density, etol);
        }
      }
    }
  });

  // Reduce results from all threads
  for (const auto &local_multipoles : thread_site_multipoles) {
    for (size_t i = 0; i < site_multipoles.size(); i++) {
      site_multipoles[i].q += local_multipoles[i].q;
    }
  }
  
  // Check if any thread used quadrature by seeing if density is non-zero
  for (const auto &local_grid_density : thread_grid_density) {
    if (local_grid_density.norm() > 0) {
      if (!m_use_quadrature) {
        m_grid_density = Vec::Zero(grid.num_points());
        m_use_quadrature = true;
      }
      m_grid_density += local_grid_density;
    }
  }

  // If we did numerical integration, process the grid density
  if (m_use_quadrature) {
    const auto &grid_weights = grid.weights();
    const auto &atom_blocks = grid.atom_blocks();

    m_grid.process_grid_density(m_grid_density, grid_points, grid_weights,
                                atom_blocks, m_sites, site_multipoles);
  }
}

} // namespace occ::dma
