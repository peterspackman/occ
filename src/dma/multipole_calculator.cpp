#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/dft/molecular_grid.h>
#include <occ/dma/add_qlm.h>
#include <occ/dma/gauss_hermite.h>
#include <occ/dma/multipole_calculator.h>
#include <occ/dma/multipole_shifter.h>
#include <occ/gto/shell_order.h>
#include <occ/qm/hf.h>

namespace occ::dma {

// Helper functions moved from dmaqlm.cpp
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

  // Process all basis function combinations
  for (int bf_i_idx = 0; bf_i_idx < shell_i.size(); bf_i_idx++) {
    IVec3 i_powers = get_powers(bf_i_idx, l_i);

    for (int bf_j_idx = 0; bf_j_idx < shell_j.size(); bf_j_idx++) {
      IVec3 j_powers = get_powers(bf_j_idx, l_j);

      // Skip if density matrix element is too small
      if (std::abs(d_block(bf_i_idx, bf_j_idx)) < 1e-12)
        continue;

      // Calculate prefactor (negative sign as in dmaqlm)
      double f = -fac * ci * cj * d_block(bf_i_idx, bf_j_idx);

      // Create 1D vectors from tensor slices for addqlm
      Vec gx_vec(nq + 1), gy_vec(nq + 1), gz_vec(nq + 1);
      for (int n = 0; n <= nq; n++) {
        gx_vec(n) = m_gx(n, i_powers(0), j_powers(0));
        gy_vec(n) = m_gy(n, i_powers(1), j_powers(1));
        gz_vec(n) = m_gz(n, i_powers(2), j_powers(2));
      }

      // Add contribution to multipoles
      addqlm(lq, m_settings.max_rank, f, gx_vec, gy_vec, gz_vec, qt);
    }
  }
}

// GridIntegrator implementation
GridIntegrator::GridIntegrator(const DMASettings &settings)
    : m_settings(settings) {}

void GridIntegrator::add_primitive_to_grid(const qm::Shell &shell_i,
                                           const qm::Shell &shell_j, int i_prim,
                                           int j_prim, double fac,
                                           const Mat &d_block, const Vec3 &P,
                                           const Mat3N &grid_points, Vec &rho,
                                           double etol) const {

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

  // Process grid points
  for (int p = 0; p < grid_points.cols(); p++) {
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
    for (int bf_i_idx = 0; bf_i_idx < shell_i.size(); bf_i_idx++) {
      IVec3 i_powers = get_powers(bf_i_idx, l_i);

      for (int bf_j_idx = 0; bf_j_idx < shell_j.size(); bf_j_idx++) {
        IVec3 j_powers = get_powers(bf_j_idx, l_j);

        // Skip if density matrix element is too small
        if (std::abs(d_block(bf_i_idx, bf_j_idx)) < 1e-12)
          continue;

        // Calculate prefactor (negative sign as in dmaqlm)
        double f = -fac * ci * cj * d_block(bf_i_idx, bf_j_idx);

        // Calculate basis function product at grid point
        double bf_product = exp_factor * x_i_powers[i_powers[0]] *
                            y_i_powers[i_powers[1]] * z_i_powers[i_powers[2]] *
                            x_j_powers[j_powers[0]] * y_j_powers[j_powers[1]] *
                            z_j_powers[j_powers[2]];

        // Add contribution to density at this grid point
        rho(p) += f * bf_product;
      }
    }
  }
}

void GridIntegrator::process_grid_density(
    const Vec &rho, const Mat3N &grid_points, const Vec &grid_weights,
    const std::vector<std::pair<size_t, size_t>> &atom_blocks,
    const DMASites &sites, std::vector<Mult> &site_multipoles) const {

  Mult qt;
  qt.q = Vec::Zero(m_settings.max_rank * m_settings.max_rank +
                   2 * m_settings.max_rank + 1);

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

    MultipoleShifter shifter(pos_i, qt, sites.positions, sites.radii, sites.limits,
          site_multipoles, m_settings.max_rank);
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

      for (int bf_i_idx = 0; bf_i_idx < shell_i.size(); bf_i_idx++) {
        IVec3 i_powers = get_powers(bf_i_idx, l_i);
        double norm_i =
            get_normalization_factor(i_powers(0), i_powers(1), i_powers(2));

        for (int bf_j_idx = 0; bf_j_idx < shell_j.size(); bf_j_idx++) {
          IVec3 j_powers = get_powers(bf_j_idx, l_j);
          double norm_j =
              get_normalization_factor(j_powers(0), j_powers(1), j_powers(2));
          m_normalized_density(first_bf[i_shell_idx] + bf_i_idx,
                               first_bf[j_shell_idx] + bf_j_idx) *=
              norm_i * norm_j;
        }
      }
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

    MultipoleShifter shifter(pos_i, qt, m_sites.positions, m_sites.radii, m_sites.limits,
          site_multipoles, m_settings.max_rank);
    shifter.shift();
  }
}

std::vector<Mult> MultipoleCalculator::calculate() {
  log::debug("Starting Distributed Multipole Analysis");
  log::debug("Site radii : {}\n", format_matrix(m_sites.radii));
  log::debug("Site limits: {}\n", format_matrix(m_sites.limits, "{}"));

  std::vector<Mult> site_multipoles(m_sites.size());

  // Handle nuclear contributions
  process_nuclear_contributions(site_multipoles);

  // Handle electronic contributions
  process_electronic_contributions(site_multipoles);

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
  grid_settings.treutler_alrichs_adjustment = false;

  auto grid_gen = dft::MolecularGrid(m_basis, grid_settings);
  grid_gen.set_atomic_radii(m_sites.radii);
  grid_gen.populate_molecular_grid_points();

  const auto &grid = grid_gen.get_molecular_grid_points();
  const auto &grid_points = grid.points();

  // Loop over atoms
  for (int atom_i = 0; atom_i < m_sites.atoms.size(); atom_i++) {
    const Vec3 pos_i = m_sites.atoms[atom_i].position();
    const auto &i_shells = atom_to_shells[atom_i];

    // Loop over shells for atom i
    for (int i_shell_idx : i_shells) {
      const auto &shell_i = shells[i_shell_idx];

      // Loop over atoms j (up to i to avoid double counting)
      for (int atom_j = 0; atom_j <= atom_i; atom_j++) {
        const bool i_equals_j = (atom_i == atom_j);
        const Vec3 &pos_j = m_sites.atoms[atom_j].position();

        // Calculate atom separation
        const Vec3 r_ij = pos_i - pos_j;
        const double r_ij_squared = r_ij.squaredNorm();

        const auto &j_shells = atom_to_shells[atom_j];

        // Loop over shells for atom j
        for (int j_shell_idx : j_shells) {
          if (i_equals_j && j_shell_idx > i_shell_idx)
            continue;

          const auto &shell_j = shells[j_shell_idx];
          const bool i_shell_equals_j_shell = (i_shell_idx == j_shell_idx);

          Mat d_block = m_normalized_density.block(
              first_bf[i_shell_idx], first_bf[j_shell_idx], shell_i.size(),
              shell_j.size());

          // Loop over primitives in shell i
          for (int i_prim = 0; i_prim < shell_i.num_primitives(); i_prim++) {
            const double alpha_i = shell_i.exponents[i_prim];

            int j_prim_max =
                i_shell_equals_j_shell ? i_prim + 1 : shell_j.num_primitives();

            for (int j_prim = 0; j_prim < j_prim_max; j_prim++) {
              const double alpha_j = shell_j.exponents[j_prim];
              const double alpha_sum = alpha_i + alpha_j;

              // Calculate shell separation
              const Vec3 r_shell_ij = shell_j.origin - shell_i.origin;
              const double shell_r2 = r_shell_ij.squaredNorm();

              // Skip if shell distance is too large or exponential term is
              // negligible
              const double dum = alpha_j * alpha_i * shell_r2 / alpha_sum;
              if (dum > m_tolerance)
                continue;

              // Factor for exponential term
              double fac = std::exp(-dum);

              // Double the factor if primitives or shells are different
              if (i_prim != j_prim || !(i_shell_equals_j_shell)) {
                fac *= 2.0;
              }

              // P is the gaussian product center
              const double p = alpha_j / alpha_sum;
              const Vec3 P = shell_i.origin + p * r_shell_ij;

              if ((alpha_i + alpha_j) > m_settings.big_exponent) {
                // Use analytical method for large exponents
                Mult qt;
                qt.q = Vec::Zero(m_settings.max_rank * m_settings.max_rank +
                                 2 * m_settings.max_rank + 1);

                m_analytical.calculate_primitive_contribution(
                    shell_i, shell_j, i_prim, j_prim, fac, d_block, P, qt);

                // Move multipoles to nearest sites
                MultipoleShifter shifter(P, qt, m_sites.positions, m_sites.radii, m_sites.limits,
                      site_multipoles, m_settings.max_rank);
                shifter.shift();

              } else {
                // For small exponents, use numerical integration
                if (!m_use_quadrature) {
                  m_grid_density = Vec::Zero(grid.num_points());
                  m_use_quadrature = true;
                }

                m_grid.add_primitive_to_grid(shell_i, shell_j, i_prim, j_prim,
                                             fac, d_block, P, grid_points,
                                             m_grid_density, etol);
              }
            }
          }
        }
      }
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
