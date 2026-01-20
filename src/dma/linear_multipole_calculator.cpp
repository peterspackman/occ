#pragma once
#include <cmath>
#include <occ/core/log.h>
#include <occ/dma/add_qlm.h>
#include <occ/dma/gauss_hermite.h>
#include <occ/dma/linear_multipole_calculator.h>
#include <occ/dma/linear_multipole_shifter.h>
#include <occ/gto/shell_order.h>

namespace occ::dma {

// Helper functions following the MultipoleCalculator pattern
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

LinearMultipoleCalculator::LinearMultipoleCalculator(
    const occ::qm::Wavefunction &wfn, const LinearDMASettings &settings)
    : m_wfn(wfn), m_settings(settings) {

  // Setup density matrix with factor of 2
  m_density_matrix = 2 * m_wfn.mo.D;

  // Setup sites and their properties
  setup_sites();

  // Setup slice information if needed
  if (m_settings.use_slices) {
    setup_slices();
  }
}

void LinearMultipoleCalculator::setup_sites() {
  const auto &atoms = m_wfn.atoms;
  const auto &positions = m_wfn.positions();
  const size_t n_atoms = atoms.size();

  // Setup site data (initially just atoms)
  m_sites = positions;
  m_site_radii = Vec::Ones(n_atoms) * m_settings.default_radius;
  m_site_limits = IVec::Constant(n_atoms, m_settings.max_rank);

  // Adjust radii for hydrogens
  for (size_t i = 0; i < n_atoms; i++) {
    if (atoms[i].atomic_number == 1) {
      m_site_radii(i) = m_settings.hydrogen_radius;
    }
  }
}

void LinearMultipoleCalculator::setup_slices() {
  const size_t n_atoms = m_wfn.atoms.size();

  // Limit max rank for slices
  int limited_max_rank = std::min(m_settings.max_rank, 11);
  for (int i = 0; i < n_atoms; i++) {
    m_site_limits(i) = std::min(m_site_limits(i), limited_max_rank);
  }

  // Sort DMA sites in order of increasing z
  m_sort_indices.resize(n_atoms);
  for (int i = 0; i < n_atoms; i++) {
    m_sort_indices[i] = i;
  }

  // Insertion sort for sites by z coordinate
  for (int i = 1; i < n_atoms; i++) {
    int j = i;
    while (j > 0 &&
           m_sites(2, m_sort_indices[j]) < m_sites(2, m_sort_indices[j - 1])) {
      std::swap(m_sort_indices[j], m_sort_indices[j - 1]);
      j--;
    }
  }

  // Coordinates of separating planes
  m_slice_separations = Vec::Zero(n_atoms + 1);
  m_slice_separations(0) = -1.0e6; // Negative "infinity"
  for (int i = 0; i < n_atoms - 1; i++) {
    int i1 = m_sort_indices[i];
    int i2 = m_sort_indices[i + 1];
    m_slice_separations(i + 1) = (m_site_radii(i1) * m_sites(2, i2) +
                                  m_site_radii(i2) * m_sites(2, i1)) /
                                 (m_site_radii(i1) + m_site_radii(i2));
  }
  m_slice_separations(n_atoms) = 1.0e6; // Positive "infinity"
}

std::vector<Mult> LinearMultipoleCalculator::calculate() {
  // Initialize logging
  if (m_settings.use_slices) {
    log::debug("Starting Distributed Multipole Analysis for linear molecules "
               "using slices");
  } else {
    log::debug("Starting Distributed Multipole Analysis for linear molecules");
  }

  const size_t n_atoms = m_wfn.atoms.size();
  std::vector<Mult> site_multipoles(n_atoms);

  // Initialize multipoles at each site
  for (size_t i = 0; i < n_atoms; i++) {
    site_multipoles[i].q = Vec::Zero(m_settings.max_rank + 1);
  }

  // Process nuclear contributions
  process_nuclear_contributions(site_multipoles);

  // Process electronic contributions
  process_electronic_contributions(site_multipoles);

  return site_multipoles;
}

void LinearMultipoleCalculator::process_nuclear_contributions(
    std::vector<Mult> &site_multipoles) {
  if (!m_settings.include_nuclei)
    return;

  const auto &atoms = m_wfn.atoms;
  const auto &positions = m_wfn.positions();

  for (int atom_i = 0; atom_i < atoms.size(); atom_i++) {
    const double zi = positions(2, atom_i); // z-coordinate for linear molecules

    // Create temporary multipole with nuclear charge
    Mult qt(m_settings.max_rank + 1);
    qt.q.setZero();
    qt.q(0) = atoms[atom_i].atomic_number;

    // Move to nearest site
    LinearMultipoleShifter shifter(zi, qt, m_sites, m_site_radii, m_site_limits,
                                   site_multipoles, m_settings.max_rank);
    shifter.move_to_sites();
  }
}

void LinearMultipoleCalculator::process_electronic_contributions(
    std::vector<Mult> &site_multipoles) {
  const auto &basis = m_wfn.basis;
  const auto &shells = basis.shells();
  const auto &atom_to_shells = basis.atom_to_shell();
  const size_t n_atoms = m_wfn.atoms.size();

  // Loop over atoms
  for (int atom_i = 0; atom_i < n_atoms; atom_i++) {
    const auto &i_shells = atom_to_shells[atom_i];

    // Loop over shells for atom i
    for (int i_shell_idx : i_shells) {
      const auto &shell_i = shells[i_shell_idx];

      // Loop over atoms j (up to i to avoid double counting)
      for (int atom_j = 0; atom_j <= atom_i; atom_j++) {
        const auto &j_shells = atom_to_shells[atom_j];

        // Loop over shells for atom j
        for (int j_shell_idx : j_shells) {
          // If atoms are the same, only process up to i_shell to avoid double
          // counting
          if (atom_i == atom_j && j_shell_idx > i_shell_idx)
            continue;

          const auto &shell_j = shells[j_shell_idx];

          process_shell_pair(shell_i, shell_j, i_shell_idx, j_shell_idx, atom_i,
                             atom_j, site_multipoles);
        }
      }
    }
  }
}

void LinearMultipoleCalculator::process_shell_pair(
    const occ::qm::Shell &shell_i, const occ::qm::Shell &shell_j,
    int i_shell_idx, int j_shell_idx, int atom_i, int atom_j,
    std::vector<Mult> &site_multipoles) {
  const auto &basis = m_wfn.basis;
  const auto &first_bf = basis.first_bf();
  const auto n_shells = basis.shells().size();

  // Get basis function ranges for shells
  const int bf_i_start = first_bf[i_shell_idx];
  const int bf_i_end =
      (i_shell_idx + 1 < n_shells) ? first_bf[i_shell_idx + 1] : basis.nbf();
  const int bf_j_start = first_bf[j_shell_idx];
  const int bf_j_end =
      (j_shell_idx + 1 < n_shells) ? first_bf[j_shell_idx + 1] : basis.nbf();

  const bool i_shell_equals_j_shell = (i_shell_idx == j_shell_idx);

  // Extract density matrix block for these shells
  Mat d_block = Mat::Zero(shell_i.size(), shell_j.size());

  for (int bf_i = bf_i_start; bf_i < bf_i_end; bf_i++) {
    int i_idx = bf_i - bf_i_start;

    if (i_shell_equals_j_shell) {
      // For same shell, only need the triangular part
      for (int bf_j = bf_j_start; bf_j <= bf_i; bf_j++) {
        int j_idx = bf_j - bf_j_start;
        d_block(i_idx, j_idx) = m_density_matrix(bf_i, bf_j);
        if (bf_i != bf_j) {
          d_block(j_idx, i_idx) = m_density_matrix(bf_i, bf_j);
        }
      }
    } else {
      // Different shells, fill entire block
      for (int bf_j = bf_j_start; bf_j < bf_j_end; bf_j++) {
        int j_idx = bf_j - bf_j_start;
        d_block(i_idx, j_idx) = m_density_matrix(bf_i, bf_j);
      }
    }
  }

  // Loop over primitives
  int i_prim_max = shell_i.num_primitives();
  for (int i_prim = 0; i_prim < i_prim_max; i_prim++) {
    int j_prim_max =
        i_shell_equals_j_shell ? i_prim + 1 : shell_j.num_primitives();

    for (int j_prim = 0; j_prim < j_prim_max; j_prim++) {
      process_primitive_pair(shell_i, shell_j, i_prim, j_prim, d_block, atom_i,
                             atom_j, site_multipoles);
    }
  }
}

void LinearMultipoleCalculator::process_primitive_pair(
    const occ::qm::Shell &shell_i, const occ::qm::Shell &shell_j, int i_prim,
    int j_prim, const Mat &d_block, int atom_i, int atom_j,
    std::vector<Mult> &site_multipoles) {
  const auto &positions = m_wfn.positions();
  const double zi = positions(2, atom_i);
  const double zj = positions(2, atom_j);
  const double zji = zi - zj;
  const double rr = zji * zji; // Squared distance along z-axis

  const double alpha_i = shell_i.exponents[i_prim];
  const double alpha_j = shell_j.exponents[j_prim];
  const double alpha_sum = alpha_i + alpha_j;

  // Skip if exponential term is negligible
  const double dum = alpha_j * alpha_i * rr / alpha_sum;
  if (dum > m_settings.tolerance)
    return;

  // Factor for exponential term
  double fac = std::exp(-dum);

  // Double the factor if primitives or atoms are different
  if (i_prim != j_prim || atom_i != atom_j) {
    fac *= 2.0;
  }

  // Calculate position of overlap center
  const double p = alpha_j / alpha_sum;
  const double zp = zi - p * zji;
  const double za = zi - zp;
  const double zb = zj - zp;

  // Get contraction coefficients
  const double ci = shell_i.coeff_normalized_dma(0, i_prim);
  const double cj = shell_j.coeff_normalized_dma(0, j_prim);

  const int l_i = shell_i.l;
  const int l_j = shell_j.l;

  if (m_settings.use_slices) {
    // Process each slice separately
    const size_t n_atoms = m_wfn.atoms.size();
    for (int slice_idx = 0; slice_idx < n_atoms; slice_idx++) {
      // Get slice boundaries
      const double z1 = m_slice_separations(slice_idx) - zp;
      const double z2 = m_slice_separations(slice_idx + 1) - zp;

      // Calculate integrals over z for this slice
      Eigen::Tensor<double, 3> gz(21, l_i + 1, l_j + 1);
      gz.setZero();

      bool skip = false;
      calculate_slice_integrals(alpha_sum, l_i, l_j, za, zb, z1, z2, gz, skip);

      if (skip)
        continue;

      // Create temporary multipole for this slice
      Mult qt(m_settings.max_rank + 1);
      qt.q.setZero();

      // Use Gauss-Hermite quadrature for x and y integrals
      const double t = std::sqrt(1.0 / alpha_sum);
      int nq = (l_i + l_j + m_settings.max_rank) / 2;
      const auto points = gauss_hermite_points(nq + 1);
      const auto weights = gauss_hermite_weights(nq + 1);

      Vec gx = Vec::Zero(21);
      for (int k = 0; k < points.size(); k++) {
        const double s = points(k) * t;
        const double g = weights(k) * t;
        double ps = g;
        const int iq_min = std::min(l_i + l_j + m_settings.max_rank, 20);
        for (int iq = 0; iq <= iq_min; iq += 2) {
          gx(iq) += ps;
          ps *= (s * s);
        }
      }

      // Process all basis function combinations
      for (int bf_i_idx = 0; bf_i_idx < shell_i.size(); bf_i_idx++) {
        IVec3 i_powers = get_powers(bf_i_idx, l_i);
        double norm_i =
            get_normalization_factor(i_powers(0), i_powers(1), i_powers(2));

        for (int bf_j_idx = 0; bf_j_idx < shell_j.size(); bf_j_idx++) {
          IVec3 j_powers = get_powers(bf_j_idx, l_j);
          double norm_j =
              get_normalization_factor(j_powers(0), j_powers(1), j_powers(2));

          // The integral is non-zero only if x and y powers are both even
          int mx = i_powers[0] + j_powers[0];
          int my = i_powers[1] + j_powers[1];

          if (mx % 2 == 0 && my % 2 == 0) {
            // Extract gz for these powers - convert to vector
            Vec gz_vec = Vec::Zero(21);
            for (int k = 0; k <= 20; k++) {
              gz_vec(k) = gz(k, i_powers[2], j_powers[2]);
            }

            // Calculate prefactor with negative sign and normalization
            double f =
                -fac * ci * cj * d_block(bf_i_idx, bf_j_idx) * norm_i * norm_j;

            addql0(std::min(nq, m_settings.max_rank), f, gx, gx, gz_vec, qt);
          }
        }
      }

      // Move multipoles to the site in this slice
      int site_idx = m_sort_indices[slice_idx];
      double zs = m_sites(2, site_idx);
      LinearMultipoleShifter::shift_along_axis(qt, 0, m_settings.max_rank,
                                               site_multipoles[site_idx],
                                               m_settings.max_rank, zp - zs);
    }
  } else {
    // Process entire molecule at once (not using slices)
    const double t = std::sqrt(1.0 / alpha_sum);
    int nq = l_i + l_j;
    const auto points = gauss_hermite_points(nq + 1);
    const auto weights = gauss_hermite_weights(nq + 1);

    // Initialize arrays for even powers of x and y
    Vec gx = Vec::Zero(21);

    // Accumulate sums of even powers for x (same used for y)
    for (int k = 0; k < points.size(); k++) {
      const double s = points(k) * t;
      const double g = weights(k) * t;

      double ps = g;
      const int iq_min = std::min(l_i + l_j + m_settings.max_rank, 20);

      for (int iq = 0; iq <= iq_min; iq += 2) {
        gx(iq) += ps;
        ps *= (s * s);
      }
    }

    // Initialize 3D tensor for z integrals
    Eigen::Tensor<double, 3> gz(21, l_i + 1, l_j + 1);
    gz.setZero();

    // Calculate integrals over z using quadrature
    for (int k = 0; k < points.size(); k++) {
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

    // Create temporary multipole
    Mult qt(m_settings.max_rank + 1);
    qt.q.setZero();

    // Process all basis function combinations
    for (int bf_i_idx = 0; bf_i_idx < shell_i.size(); bf_i_idx++) {
      IVec3 i_powers = get_powers(bf_i_idx, l_i);
      double norm_i =
          get_normalization_factor(i_powers(0), i_powers(1), i_powers(2));

      for (int bf_j_idx = 0; bf_j_idx < shell_j.size(); bf_j_idx++) {
        IVec3 j_powers = get_powers(bf_j_idx, l_j);
        double norm_j =
            get_normalization_factor(j_powers(0), j_powers(1), j_powers(2));

        // The integral is non-zero only if x and y powers are both even
        int mx = i_powers[0] + j_powers[0];
        int my = i_powers[1] + j_powers[1];

        if (mx % 2 == 0 && my % 2 == 0) {
          // Extract gz for these powers
          Vec gz_vec = Vec::Zero(21);
          for (int k = 0; k <= 20; k++) {
            gz_vec(k) = gz(k, i_powers[2], j_powers[2]);
          }

          // Calculate prefactor with negative sign and normalization
          double f =
              -fac * ci * cj * d_block(bf_i_idx, bf_j_idx) * norm_i * norm_j;

          // Add contribution to multipoles
          addql0(std::min(nq, m_settings.max_rank), f, gx, gx, gz_vec, qt);
        }
      }
    }

    // Move multipoles to nearest site
    LinearMultipoleShifter shifter(zp, qt, m_sites, m_site_radii, m_site_limits,
                                   site_multipoles, m_settings.max_rank);
    shifter.move_to_sites();
  }
}

void LinearMultipoleCalculator::calculate_slice_integrals(
    double aa, int la, int lb, double za, double zb, double z1, double z2,
    Eigen::Tensor<double, 3> &gz, bool &skip) const {
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

} // namespace occ::dma
