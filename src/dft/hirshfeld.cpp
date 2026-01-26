#include <occ/core/log.h>
#include <occ/core/multipole.h>
#include <occ/core/parallel.h>
#include <occ/dft/hirshfeld.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>

using occ::Mat;
using occ::Mat3N;
using occ::Vec;
using occ::core::Multipole;

namespace occ::dft {

namespace impl {

inline void hirshfeld_kernel_restricted(
    Eigen::Ref<const Mat> r, Eigen::Ref<const Mat> rho,
    Eigen::Ref<const Vec> weights, Eigen::Ref<const Mat> rho_pro,
    Eigen::Ref<Vec> hirshfeld_charges, Eigen::Ref<Vec> atom_volumes,
    double &num_electrons, double &num_electrons_promol) {

  for (int j = 0; j < rho_pro.rows(); j++) {
    double protot = rho_pro.row(j).sum();
    if (protot < 1e-30)
      continue;
    // Calculate weight function (ratio of atomic to promolecular density)
    occ::RowVec hirshfeld_weights = rho_pro.row(j).array() / protot;
    double fac = 2 * rho(j, 0) * weights(j);
    hirshfeld_charges.array() -= hirshfeld_weights.array() * fac;
    num_electrons_promol += protot * weights(j);

    // Calculate atomic volumes (integral of electron density * r³)
    const auto &rja = r.row(j).array();
    atom_volumes.array() +=
        (hirshfeld_weights.array() * rja * rja * rja).transpose().array() * 2 *
        rho(j, 0) * weights(j);

    num_electrons += 2 * rho(j, 0) * weights(j);
  }
}

inline void hirshfeld_kernel_unrestricted(
    Eigen::Ref<const Mat> r, Eigen::Ref<const Mat> rho,
    Eigen::Ref<const Vec> weights, Eigen::Ref<const Mat> rho_pro,
    Eigen::Ref<Vec> hirshfeld_charges, Eigen::Ref<Vec> atom_volumes,
    double &num_electrons, double &num_electrons_promol) {

  const auto &rho_a = occ::qm::block::a(rho);
  const auto &rho_b = occ::qm::block::b(rho);
  Mat rho_tot = rho_a + rho_b;
  hirshfeld_kernel_restricted(r, rho_tot, weights, rho_pro, hirshfeld_charges,
                              atom_volumes, num_electrons,
                              num_electrons_promol);
}

inline void hirshfeld_multipole_kernel_restricted(
    Eigen::Ref<const Mat> r, const std::vector<Mat3N> &r_vec,
    Eigen::Ref<const Mat> rho, Eigen::Ref<const Vec> weights,
    Eigen::Ref<const Mat> rho_pro,
    std::vector<occ::core::Multipole<4>> &multipoles,
    Eigen::Ref<Vec> atom_volumes, double &num_electrons,
    double &num_electrons_promol, int max_multipole_order) {

  for (int j = 0; j < rho_pro.rows(); j++) {
    double protot = rho_pro.row(j).sum();
    if (protot < 1e-30)
      continue;

    // Calculate weight function (ratio of atomic to promolecular density)
    occ::RowVec hirshfeld_weights = rho_pro.row(j).array() / protot;
    double density = 2.0 * rho(j, 0); // factor of 2 for restricted orbitals
    double weighted_density = density * weights(j);

    for (size_t atom_idx = 0; atom_idx < multipoles.size(); atom_idx++) {
      const double weight = hirshfeld_weights(atom_idx);
      if (weight < 1e-30)
        continue;

      // Atom-centered contribution for this grid point
      const double contribution = weight * weighted_density;

      // Monopole (charge)
      multipoles[atom_idx].components[0] -= contribution;

      // Process higher order multipoles if requested
      if (max_multipole_order >= 1) {
        const auto &r_ij =
            r_vec[atom_idx].col(j); // vector from atom to grid point
        const double x = r_ij.x();
        const double y = r_ij.y();
        const double z = r_ij.z();

        // Dipole
        multipoles[atom_idx].components[1] -= x * contribution; // Dx
        multipoles[atom_idx].components[2] -= y * contribution; // Dy
        multipoles[atom_idx].components[3] -= z * contribution; // Dz

        if (max_multipole_order >= 2) {
          // Quadrupole
          multipoles[atom_idx].components[4] -= x * x * contribution; // Qxx
          multipoles[atom_idx].components[5] -= x * y * contribution; // Qxy
          multipoles[atom_idx].components[6] -= x * z * contribution; // Qxz
          multipoles[atom_idx].components[7] -= y * y * contribution; // Qyy
          multipoles[atom_idx].components[8] -= y * z * contribution; // Qyz
          multipoles[atom_idx].components[9] -= z * z * contribution; // Qzz

          if (max_multipole_order >= 3) {
            // Octupole
            multipoles[atom_idx].components[10] -=
                x * x * x * contribution; // Oxxx
            multipoles[atom_idx].components[11] -=
                x * x * y * contribution; // Oxxy
            multipoles[atom_idx].components[12] -=
                x * x * z * contribution; // Oxxz
            multipoles[atom_idx].components[13] -=
                x * y * y * contribution; // Oxyy
            multipoles[atom_idx].components[14] -=
                x * y * z * contribution; // Oxyz
            multipoles[atom_idx].components[15] -=
                x * z * z * contribution; // Oxzz
            multipoles[atom_idx].components[16] -=
                y * y * y * contribution; // Oyyy
            multipoles[atom_idx].components[17] -=
                y * y * z * contribution; // Oyyz
            multipoles[atom_idx].components[18] -=
                y * z * z * contribution; // Oyzz
            multipoles[atom_idx].components[19] -=
                z * z * z * contribution; // Ozzz

            if (max_multipole_order >= 4) {
              // Hexadecapole
              multipoles[atom_idx].components[20] -=
                  x * x * x * x * contribution; // Hxxxx
              multipoles[atom_idx].components[21] -=
                  x * x * x * y * contribution; // Hxxxy
              multipoles[atom_idx].components[22] -=
                  x * x * x * z * contribution; // Hxxxz
              multipoles[atom_idx].components[23] -=
                  x * x * y * y * contribution; // Hxxyy
              multipoles[atom_idx].components[24] -=
                  x * x * y * z * contribution; // Hxxyz
              multipoles[atom_idx].components[25] -=
                  x * x * z * z * contribution; // Hxxzz
              multipoles[atom_idx].components[26] -=
                  x * y * y * y * contribution; // Hxyyy
              multipoles[atom_idx].components[27] -=
                  x * y * y * z * contribution; // Hxyyz
              multipoles[atom_idx].components[28] -=
                  x * y * z * z * contribution; // Hxyzz
              multipoles[atom_idx].components[29] -=
                  x * z * z * z * contribution; // Hxzzz
              multipoles[atom_idx].components[30] -=
                  y * y * y * y * contribution; // Hyyyy
              multipoles[atom_idx].components[31] -=
                  y * y * y * z * contribution; // Hyyyz
              multipoles[atom_idx].components[32] -=
                  y * y * z * z * contribution; // Hyyzz
              multipoles[atom_idx].components[33] -=
                  y * z * z * z * contribution; // Hyzzz
              multipoles[atom_idx].components[34] -=
                  z * z * z * z * contribution; // Hzzzz
            }
          }
        }
      }

      const double r_ij_norm = r(j, atom_idx);
      atom_volumes(atom_idx) +=
          weight * r_ij_norm * r_ij_norm * r_ij_norm * weighted_density;
    }

    num_electrons_promol += protot * weights(j);
    num_electrons += weighted_density;
  }
}

void hirshfeld_multipole_kernel_unrestricted(
    Eigen::Ref<const Mat> r, const std::vector<Mat3N> &r_vec,
    Eigen::Ref<const Mat> rho, Eigen::Ref<const Vec> weights,
    Eigen::Ref<const Mat> rho_pro,
    std::vector<occ::core::Multipole<4>> &multipoles,
    Eigen::Ref<Vec> atom_volumes, double &num_electrons,
    double &num_electrons_promol, int max_multipole_order) {

  const auto &rho_a = occ::qm::block::a(rho);
  const auto &rho_b = occ::qm::block::b(rho);
  Mat rho_tot = rho_a + rho_b;
  hirshfeld_multipole_kernel_restricted(
      r, r_vec, rho_tot, weights, rho_pro, multipoles, atom_volumes,
      num_electrons, num_electrons_promol, max_multipole_order);
}

} // namespace impl

HirshfeldPartition::HirshfeldPartition(const occ::gto::AOBasis &basis,
                                       int max_multipole_order, int charge)
    : m_basis(basis), m_grid(basis), m_max_multipole_order(max_multipole_order),
      m_charge(charge) {

  for (size_t i = 0; i < basis.atoms().size(); i++) {
    m_atom_grids.push_back(m_grid.get_partitioned_atom_grid(i));
  }

  size_t num_grid_points = std::accumulate(
      m_atom_grids.begin(), m_atom_grids.end(), 0.0,
      [&](double tot, const auto &grid) { return tot + grid.points.cols(); });

  occ::log::debug("HirshfeldPartition: {} total grid points", num_grid_points);

  // Determine if this is an atomic ion
  m_atomic_ion = (m_atom_grids.size() == 1) && (m_charge != 0);
  std::vector<int> oxidation_states = {};
  if (m_atomic_ion) {
    oxidation_states.push_back(m_charge);
  }

  m_slater_basis = occ::slater::slaterbasis_for_atoms(
      m_basis.atoms(), "thakkar", oxidation_states);
}

Vec HirshfeldPartition::calculate(const occ::qm::MolecularOrbitals &mo) {
  // Only recalculate if density matrix has changed
  if (m_density_matrix.size() != 0 &&
      occ::util::all_close(mo.D, m_density_matrix)) {
    return m_hirshfeld_charges;
  }

  m_density_matrix = mo.D;
  compute_hirshfeld_weights(mo, false);
  return m_hirshfeld_charges;
}

std::vector<occ::core::Multipole<4>>
HirshfeldPartition::calculate_multipoles(const occ::qm::MolecularOrbitals &mo) {

  if (m_density_matrix.size() != 0 &&
      occ::util::all_close(mo.D, m_density_matrix) && !m_multipoles.empty()) {
    return m_multipoles;
  }

  m_density_matrix = mo.D;
  compute_hirshfeld_weights(mo, true);
  return m_multipoles;
}

void HirshfeldPartition::compute_hirshfeld_weights(
    const occ::qm::MolecularOrbitals &mo, bool calculate_higher_multipoles) {

  const auto &atoms = m_basis.atoms();
  const size_t num_atoms = atoms.size();

  bool unrestricted = (mo.kind == occ::qm::SpinorbitalKind::Unrestricted);
  occ::log::debug("HirshfeldPartition using {} wavefunction",
                  unrestricted ? "unrestricted" : "restricted");

  constexpr size_t BLOCKSIZE = 64;
  int num_rows_factor = unrestricted ? 2 : 1;

  constexpr int max_derivative{1}; // Only need density, not derivatives

  // Use TBB-based thread-local storage
  occ::parallel::thread_local_storage<Vec> tl_hirshfeld_charges(Vec::Zero(num_atoms));
  occ::parallel::thread_local_storage<std::vector<occ::core::Multipole<4>>> tl_multipoles(
    [num_atoms]() {
      std::vector<occ::core::Multipole<4>> thread_multipoles(num_atoms);
      for (auto &m : thread_multipoles) {
        std::fill(m.components.begin(), m.components.end(), 0.0);
      }
      return thread_multipoles;
    }
  );

  occ::parallel::thread_local_storage<Vec> tl_atom_volumes(Vec::Zero(num_atoms));
  occ::parallel::thread_local_storage<Vec> tl_free_atom_volumes(Vec::Zero(num_atoms));
  occ::parallel::thread_local_storage<double> tl_num_electrons(0.0);
  occ::parallel::thread_local_storage<double> tl_num_electrons_promol(0.0);

  for (const auto &atom_grid : m_atom_grids) {
    const auto &atom_pts = atom_grid.points;
    const auto &atom_weights = atom_grid.weights;
    const size_t npt_total = atom_pts.cols();
    const size_t num_blocks = npt_total / BLOCKSIZE + 1;

    // Use TBB-based thread-local storage for temporary variables
    occ::parallel::thread_local_storage<occ::gto::GTOValues> gto_vals_local(
      [this]() {
        occ::gto::GTOValues gto_vals;
        gto_vals.reserve(m_basis.nbf(), BLOCKSIZE, max_derivative);
        return gto_vals;
      }
    );

    occ::parallel::parallel_for(size_t(0), num_blocks, [&](size_t block) {
      auto &gto_vals = gto_vals_local.local();
      auto &hirshfeld_charges = tl_hirshfeld_charges.local();
      auto &atom_volumes = tl_atom_volumes.local();
      auto &free_atom_volumes = tl_free_atom_volumes.local();
      auto &num_e = tl_num_electrons.local();
      auto &num_e_promol = tl_num_electrons_promol.local();

      Eigen::Index l = block * BLOCKSIZE;
      Eigen::Index u = std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
      Eigen::Index npt = u - l;
      if (npt <= 0)
        return;

      Mat hirshfeld_weights = Mat::Zero(npt, num_atoms);
      Mat r = Mat::Zero(npt, num_atoms);
      Mat rho = Mat::Zero(num_rows_factor * npt,
                          occ::density::num_components(max_derivative));

      std::vector<Mat3N> r_vec;
      if (calculate_higher_multipoles) {
        r_vec.resize(num_atoms, Mat3N::Zero(3, npt));
      }

      const auto &pts_block = atom_pts.middleCols(l, npt);
      const auto &weights_block = atom_weights.segment(l, npt);

      occ::gto::evaluate_basis(m_basis, pts_block, gto_vals, max_derivative);

      if (unrestricted) {
        occ::density::evaluate_density<
            max_derivative, occ::qm::SpinorbitalKind::Unrestricted>(
            mo.D, gto_vals, rho);
      } else {
        occ::density::evaluate_density<max_derivative,
                                       occ::qm::SpinorbitalKind::Restricted>(
            mo.D, gto_vals, rho);
      }

      for (int i = 0; i < num_atoms; i++) {
        const auto &sb = m_slater_basis[i];
        occ::Vec3 pos{atoms[i].x, atoms[i].y, atoms[i].z};

        if (calculate_higher_multipoles) {
          for (int j = 0; j < npt; j++) {
            r_vec[i].col(j) = pts_block.col(j) - pos;
          }
        }

        r.col(i) = (pts_block.colwise() - pos).colwise().norm();
        const auto &ria = r.col(i).array();

        hirshfeld_weights.col(i) = sb.rho(r.col(i));

        free_atom_volumes(i) += (hirshfeld_weights.col(i).array() *
                                 weights_block.array() * ria * ria * ria)
                                    .sum();
      }

      if (calculate_higher_multipoles) {
        auto &multipoles = tl_multipoles.local();
        if (unrestricted) {
          impl::hirshfeld_multipole_kernel_unrestricted(
              r, r_vec, rho, weights_block, hirshfeld_weights,
              multipoles, atom_volumes, num_e, num_e_promol,
              m_max_multipole_order);
        } else {
          impl::hirshfeld_multipole_kernel_restricted(
              r, r_vec, rho, weights_block, hirshfeld_weights,
              multipoles, atom_volumes, num_e, num_e_promol,
              m_max_multipole_order);
        }
      } else {
        if (unrestricted) {
          impl::hirshfeld_kernel_unrestricted(
              r, rho, weights_block, hirshfeld_weights, hirshfeld_charges,
              atom_volumes, num_e, num_e_promol);
        } else {
          impl::hirshfeld_kernel_restricted(
              r, rho, weights_block, hirshfeld_weights, hirshfeld_charges,
              atom_volumes, num_e, num_e_promol);
        }
      }
    });
  }

  // Initialize charges with nuclear charges
  m_hirshfeld_charges = Vec::Zero(num_atoms);
  const auto &ecp_electrons = m_basis.ecp_electrons();
  for (int i = 0; i < num_atoms; i++) {
    m_hirshfeld_charges(i) = static_cast<double>(atoms[i].atomic_number);
    if (ecp_electrons.size() >= i) {
      m_hirshfeld_charges(i) -= ecp_electrons[i];
    }
  }

  double num_electrons{0.0};
  double num_electrons_promol{0.0};
  m_atom_volumes = Vec::Zero(num_atoms);
  m_free_atom_volumes = Vec::Zero(num_atoms);

  if (calculate_higher_multipoles) {
    m_multipoles.resize(num_atoms);
    for (auto &m : m_multipoles) {
      std::fill(m.components.begin(), m.components.end(), 0.0);
    }
    for (int i = 0; i < num_atoms; i++) {
      m_multipoles[i].components[0] = m_hirshfeld_charges(i);
    }
  }

  // Reduce results from thread-local storage
  if (calculate_higher_multipoles) {
    for (const auto &multipoles : tl_multipoles) {
      for (size_t j = 0; j < num_atoms; j++) {
        // Add all multipole components except monopole (already set above)
        for (size_t k = 1; k < m_multipoles[j].components.size(); k++) {
          m_multipoles[j].components[k] += multipoles[j].components[k];
        }
        // Add the monopole (charge) component separately
        m_hirshfeld_charges(j) += multipoles[j].components[0];
        m_multipoles[j].components[0] = m_hirshfeld_charges(j);
      }
    }
  } else {
    for (const auto &charges : tl_hirshfeld_charges) {
      m_hirshfeld_charges += charges;
    }
  }

  for (const auto &volumes : tl_atom_volumes) {
    m_atom_volumes += volumes;
  }
  for (const auto &free_volumes : tl_free_atom_volumes) {
    m_free_atom_volumes += free_volumes;
  }
  for (const auto &ne : tl_num_electrons) {
    num_electrons += ne;
  }
  for (const auto &ne_promol : tl_num_electrons_promol) {
    num_electrons_promol += ne_promol;
  }

  occ::log::debug("Hirshfeld analysis: electrons in molecule = {:.6f}, in "
                  "promolecule = {:.6f}",
                  num_electrons, num_electrons_promol);

  occ::log::debug("Hirshfeld charges:");
  for (int i = 0; i < m_hirshfeld_charges.rows(); i++) {
    occ::log::debug("  Atom {}: {:.6f}", i, m_hirshfeld_charges(i));
  }

  // smear remaining charge across all atoms
  m_hirshfeld_charges.array() +=
      ((m_charge - m_hirshfeld_charges.sum()) / m_hirshfeld_charges.rows());

  if (calculate_higher_multipoles) {
    occ::log::debug("Hirshfeld multipoles:");
    for (size_t i = 0; i < m_multipoles.size(); i++) {
      occ::log::debug("  Atom {}:", i);
      // set the charge to incorporate the smeared charge
      m_multipoles[i].components[0] = m_hirshfeld_charges(i);
      occ::log::debug("{}", m_multipoles[i].to_string());
    }
  }
}

Vec calculate_hirshfeld_charges(const occ::gto::AOBasis &basis,
                                const occ::qm::MolecularOrbitals &mo,
                                int charge) {
  HirshfeldPartition hirshfeld(basis, 0, charge);
  return hirshfeld.calculate(mo);
}

std::vector<occ::core::Multipole<4>>
calculate_hirshfeld_multipoles(const occ::gto::AOBasis &basis,
                               const occ::qm::MolecularOrbitals &mo,
                               int max_multipole_order, int charge) {
  HirshfeldPartition hirshfeld(basis, max_multipole_order, charge);
  return hirshfeld.calculate_multipoles(mo);
}

} // namespace occ::dft
