#include <fmt/core.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>
#include <occ/xdm/becke_hole.h>
#include <occ/xdm/xdm.h>

using occ::Mat;
using occ::Mat3N;
using occ::Vec;
using occ::core::Element;

namespace occ::xdm {

double xdm_polarizability(int n, double v, double vfree, bool charged) {
  // TODO handle charged atoms!
  // Thakkar polarizabilities for default oxidation states
  // are in occ::interaction but is this enough?
  return v * Element(n).polarizability(charged) / vfree;
}

std::tuple<double, Mat3N, Mat3N>
xdm_dispersion_interaction_energy(const XDMAtomList &atom_info_a,
                                  const XDMAtomList &atom_info_b,
                                  const XDM::Parameters &params) {
  const auto &atoms_a = atom_info_a.atoms;
  const auto &moments_a = atom_info_a.moments;
  const auto &polarizabilities_a = atom_info_a.polarizabilities;

  const auto &atoms_b = atom_info_b.atoms;
  const auto &moments_b = atom_info_b.moments;
  const auto &polarizabilities_b = atom_info_b.polarizabilities;

  const size_t num_atoms_a = atoms_a.size();
  const size_t num_atoms_b = atoms_b.size();

  using occ::Vec3;

  // Generate atom pair interactions for parallel processing
  std::vector<std::pair<int, int>> atom_pairs;
  for (int i = 0; i < num_atoms_a; i++) {
    for (int j = 0; j < num_atoms_b; j++) {
      atom_pairs.emplace_back(i, j);
    }
  }

  // Use thread-local storage for accumulating results
  occ::parallel::thread_local_storage<Mat3N> forces_a_local(Mat3N::Zero(3, num_atoms_a));
  occ::parallel::thread_local_storage<Mat3N> forces_b_local(Mat3N::Zero(3, num_atoms_b));
  occ::parallel::thread_local_storage<double> edisp_local(0.0);

  occ::parallel::parallel_for(size_t(0), atom_pairs.size(), [&](size_t pair_idx) {
    auto &forces_a = forces_a_local.local();
    auto &forces_b = forces_b_local.local();
    auto &edisp = edisp_local.local();
    
    const int i = atom_pairs[pair_idx].first;
    const int j = atom_pairs[pair_idx].second;
    
    Vec3 pi = {atoms_a[i].x, atoms_a[i].y, atoms_a[i].z};
    Vec3 pj = {atoms_b[j].x, atoms_b[j].y, atoms_b[j].z};
    Vec3 v_ij = pj - pi;
    double pol_i = polarizabilities_a(i);
    double pol_j = polarizabilities_b(j);
    double factor = pol_i * pol_j / (moments_a(0, i) * pol_j + moments_b(0, j) * pol_i);
    double rij = v_ij.norm();

    if (rij < 1e-15) return;

    double rij2 = rij * rij;
    double rij4 = rij2 * rij2;
    double rij6 = rij4 * rij2;
    double rij8 = rij4 * rij4;
    double rij10 = rij4 * rij6;

    double c6 = factor * moments_a(0, i) * moments_b(0, j);
    double c8 = 1.5 * factor * (moments_a(0, i) * moments_b(1, j) + moments_a(1, i) * moments_b(0, j));
    double c10 = 2.0 * factor * (moments_a(0, i) * moments_b(2, j) + moments_a(2, i) * moments_b(0, j)) +
                 4.2 * factor * moments_a(1, i) * moments_b(1, j);
    double rc = (std::sqrt(c8 / c6) + std::sqrt(std::sqrt(c10 / c6)) + std::sqrt(c10 / c8)) / 3;
    double rvdw = params.a1 * rc + params.a2 * occ::units::ANGSTROM_TO_BOHR;
    double rvdw2 = rvdw * rvdw;
    double rvdw4 = rvdw2 * rvdw2;
    double rvdw6 = rvdw4 * rvdw2;
    double rvdw8 = rvdw4 * rvdw4;
    double rvdw10 = rvdw4 * rvdw6;

    occ::log::debug("{:>3d} {:>3d} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}",
                    i, j, rij, c6, c8, c10, rc, rvdw);

    edisp -= c6 / (rvdw6 + rij6) + c8 / (rvdw8 + rij8) + c10 / (rvdw10 + rij10);

    double c6_com = 6.0 * c6 * rij4 / ((rvdw6 + rij6) * (rvdw6 + rij6));
    double c8_com = 8.0 * c8 * rij6 / ((rvdw8 + rij8) * (rvdw8 + rij8));
    double c10_com = 10.0 * c10 * rij8 / ((rvdw10 + rij10) * (rvdw10 + rij10));
    
    forces_a.col(i) += (c6_com + c8_com + c10_com) * v_ij;
    forces_b.col(j) -= (c6_com + c8_com + c10_com) * v_ij;
  });

  // Reduce results from all threads
  Mat3N forces_a = Mat3N::Zero(3, num_atoms_a);
  Mat3N forces_b = Mat3N::Zero(3, num_atoms_b);
  double edisp = 0.0;
  
  for (const auto &forces_a_thread : forces_a_local) {
    forces_a += forces_a_thread;
  }
  for (const auto &forces_b_thread : forces_b_local) {
    forces_b += forces_b_thread;
  }
  for (const auto &edisp_thread : edisp_local) {
    edisp += edisp_thread;
  }
  
  return {edisp, forces_a, forces_b};
}

std::pair<double, Mat3N> xdm_dispersion_energy(const XDMAtomList &atom_info,
                                               const XDM::Parameters &params) {
  const auto &atoms = atom_info.atoms;
  const auto &volume = atom_info.volume;
  const auto &moments = atom_info.moments;
  const auto &volume_free = atom_info.volume_free;
  const auto &polarizabilities = atom_info.polarizabilities;
  const size_t num_atoms = atoms.size();
  using occ::Vec3;

  occ::log::debug("{:>20s} {:>20s} {:>20s}", "Volume", "Volume Free",
                  "Polarizability");
  for (int i = 0; i < volume.rows(); i++) {
    occ::log::debug("{:20.12f} {:20.12f} {:20.12f}", volume(i), volume_free(i),
                    polarizabilities(i));
  }

  // Generate atom pairs for parallel processing
  std::vector<std::pair<int, int>> atom_pairs;
  for (int i = 0; i < num_atoms; i++) {
    for (int j = i + 1; j < num_atoms; j++) {
      atom_pairs.emplace_back(i, j);
    }
  }

  // Use thread-local storage for accumulating results
  occ::parallel::thread_local_storage<Mat3N> forces_local(Mat3N::Zero(3, num_atoms));
  occ::parallel::thread_local_storage<double> edisp_local(0.0);

  occ::parallel::parallel_for(size_t(0), atom_pairs.size(), [&](size_t pair_idx) {
    auto &forces = forces_local.local();
    auto &edisp = edisp_local.local();
    
    const int i = atom_pairs[pair_idx].first;
    const int j = atom_pairs[pair_idx].second;
    
    Vec3 pi = {atoms[i].x, atoms[i].y, atoms[i].z};
    Vec3 pj = {atoms[j].x, atoms[j].y, atoms[j].z};
    Vec3 v_ij = pj - pi;
    double pol_i = polarizabilities(i);
    double pol_j = polarizabilities(j);
    double factor = pol_i * pol_j / (moments(0, i) * pol_j + moments(0, j) * pol_i);
    double rij = v_ij.norm();

    if (rij < 1e-15) return;

    double rij2 = rij * rij;
    double rij4 = rij2 * rij2;
    double rij6 = rij4 * rij2;
    double rij8 = rij4 * rij4;
    double rij10 = rij4 * rij6;

    double c6 = factor * moments(0, i) * moments(0, j);
    double c8 = 1.5 * factor * (moments(0, i) * moments(1, j) + moments(1, i) * moments(0, j));
    double c10 = 2.0 * factor * (moments(0, i) * moments(2, j) + moments(2, i) * moments(0, j)) +
                 4.2 * factor * moments(1, i) * moments(1, j);
    double rc = (std::sqrt(c8 / c6) + std::sqrt(std::sqrt(c10 / c6)) + std::sqrt(c10 / c8)) / 3;
    double rvdw = params.a1 * rc + params.a2 * occ::units::ANGSTROM_TO_BOHR;
    double rvdw2 = rvdw * rvdw;
    double rvdw4 = rvdw2 * rvdw2;
    double rvdw6 = rvdw4 * rvdw2;
    double rvdw8 = rvdw4 * rvdw4;
    double rvdw10 = rvdw4 * rvdw6;

    occ::log::debug("{:>3d} {:>3d} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}",
                    i, j, rij, c6, c8, c10, rc, rvdw);

    edisp -= c6 / (rvdw6 + rij6) + c8 / (rvdw8 + rij8) + c10 / (rvdw10 + rij10);

    double c6_com = 6.0 * c6 * rij4 / ((rvdw6 + rij6) * (rvdw6 + rij6));
    double c8_com = 8.0 * c8 * rij6 / ((rvdw8 + rij8) * (rvdw8 + rij8));
    double c10_com = 10.0 * c10 * rij8 / ((rvdw10 + rij10) * (rvdw10 + rij10));
    
    forces.col(i) += (c6_com + c8_com + c10_com) * v_ij;
    forces.col(j) -= (c6_com + c8_com + c10_com) * v_ij;
  });

  // Reduce results from all threads
  Mat3N forces = Mat3N::Zero(3, num_atoms);
  double edisp = 0.0;
  for (const auto &forces_thread : forces_local) {
    forces += forces_thread;
  }
  for (const auto &edisp_thread : edisp_local) {
    edisp += edisp_thread;
  }
  
  return {edisp, forces};
}

namespace impl {

void xdm_moment_kernel_restricted(
    Eigen::Ref<const Mat> r, Eigen::Ref<const Mat> rho,
    Eigen::Ref<const Vec> weights, Eigen::Ref<const Mat> rho_pro,
    Eigen::Ref<Vec> hirshfeld_charges, Eigen::Ref<Mat> volume,
    Eigen::Ref<Mat> moments, double &num_electrons,
    double &num_electrons_promol) {

  for (int j = 0; j < rho_pro.rows(); j++) {
    double protot = rho_pro.row(j).sum();
    if (protot < 1e-30)
      continue;
    // now it holds the weight function
    occ::RowVec hirshfeld_weights = rho_pro.row(j).array() / protot;
    double fac = 2 * rho(j, 0) * weights(j);
    hirshfeld_charges.array() -= hirshfeld_weights.array() * fac;
    num_electrons_promol += protot * weights(j);
    double lapl = rho(j, 4);
    double tau = 2 * rho(j, 5);
    double sigma =
        rho(j, 1) * rho(j, 1) + rho(j, 2) * rho(j, 2) + rho(j, 3) * rho(j, 3);
    double dsigs = tau - 0.25 * sigma / std::max(rho(j, 0), 1e-30);
    double q = (lapl - 2 * dsigs) / 6.0;

    double bhole = occ::xdm::becke_hole_br89(rho(j, 0), q, 1.0);
    const auto &rja = r.row(j).array();
    occ::RowVec r_sub_b =
        (rja - bhole)
            .unaryExpr([](double x) { return std::max(x, 0.0); })
            .transpose();
    moments.row(0).array() += 2 * hirshfeld_weights.array() *
                              (rja - r_sub_b.array()).pow(2) * rho(j, 0) *
                              weights(j);
    moments.row(1).array() += 2 * hirshfeld_weights.array() *
                              (rja.pow(2) - r_sub_b.array().pow(2)).pow(2) *
                              rho(j, 0) * weights(j);
    moments.row(2).array() += 2 * hirshfeld_weights.array() *
                              (rja.pow(3) - r_sub_b.array().pow(3)).pow(2) *
                              rho(j, 0) * weights(j);
    num_electrons += 2 * rho(j, 0) * weights(j);
    volume.array() += (hirshfeld_weights.array() * r.row(j).array() *
                       r.row(j).array() * r.row(j).array())
                          .transpose()
                          .array() *
                      2 * rho(j, 0) * weights(j);
  }
}

void xdm_moment_kernel_unrestricted(
    Eigen::Ref<const Mat> r, Eigen::Ref<const Mat> rho,
    Eigen::Ref<const Vec> weights, Eigen::Ref<const Mat> rho_pro,
    Eigen::Ref<Vec> hirshfeld_charges, Eigen::Ref<Mat> volume,
    Eigen::Ref<Mat> moments, double &num_electrons,
    double &num_electrons_promol) {

  const auto &rho_a = occ::qm::block::a(rho);
  const auto &rho_b = occ::qm::block::b(rho);
  Mat rho_tot = rho_a + rho_b;
  xdm_moment_kernel_restricted(r, rho_tot, weights, rho_pro, hirshfeld_charges,
                               volume, moments, num_electrons,
                               num_electrons_promol);
}

} // namespace impl

struct XDMResult {
  double energy{0.0};
};

XDM::XDM(const occ::qm::AOBasis &basis, int charge,
         const XDM::Parameters &params)
    : m_basis(basis), m_grid(basis), m_charge(charge), m_params(params) {
  for (int i = 0; i < basis.atoms().size(); i++) {
    m_atom_grids.push_back(m_grid.get_partitioned_atom_grid(i));
  }
  size_t num_grid_points = std::accumulate(
      m_atom_grids.begin(), m_atom_grids.end(), 0.0,
      [&](double tot, const auto &grid) { return tot + grid.points.cols(); });
  occ::log::debug("{} total grid points used for XDM", num_grid_points);
  m_atomic_ion = (m_atom_grids.size() == 1) && (m_charge != 0);
  std::vector<int> ox = {};
  if (m_atomic_ion) {
    ox.push_back(m_charge);
  }
  m_slater_basis =
      occ::slater::slaterbasis_for_atoms(m_basis.atoms(), "thakkar", ox);
}

double XDM::energy(const occ::qm::MolecularOrbitals &mo) {
  occ::log::debug("MO has {} alpha electrons {} beta electrons\n", mo.n_alpha,
                  mo.n_beta);
  occ::timing::start(occ::timing::category::xdm);
  populate_moments(mo);
  populate_polarizabilities();
  occ::log::debug("moments\n{}", format_matrix(m_moments));

  std::tie(m_energy, m_forces) = xdm_dispersion_energy(
      {m_basis.atoms(), m_polarizabilities, m_moments, m_volume, m_volume_free},
      m_params);
  occ::timing::stop(occ::timing::category::xdm);
  return m_energy;
}

const Mat3N &XDM::forces(const occ::qm::MolecularOrbitals &mo) {
  populate_moments(mo);
  populate_polarizabilities();

  std::tie(m_energy, m_forces) = xdm_dispersion_energy(
      {m_basis.atoms(), m_polarizabilities, m_moments, m_volume, m_volume_free},
      m_params);
  return m_forces;
}

void XDM::populate_moments(const occ::qm::MolecularOrbitals &mo) {
  if (m_density_matrix.size() != 0 &&
      occ::util::all_close(mo.D, m_density_matrix)) {
    return;
  }
  m_density_matrix = mo.D;

  // Dispatch to templated implementation based on spinorbital kind (like DFT does)
  if (mo.kind == occ::qm::SpinorbitalKind::Unrestricted) {
    populate_moments_impl<occ::qm::SpinorbitalKind::Unrestricted>(mo);
  } else {
    populate_moments_impl<occ::qm::SpinorbitalKind::Restricted>(mo);
  }
}

template <occ::qm::SpinorbitalKind spinorbital_kind>
void XDM::populate_moments_impl(const occ::qm::MolecularOrbitals &mo) {
  const auto &atoms = m_basis.atoms();
  const size_t num_atoms = atoms.size();

  occ::log::debug("XDM using {} wavefunction",
                  spinorbital_kind == occ::qm::SpinorbitalKind::Unrestricted ? "unrestricted" : "restricted");

  constexpr int max_derivative{2};
  constexpr int num_rows_factor = (spinorbital_kind == occ::qm::SpinorbitalKind::Unrestricted) ? 2 : 1;

  occ::log::debug("XDM: Processing {} atom grids", m_atom_grids.size());

  // Use thread-local storage for results only (like DFT does)
  occ::parallel::thread_local_storage<Vec> tl_hirshfeld_charges(Vec::Zero(num_atoms));
  occ::parallel::thread_local_storage<Vec> tl_volumes(Vec::Zero(num_atoms));
  occ::parallel::thread_local_storage<Vec> tl_free_volumes(Vec::Zero(num_atoms));
  occ::parallel::thread_local_storage<Mat> tl_moments(Mat::Zero(3, num_atoms));
  occ::parallel::thread_local_storage<double> tl_num_electrons(0.0);
  occ::parallel::thread_local_storage<double> tl_num_electrons_promol(0.0);

  // Process each atom grid sequentially, parallelize over blocks within each grid (like DFT)
  for (const auto &atom_grid : m_atom_grids) {
    const auto &atom_pts = atom_grid.points;
    const auto &atom_weights = atom_grid.weights;
    const size_t npt_total = atom_pts.cols();

    occ::log::debug("XDM: Processing grid with {} points", npt_total);

    // Adaptive grain size (like DFT)
    const size_t min_points_per_chunk = std::max(size_t(32), m_basis.nbf() / 4);
    const size_t max_chunks = occ::parallel::nthreads * 8;
    size_t adaptive_grain = std::max(min_points_per_chunk, npt_total / max_chunks);

    occ::log::debug("XDM: nthreads={}, npt_total={}, adaptive_grain={}, max_chunks={}",
                    occ::parallel::nthreads, npt_total, adaptive_grain, max_chunks);

    // Use static_partitioner to disable work stealing in WASM (more stable)
    tbb::static_partitioner partitioner;
    tbb::parallel_for(
        tbb::blocked_range<size_t>(0, npt_total, adaptive_grain),
        [&](const tbb::blocked_range<size_t> &range) {
            const size_t l = range.begin();
            const size_t u = range.end();
            const size_t npt = u - l;

            if (npt == 0) return;

            const auto &pts_block = atom_pts.middleCols(l, npt);
            const auto &weights_block = atom_weights.segment(l, npt);

            // Allocate working matrices locally (exactly like DFT does)
            Mat rho(num_rows_factor * npt, occ::density::num_components(max_derivative));
            Mat hirshfeld_weights = Mat::Zero(npt, num_atoms);
            Mat r = Mat::Zero(npt, num_atoms);

            // Evaluate basis locally (like DFT does)
            auto gto_vals = occ::gto::evaluate_basis(m_basis, pts_block, max_derivative);

            // Get thread-local results storage
            auto &hirshfeld_charges = tl_hirshfeld_charges.local();
            auto &volume = tl_volumes.local();
            auto &volume_free = tl_free_volumes.local();
            auto &moments = tl_moments.local();
            auto &num_e = tl_num_electrons.local();
            auto &num_e_promol = tl_num_electrons_promol.local();

            // Evaluate density (compile-time dispatch like DFT)
            occ::density::evaluate_density<max_derivative, spinorbital_kind>(
                mo.D, gto_vals, rho);

            // Calculate Hirshfeld weights
            for (int i = 0; i < num_atoms; i++) {
              const auto &sb = m_slater_basis[i];
              occ::Vec3 pos{atoms[i].x, atoms[i].y, atoms[i].z};
              r.col(i) = (pts_block.colwise() - pos).colwise().norm();
              const auto &ria = r.col(i).array();
              hirshfeld_weights.col(i) = sb.rho(r.col(i));
              volume_free(i) += (hirshfeld_weights.col(i).array() *
                                 weights_block.array() * ria * ria * ria).sum();
            }

            // Calculate moments (compile-time dispatch like DFT)
            if constexpr (spinorbital_kind == occ::qm::SpinorbitalKind::Unrestricted) {
              impl::xdm_moment_kernel_unrestricted(r, rho, weights_block, hirshfeld_weights,
                                                   hirshfeld_charges, volume, moments, num_e, num_e_promol);
            } else {
              impl::xdm_moment_kernel_restricted(r, rho, weights_block, hirshfeld_weights,
                                                 hirshfeld_charges, volume, moments, num_e, num_e_promol);
            }
        },
        partitioner
    );
  }

  occ::log::debug("XDM: Finished parallel_for, reducing results");

  m_hirshfeld_charges = Vec::Zero(num_atoms);
  const auto &ecp_electrons = m_basis.ecp_electrons();
  occ::log::debug("XDM: ecp_electrons.size()={}, num_atoms={}", ecp_electrons.size(), num_atoms);
  for (int i = 0; i < num_atoms; i++) {
    m_hirshfeld_charges(i) = static_cast<double>(atoms[i].atomic_number);
    if (ecp_electrons.size() > i) {  // Fixed: was >=, should be >
      m_hirshfeld_charges(i) -= ecp_electrons[i];
    }
  }
  double num_electrons{0.0};
  double num_electrons_promol{0.0};
  m_moments = Mat::Zero(3, num_atoms);
  m_volume = Vec::Zero(num_atoms);
  m_volume_free = Vec::Zero(num_atoms);

  // Reduce results from thread-local storage
  for (const auto &charges : tl_hirshfeld_charges) {
    m_hirshfeld_charges += charges;
  }
  for (const auto &vol : tl_volumes) {
    m_volume += vol;
  }
  for (const auto &vol_free : tl_free_volumes) {
    m_volume_free += vol_free;
  }
  for (const auto &mom : tl_moments) {
    m_moments += mom;
  }
  for (const auto &ne : tl_num_electrons) {
    num_electrons += ne;
  }
  for (const auto &ne_promol : tl_num_electrons_promol) {
    num_electrons_promol += ne_promol;
  }

  occ::log::debug("Num electrons {:20.12f}, promolecule {:20.12f}\n",
                  num_electrons, num_electrons_promol);
  occ::log::debug("Hirshfeld charges");
  for (int i = 0; i < m_hirshfeld_charges.rows(); i++) {
    occ::log::debug("Atom {}: {:12.5f}", i, m_hirshfeld_charges(i));
  }
}

void XDM::populate_polarizabilities() {
  m_polarizabilities = Vec(m_volume.rows());
  const auto &atoms = m_basis.atoms();
  for (int i = 0; i < m_polarizabilities.rows(); i++) {
    m_polarizabilities(i) = xdm_polarizability(
        atoms[i].atomic_number, m_volume(i), m_volume_free(i), m_atomic_ion);
  }
}

// Explicit template instantiations
template void XDM::populate_moments_impl<occ::qm::SpinorbitalKind::Restricted>(
    const occ::qm::MolecularOrbitals &mo);
template void XDM::populate_moments_impl<occ::qm::SpinorbitalKind::Unrestricted>(
    const occ::qm::MolecularOrbitals &mo);

} // namespace occ::xdm
