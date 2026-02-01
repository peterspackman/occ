#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/dft/voronoi_charges.h>
#include <occ/gto/density.h>
#include <occ/gto/gto.h>

using occ::Mat;
using occ::Mat3N;
using occ::Vec;
using occ::Vec3;

namespace occ::dft {

namespace impl {

inline void voronoi_kernel_restricted(Eigen::Ref<const Mat> r,
                                      Eigen::Ref<const Mat> rho,
                                      Eigen::Ref<const Vec> weights,
                                      Eigen::Ref<const Mat> voronoi_weights,
                                      Eigen::Ref<Vec> voronoi_charges,
                                      Eigen::Ref<Vec> atom_volumes,
                                      double &num_electrons) {

  for (int j = 0; j < voronoi_weights.rows(); j++) {
    double fac = 2 * rho(j, 0) * weights(j);
    voronoi_charges.array() -= voronoi_weights.row(j).array() * fac;

    // Calculate atomic volumes (integral of electron density * r³)
    const auto &rja = r.row(j).array();
    atom_volumes.array() +=
        (voronoi_weights.row(j).array() * rja * rja * rja).transpose().array() *
        2 * rho(j, 0) * weights(j);

    num_electrons += 2 * rho(j, 0) * weights(j);
  }
}

inline void voronoi_kernel_unrestricted(Eigen::Ref<const Mat> r,
                                        Eigen::Ref<const Mat> rho,
                                        Eigen::Ref<const Vec> weights,
                                        Eigen::Ref<const Mat> voronoi_weights,
                                        Eigen::Ref<Vec> voronoi_charges,
                                        Eigen::Ref<Vec> atom_volumes,
                                        double &num_electrons) {

  const auto &rho_a = occ::qm::block::a(rho);
  const auto &rho_b = occ::qm::block::b(rho);
  Mat rho_tot = rho_a + rho_b;
  voronoi_kernel_restricted(r, rho_tot, weights, voronoi_weights,
                            voronoi_charges, atom_volumes, num_electrons);
}

} // namespace impl

VoronoiPartition::VoronoiPartition(const occ::gto::AOBasis &basis, int charge,
                                   double temperature, bool use_vdw_radii,
                                   const occ::io::GridSettings &grid_settings)
    : m_basis(basis), m_grid(basis, grid_settings), m_charge(charge),
      m_temperature(temperature), m_use_vdw_radii(use_vdw_radii) {

  for (size_t i = 0; i < basis.atoms().size(); i++) {
    m_atom_grids.push_back(m_grid.get_partitioned_atom_grid(i));
  }

  size_t num_grid_points = std::accumulate(
      m_atom_grids.begin(), m_atom_grids.end(), 0.0,
      [&](double tot, const auto &grid) { return tot + grid.points.cols(); });

  occ::log::debug("VoronoiPartition: {} total grid points", num_grid_points);
}

double VoronoiPartition::logsumexp_min(const Vec &distances) const {
  if (distances.size() == 0)
    return 0.0;

  Vec neg_scaled = -distances.array() / m_temperature;
  double max_val = neg_scaled.maxCoeff();

  // Compute LogSumExp with numerical stability
  double sum_exp = (neg_scaled.array() - max_val).exp().sum();
  return -m_temperature * (max_val + std::log(sum_exp));
}

Vec VoronoiPartition::compute_voronoi_weights(
    const Vec3 &point, const Mat3N &atom_positions,
    const Eigen::VectorXi &atomic_numbers) const {
  const size_t num_atoms = atom_positions.cols();
  Vec distances(num_atoms);

  // Compute distances from point to all atoms
  for (size_t i = 0; i < num_atoms; i++) {
    double dist = (point - atom_positions.col(i)).norm();

    if (m_use_vdw_radii) {
      // Scale distance by VDW radius
      double vdw_radius =
          occ::core::Element(atomic_numbers(i)).van_der_waals_radius() *
          occ::units::ANGSTROM_TO_BOHR;
      distances(i) = dist / vdw_radius;
    } else {
      distances(i) = dist;
    }
  }

  Vec weights = Vec::Zero(num_atoms);

  // Compute LogSumExp-based Voronoi weights
  for (size_t i = 0; i < num_atoms; i++) {
    Vec other_distances(num_atoms - 1);
    int idx = 0;
    for (size_t j = 0; j < num_atoms; j++) {
      if (j != i) {
        other_distances(idx++) = distances(j);
      }
    }

    double min_other = logsumexp_min(other_distances);
    double weight_factor =
        std::exp(-(distances(i) - min_other) / m_temperature);
    weights(i) = weight_factor / (1.0 + weight_factor);
  }

  // Normalize weights to sum to 1
  double sum_weights = weights.sum();
  if (sum_weights > 1e-12) {
    weights /= sum_weights;
  }

  return weights;
}

Vec VoronoiPartition::calculate(const occ::qm::MolecularOrbitals &mo) {
  // Only recalculate if density matrix has changed
  if (m_density_matrix.size() != 0 &&
      occ::util::all_close(mo.D, m_density_matrix)) {
    return m_voronoi_charges;
  }

  m_density_matrix = mo.D;
  compute_voronoi_weights(mo);
  return m_voronoi_charges;
}

void VoronoiPartition::compute_voronoi_weights(
    const occ::qm::MolecularOrbitals &mo) {
  int nthreads = occ::parallel::get_num_threads();
  const auto &atoms = m_basis.atoms();
  const size_t num_atoms = atoms.size();

  bool unrestricted = (mo.kind == occ::qm::SpinorbitalKind::Unrestricted);
  occ::log::debug("VoronoiPartition using {} wavefunction",
                  unrestricted ? "unrestricted" : "restricted");

  constexpr size_t BLOCKSIZE = 64;
  int num_rows_factor = unrestricted ? 2 : 1;
  constexpr int max_derivative{1}; // Only need density, not derivatives

  // Thread-local storage for parallel accumulation
  occ::parallel::thread_local_storage<Vec> tl_voronoi_charges(Vec::Zero(num_atoms));
  occ::parallel::thread_local_storage<Vec> tl_atom_volumes(Vec::Zero(num_atoms));  
  occ::parallel::thread_local_storage<double> tl_num_electrons(0.0);

  // Convert atom positions to Bohr for distance calculations
  Mat3N atom_positions(3, num_atoms);
  for (size_t i = 0; i < num_atoms; i++) {
    atom_positions.col(i) << atoms[i].x * occ::units::ANGSTROM_TO_BOHR,
        atoms[i].y * occ::units::ANGSTROM_TO_BOHR,
        atoms[i].z * occ::units::ANGSTROM_TO_BOHR;
  }

  for (const auto &atom_grid : m_atom_grids) {
    const auto &atom_pts = atom_grid.points;
    const auto &atom_weights = atom_grid.weights;
    const size_t npt_total = atom_pts.cols();
    const size_t num_blocks = npt_total / BLOCKSIZE + 1;

    occ::parallel::parallel_for(size_t(0), num_blocks, [&](size_t block) {
      occ::gto::GTOValues gto_vals;
      gto_vals.reserve(m_basis.nbf(), BLOCKSIZE, max_derivative);
      auto &voronoi_charges = tl_voronoi_charges.local();
      auto &atom_volumes = tl_atom_volumes.local();
      auto &num_e = tl_num_electrons.local();

      Eigen::Index l = block * BLOCKSIZE;
      Eigen::Index u = std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
      Eigen::Index npt = u - l;
      if (npt <= 0)
        return;

      Mat voronoi_weights = Mat::Zero(npt, num_atoms);
      Mat r = Mat::Zero(npt, num_atoms);
      Mat rho = Mat::Zero(num_rows_factor * npt,
                          occ::density::num_components(max_derivative));

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

      // Compute Voronoi weights for each grid point
      Eigen::VectorXi atomic_numbers(num_atoms);
      for (size_t i = 0; i < num_atoms; i++) {
        atomic_numbers(i) = atoms[i].atomic_number;
      }
      for (int j = 0; j < npt; j++) {
        Vec3 point = pts_block.col(j);
        Vec weights =
            compute_voronoi_weights(point, atom_positions, atomic_numbers);
        voronoi_weights.row(j) = weights.transpose();
      }

      // Compute distances for volume calculation
      for (int i = 0; i < num_atoms; i++) {
        Vec3 pos = atom_positions.col(i);
        r.col(i) = (pts_block.colwise() - pos).colwise().norm();
      }

      if (unrestricted) {
        impl::voronoi_kernel_unrestricted(r, rho, weights_block,
                                          voronoi_weights, voronoi_charges,
                                          atom_volumes, num_e);
      } else {
        impl::voronoi_kernel_restricted(r, rho, weights_block,
                                        voronoi_weights, voronoi_charges,
                                        atom_volumes, num_e);
      }
    });
  }

  // Initialize charges with nuclear charges
  m_voronoi_charges = Vec::Zero(num_atoms);
  const auto &ecp_electrons = m_basis.ecp_electrons();
  for (int i = 0; i < num_atoms; i++) {
    m_voronoi_charges(i) = static_cast<double>(atoms[i].atomic_number);
    if (ecp_electrons.size() >= i) {
      m_voronoi_charges(i) -= ecp_electrons[i];
    }
  }

  double num_electrons{0.0};
  m_atom_volumes = Vec::Zero(num_atoms);

  // Reduce thread-local results
  for (const auto& local_charges : tl_voronoi_charges) {
    m_voronoi_charges += local_charges;
  }
  
  for (const auto& local_volumes : tl_atom_volumes) {
    m_atom_volumes += local_volumes;
  }
  
  for (const auto& local_electrons : tl_num_electrons) {
    num_electrons += local_electrons;
  }

  occ::log::debug("Voronoi analysis: electrons in molecule = {:.6f}",
                  num_electrons);

  occ::log::debug("Voronoi charges:");
  for (int i = 0; i < m_voronoi_charges.rows(); i++) {
    occ::log::debug("  Atom {}: {:.6f}", i, m_voronoi_charges(i));
  }

  // Smear remaining charge across all atoms
  m_voronoi_charges.array() +=
      ((m_charge - m_voronoi_charges.sum()) / m_voronoi_charges.rows());
}

Vec calculate_voronoi_charges(const occ::gto::AOBasis &basis,
                              const occ::qm::MolecularOrbitals &mo, int charge,
                              double temperature, bool use_vdw_radii,
                              const occ::io::GridSettings &grid_settings) {
  VoronoiPartition voronoi(basis, charge, temperature, use_vdw_radii,
                           grid_settings);
  return voronoi.calculate(mo);
}

} // namespace occ::dft