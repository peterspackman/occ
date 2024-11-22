#include <LBFGS.h>
#include <fmt/os.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/qm/chelpg.h>
#include <occ/qm/hf.h>
#include <occ/qm/wavefunction.h>

namespace occ::qm {

Vec chelpg_charges(const Wavefunction &wfn) {

  const double extra = 2.8 * occ::units::ANGSTROM_TO_BOHR;
  const double separation = 0.3 * occ::units::ANGSTROM_TO_BOHR;
  occ::log::debug("CHELPG extra = {:.3f}, separation = {:.3f}", extra,
                  separation);
  Mat3N atom_positions = wfn.positions();
  const int num_atoms = atom_positions.cols();

  Vec3 box_lower = atom_positions.rowwise().minCoeff();
  box_lower.array() -= extra;
  Vec3 box_upper = atom_positions.rowwise().maxCoeff();
  box_upper.array() += extra;

  size_t nx = std::ceil((box_upper.x() - box_lower.x()) / separation) + 1;
  double sep_x = (box_upper.x() - box_lower.x()) / nx;
  size_t ny = std::ceil((box_upper.y() - box_lower.y()) / separation) + 1;
  double sep_y = (box_upper.y() - box_lower.y()) / ny;
  size_t nz = std::ceil((box_upper.z() - box_lower.z()) / separation) + 1;
  double sep_z = (box_upper.z() - box_lower.z()) / nz;

  occ::log::debug("Bottom left corner: {:.3f} {:.3f} {:.3f}", box_lower.x(),
                  box_lower.y(), box_lower.z());
  occ::log::debug("Top right corner: {:.3f} {:.3f} {:.3f}", box_upper.x(),
                  box_upper.y(), box_upper.z());
  occ::log::info("Maximum num pts: {}", nx * ny * nz);
  occ::log::info("Separations (au): {:.3f} {:.3f} {:.3f}", sep_x, sep_y, sep_z);

  Mat3N grid_points(3, nx * ny * nz);
  Mat distances(nx * ny * nz, num_atoms);
  Eigen::Index current{0};
  Vec radii = wfn.atomic_numbers().unaryExpr([](int n) {
    return static_cast<double>(core::Element(n).van_der_waals_radius()) *
           occ::units::ANGSTROM_TO_BOHR;
  });
  Vec3 pt;
  for (size_t i = 0; i < nx; i++) {
    pt(0) = i * sep_x + box_lower.x();
    for (size_t j = 0; j < ny; j++) {
      pt(1) = j * sep_y + box_lower.y();
      for (size_t k = 0; k < nz; k++) {
        pt(2) = k * sep_z + box_lower.z();
        Vec r = (atom_positions.colwise() - pt).colwise().norm();
        if ((r.array() > extra).all())
          continue;
        if (((r.array() - radii.array()) < 0).any()) {
          continue;
        }
        distances.row(current) = r;
        grid_points.col(current) = pt;
        current++;
      }
    }
  }
  occ::log::info("Actual num points used in fit: {}", current);

  grid_points.conservativeResize(3, current);
  distances.conservativeResize(current, num_atoms);

  HartreeFock hf(wfn.basis);
  hf.set_system_charge(wfn.charge());
  Vec esp = hf.electronic_electric_potential_contribution(wfn.mo, grid_points) +
            hf.nuclear_electric_potential_contribution(grid_points);

  double net_charge = hf.system_charge();
  Mat inverse_distances = 1.0 / distances.array();
  Mat G = Mat::Zero(num_atoms, num_atoms);
  Vec e = Vec::Zero(num_atoms);
  double wk = 1.0;
  for (int i = 0; i < num_atoms; i++) {
    e(i) = wk * esp.dot(inverse_distances.col(i));
    for (int j = 0; j < num_atoms; j++) {
      for (int k = 0; k < inverse_distances.rows(); k++) {
        G(i, j) += wk * inverse_distances(k, i) * inverse_distances(k, j);
      }
    }
  }
  Mat Ginv = G.inverse();
  Vec g = Ginv * e;
  double alpha = (net_charge - g.sum()) / (Ginv.sum());
  for (int i = 0; i < num_atoms; i++) {
    g(i) += alpha * Ginv.row(i).sum();
  }
  double rmsd = (inverse_distances * g - esp).norm() / esp.size();
  occ::log::info("CHELPG RMSD: {:.5e}", rmsd);

  return g;
}

} // namespace occ::qm
