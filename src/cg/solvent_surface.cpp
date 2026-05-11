#include <occ/cg/solvent_surface.h>

namespace occ::cg {

double SolventSurface::total_energy() const { return energies.sum(); }

double SolventSurface::total_area() const { return areas.sum(); }

size_t SolventSurface::size() const { return positions.cols(); }

double SMDSolventSurfaces::total_energy() const {
  return coulomb.total_energy() + cds.total_energy() +
         electronic_energies.sum();
}

SMDSolventSurfaces
from_xtb_surfaces(const occ::xtb::SolvationSurfaces &xtb_surfaces) {
  SMDSolventSurfaces out;
  if (xtb_surfaces.coulomb) {
    const auto &c = *xtb_surfaces.coulomb;
    out.coulomb.positions = c.positions;
    out.coulomb.areas = c.areas;
    out.coulomb.energies = c.energies;
  }
  if (xtb_surfaces.cds) {
    const auto &d = *xtb_surfaces.cds;
    out.cds.positions = d.positions;
    out.cds.areas = d.areas;
    out.cds.energies = d.energies;
  }
  // The cg partitioner adds `coulomb.energies(i) + electronic_energies(i)` per
  // element; we've already folded everything into coulomb.energies, so this
  // stays at zero.
  out.electronic_energies = occ::Vec::Zero(out.coulomb.positions.cols());
  out.total_solvation_energy = xtb_surfaces.total_energy();
  return out;
}

} // namespace occ::cg
