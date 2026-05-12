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
from_scrf_surfaces(const occ::scrf::SolvationSurfaces &scrf_surfaces) {
  SMDSolventSurfaces out;
  if (scrf_surfaces.coulomb) {
    const auto &c = *scrf_surfaces.coulomb;
    out.coulomb.positions = c.positions;
    out.coulomb.areas = c.areas;
    out.coulomb.energies = c.energies;
  }
  if (scrf_surfaces.cds) {
    const auto &d = *scrf_surfaces.cds;
    out.cds.positions = d.positions;
    out.cds.areas = d.areas;
    out.cds.energies = d.energies;
  }
  // The cg partitioner adds `coulomb.energies(i) + electronic_energies(i)` per
  // element; coulomb.energies already carries the full per-element ES
  // contribution (xTB: ½σ·φ direct; DFT: same value, algebraically equal to
  // the legacy nuc+elec+pol decomposition). The caller fills in
  // `electronic_energies` separately if any residual electronic contribution
  // needs to be smeared across the cavity (DFT free-energy pipeline).
  out.electronic_energies = occ::Vec::Zero(out.coulomb.positions.cols());
  out.total_solvation_energy = scrf_surfaces.total_energy();
  return out;
}

} // namespace occ::cg
