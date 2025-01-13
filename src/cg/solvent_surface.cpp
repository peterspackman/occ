#include <occ/cg/solvent_surface.h>

namespace occ::cg {

double SolventSurface::total_energy() const { return energies.sum(); }

double SolventSurface::total_area() const { return areas.sum(); }

size_t SolventSurface::size() const { return positions.cols(); }

double SMDSolventSurfaces::total_energy() const {
  return coulomb.total_energy() + cds.total_energy() +
         electronic_energies.sum();
}

} // namespace occ::cg
