#include <occ/cg/solvation_types.h>


namespace occ::cg {

void NeighborContribution::assign(const NeighborContribution &other) {
  coulomb = other.coulomb;
  cds = other.cds;
  area_coulomb = other.area_coulomb;
  area_cds = other.area_cds;
}

} // namespace occ::cg
