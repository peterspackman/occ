#pragma once
#include <nlohmann/json.hpp>
#include <occ/cg/solvent_surface.h>
#include <occ/io/eigen_json.h>

namespace occ::cg {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SolventSurface, positions, energies, areas);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SMDSolventSurfaces, coulomb, cds,
                                   electronic_energies, total_solvation_energy,
                                   electronic_contribution,
                                   gas_phase_contribution,
                                   free_energy_correction);

} // namespace occ::cg
