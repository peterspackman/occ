#pragma once
#include <nlohmann/json.hpp>
#include <occ/cg/solvation_data.h>
#include <occ/io/eigen_json.h>

namespace occ::cg {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SurfaceField, name, values);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CavitySurface, name, positions, areas,
                                   energies, descriptors);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SolvationData, cavities,
                                   total_solvation_energy,
                                   electronic_contribution);

} // namespace occ::cg
