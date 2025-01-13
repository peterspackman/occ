#pragma once
#include <nlohmann/json.hpp>
#include <occ/crystal/crystal.h>
#include <occ/crystal/hkl.h>
#include <string>

namespace occ::main {

struct FacetEnergies {
  occ::crystal::HKL hkl;
  double offset{0.0};
  std::vector<std::vector<size_t>> interaction_energy_counts;
  double energy{0.0};
  double area{0.0};
};

struct CrystalSurfaceEnergies {
  occ::crystal::Crystal crystal;
  std::vector<double> unique_interaction_energies;
  std::vector<FacetEnergies> facets;
};

CrystalSurfaceEnergies
calculate_crystal_surface_energies(const std::string &filename,
                                   const occ::crystal::Crystal &crystal,
                                   const occ::crystal::CrystalDimers &uc_dimers,
                                   int max_number_of_surfaces, int sign = -1);

void to_json(nlohmann::json &j, const FacetEnergies &);
void to_json(nlohmann::json &j, const CrystalSurfaceEnergies &);

} // namespace occ::main
