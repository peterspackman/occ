#include <occ/cg/solvation_data.h>

namespace occ::cg {

double CavitySurface::total_energy() const {
  double total = 0.0;
  for (const auto &field : energies)
    total += field.values.sum();
  return total;
}

double SolvationData::total_energy() const {
  double total = 0.0;
  for (const auto &cavity : cavities)
    total += cavity.total_energy();
  return total;
}

const CavitySurface *SolvationData::find(std::string_view name) const {
  for (const auto &cavity : cavities) {
    if (cavity.name == name)
      return &cavity;
  }
  return nullptr;
}

SolvationData to_solvation_data(const SMDSolventSurfaces &surfaces) {
  SolvationData out;
  out.total_solvation_energy = surfaces.total_solvation_energy;
  out.electronic_contribution = surfaces.electronic_contribution;
  out.gas_phase_contribution = surfaces.gas_phase_contribution;
  out.free_energy_correction = surfaces.free_energy_correction;

  if (surfaces.coulomb.size() > 0) {
    CavitySurface cavity;
    cavity.name = "coulomb";
    cavity.positions = surfaces.coulomb.positions;
    cavity.areas = surfaces.coulomb.areas;
    Vec electrostatic = surfaces.coulomb.energies;
    if (surfaces.electronic_energies.size() == electrostatic.size())
      electrostatic += surfaces.electronic_energies;
    cavity.energies.push_back({"coulomb", std::move(electrostatic)});
    out.cavities.push_back(std::move(cavity));
  }

  if (surfaces.cds.size() > 0) {
    CavitySurface cavity;
    cavity.name = "cds";
    cavity.positions = surfaces.cds.positions;
    cavity.areas = surfaces.cds.areas;
    cavity.energies.push_back({"cds", surfaces.cds.energies});
    out.cavities.push_back(std::move(cavity));
  }

  return out;
}

} // namespace occ::cg
