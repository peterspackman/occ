#include <occ/cg/result_types.h>

namespace occ::cg {

double DimerResult::total_energy() const {
  return energy_component(components::total);
}

void DimerResult::set_energy_component(const std::string &key, double value) {
  energy_components[key] = value;
}

double DimerResult::energy_component(const std::string &key) const {
  auto it = energy_components.find(key);
  return (it != energy_components.end()) ? it->second : 0.0;
}

bool DimerResult::has_energy_component(const std::string &key) const {
  return energy_components.contains(key);
}

double MoleculeResult::total_energy() const {
  return energy_component(components::total);
}

void MoleculeResult::set_energy_component(const std::string &key,
                                          double value) {
  energy_components[key] = value;
}

double MoleculeResult::energy_component(const std::string &key) const {
  auto it = energy_components.find(key);
  return (it != energy_components.end()) ? it->second : 0.0;
}

bool MoleculeResult::has_energy_component(const std::string &key) const {
  return energy_components.contains(key);
}

void MoleculeResult::add_dimer_result(const DimerResult &dimer) {
  if (dimer.is_nearest_neighbor) {
    energy_components[components::total] += dimer.total_energy();
    energy_components[components::crystal_total] +=
        dimer.energy_component(components::crystal_total);
  }
  dimer_results.push_back(dimer);
}

} // namespace occ::cg
