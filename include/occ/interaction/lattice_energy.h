#pragma once
#include <occ/interaction/energy_model_base.h>
#include <occ/interaction/lattice_convergence_settings.h>
#include <occ/interaction/pair_energy.h>
#include <occ/interaction/polarization_partitioning.h>
#include <occ/interaction/wolf.h>

namespace occ::interaction {

struct LatticeEnergyResult {
  double lattice_energy{0.0};
  occ::crystal::CrystalDimers dimers;
  std::vector<CEEnergyComponents> energy_components;
  std::vector<polarization_partitioning::MoleculeCouplingResults> coupling_terms;
};

class LatticeEnergyCalculator {
public:
  LatticeEnergyCalculator(std::unique_ptr<EnergyModelBase> model,
                          const crystal::Crystal &crystal,
                          const std::string &basename,
                          LatticeConvergenceSettings settings);

  LatticeEnergyResult compute();

private:
  void initialize_wolf_sum();
  bool is_converged(double current, double previous) const;
  void report_convergence_statistics(const CEEnergyComponents &total);
  std::vector<polarization_partitioning::MoleculeCouplingResults>
  apply_crystal_field_polarization_partitioning(
    crystal::CrystalDimers &dimers,
    std::vector<CEEnergyComponents> &energies,
    double radius);

  std::unique_ptr<EnergyModelBase> m_energy_model;
  crystal::Crystal m_crystal;
  std::string m_basename;
  LatticeConvergenceSettings m_settings;
  std::unique_ptr<WolfSum> m_wolf_sum;
};

} // namespace occ::interaction
