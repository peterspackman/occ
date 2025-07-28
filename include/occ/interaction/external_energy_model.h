#pragma once
#include <occ/crystal/crystal.h>
#include <occ/interaction/energy_model_base.h>
#include <nlohmann/json.hpp>
#include <string>

namespace occ::interaction {

struct ExternalEnergyOptions {
  std::string command;
  double energy_threshold{1e-8};  // Ignore interactions below this threshold (Hartree)
  int timeout_seconds{30};        // Timeout for external program calls
  std::string working_directory{"."}; // Working directory for external program
};

class ExternalEnergyModel : public EnergyModelBase {
public:
  explicit ExternalEnergyModel(const crystal::Crystal &crystal,
                               const ExternalEnergyOptions &options);
  
  // Backward compatibility constructor
  explicit ExternalEnergyModel(const crystal::Crystal &crystal,
                               const std::string &command);

  CEEnergyComponents compute_energy(const core::Dimer &dimer) override;
  Mat3N compute_electric_field(const core::Dimer &dimer) override;
  const std::vector<Vec> &partial_charges() const override;
  double coulomb_scale_factor() const override { return 1.0; }

  // Configuration methods
  void set_energy_threshold(double threshold) { m_options.energy_threshold = threshold; }
  void set_timeout(int seconds) { m_options.timeout_seconds = seconds; }
  double get_energy_threshold() const { return m_options.energy_threshold; }

private:
  crystal::Crystal m_crystal;
  ExternalEnergyOptions m_options;
  std::vector<double> m_monomer_energies;
  std::vector<Vec> m_partial_charges;

  double compute_single_point_energy(const core::Molecule &molecule) const;
  double call_external_program(const nlohmann::json &input) const;
  void validate_external_response(const nlohmann::json &response) const;
  nlohmann::json prepare_input_json(const core::Molecule &molecule, const std::string &task = "single_point_energy") const;
};

} // namespace occ::interaction