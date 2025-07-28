#include <occ/interaction/external_energy_model.h>
#include <occ/io/core_json.h>
#include <occ/core/log.h>
#include <occ/3rdparty/subprocess.hpp>
#include <nlohmann/json.hpp>
#include <sstream>
#include <filesystem>
#include <ctime>
#include <cmath>

namespace occ::interaction {

ExternalEnergyModel::ExternalEnergyModel(const crystal::Crystal &crystal,
                                         const ExternalEnergyOptions &options)
    : m_crystal(crystal), m_options(options) {

  if (m_options.command.empty()) {
    throw std::invalid_argument("External command cannot be empty");
  }

  occ::log::info("Initializing ExternalEnergyModel with command: {}", m_options.command);
  occ::log::info("Energy threshold: {} Hartree", m_options.energy_threshold);

  // Compute monomer energies for all symmetry unique molecules
  for (const auto &mol : crystal.symmetry_unique_molecules()) {
    double energy = compute_single_point_energy(mol);
    m_monomer_energies.push_back(energy);
    
    // For now, set empty partial charges (external program could provide these)
    Vec charges = Vec::Zero(mol.size());
    m_partial_charges.push_back(charges);
  }
}

// Backward compatibility constructor
ExternalEnergyModel::ExternalEnergyModel(const crystal::Crystal &crystal,
                                         const std::string &command)
    : ExternalEnergyModel(crystal, ExternalEnergyOptions{command}) {
}

CEEnergyComponents ExternalEnergyModel::compute_energy(const core::Dimer &dimer) {
  core::Molecule mol_A = dimer.a();
  core::Molecule mol_B = dimer.b();

  // Create dimer molecule
  core::Molecule dimer_mol(dimer.atomic_numbers(), dimer.positions());
  
  double e_a = m_monomer_energies[mol_A.asymmetric_molecule_idx()];
  double e_b = m_monomer_energies[mol_B.asymmetric_molecule_idx()];
  double e_ab = compute_single_point_energy(dimer_mol);

  CEEnergyComponents result;
  double interaction_energy = e_ab - e_a - e_b;
  
  // Apply energy threshold - ignore very small interactions to reduce numerical noise
  if (std::abs(interaction_energy) < m_options.energy_threshold) {
    occ::log::trace("Interaction energy {} below threshold {}, setting to zero", 
                    interaction_energy, m_options.energy_threshold);
    interaction_energy = 0.0;
  }
  
  result.total = interaction_energy;
  result.is_computed = true;
  return result;
}

Mat3N ExternalEnergyModel::compute_electric_field(const core::Dimer &) {
  // External model doesn't provide electric field calculations by default
  return Mat3N::Zero(3, 1);
}

const std::vector<Vec> &ExternalEnergyModel::partial_charges() const {
  return m_partial_charges;
}

double ExternalEnergyModel::compute_single_point_energy(const core::Molecule &molecule) const {
  nlohmann::json input = prepare_input_json(molecule);
  return call_external_program(input);
}

nlohmann::json ExternalEnergyModel::prepare_input_json(const core::Molecule &molecule, const std::string &task) const {
  nlohmann::json input;
  input["molecule"] = molecule;
  input["task"] = task;
  
  // Add metadata that external programs might find useful
  input["metadata"] = {
    {"source", "OCC ExternalEnergyModel"},
    {"timestamp", std::time(nullptr)},
    {"num_atoms", molecule.size()}
  };
  
  return input;
}

void ExternalEnergyModel::validate_external_response(const nlohmann::json &response) const {
  if (response.contains("error")) {
    throw std::runtime_error("External program error: " + response["error"].get<std::string>());
  }
  
  if (!response.contains("energy")) {
    throw std::runtime_error("External program response missing 'energy' field");
  }
  
  if (!response["energy"].is_number()) {
    throw std::runtime_error("External program returned non-numeric energy");
  }
  
  double energy = response["energy"].get<double>();
  if (!std::isfinite(energy)) {
    throw std::runtime_error("External program returned non-finite energy: " + std::to_string(energy));
  }
}

double ExternalEnergyModel::call_external_program(const nlohmann::json &input) const {
  using subprocess::CompletedProcess;
  using subprocess::PipeOption;
  
  std::string input_str = input.dump();
  
  // Parse command into arguments (simple space-based splitting)
  std::vector<std::string> command_line;
  std::istringstream iss(m_options.command);
  std::string token;
  while (std::getline(iss, token, ' ')) {
    if (!token.empty()) {
      command_line.push_back(token);
    }
  }
  
  if (command_line.empty()) {
    throw std::runtime_error("Empty command provided to ExternalEnergyModel");
  }
  
  // Set up subprocess options
  subprocess::RunOptions options;
  options.cin = input_str;  // Pass JSON to stdin
  options.cout = subprocess::PipeOption::pipe;
  options.cerr = subprocess::PipeOption::pipe;
  if (!m_options.working_directory.empty()) {
    options.cwd = m_options.working_directory;
  }
  
  occ::log::trace("Running external command: {}", m_options.command);
  
  auto process = subprocess::run(command_line, options);
  
  if (process.returncode != 0) {
    std::string error_msg = fmt::format(
      "External energy command failed with return code {}\n"
      "Command: {}\n"
      "Stdout: {}\n"
      "Stderr: {}",
      process.returncode, m_options.command, process.cout, process.cerr
    );
    occ::log::error("{}", error_msg);
    throw std::runtime_error("External energy calculation failed");
  }
  
  // Parse and validate JSON response
  try {
    nlohmann::json response = nlohmann::json::parse(process.cout);
    validate_external_response(response);
    return response["energy"].get<double>();
    
  } catch (const nlohmann::json::parse_error &e) {
    std::string error_msg = fmt::format(
      "Failed to parse JSON response from external program: {}\n"
      "Raw response: {}", e.what(), process.cout
    );
    occ::log::error("{}", error_msg);
    throw std::runtime_error("Failed to parse JSON response from external program");
  }
}

} // namespace occ::interaction