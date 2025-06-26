#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fstream>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/dft/hirshfeld.h>
#include <occ/driver/single_point.h>
#include <occ/interaction/wolf.h>
#include <occ/io/load_geometry.h>
#include <occ/io/occ_input.h>
#include <occ/main/cli_validators.h>
#include <occ/main/occ_embed.h>
#include <occ/main/version.h>
#include <occ/qm/chelpg.h>
#include <occ/qm/expectation.h>
#include <occ/qm/external_potential.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <optional>

namespace occ::main {

namespace fs = std::filesystem;
using occ::io::OccInput;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;

crystal::Crystal read_crystal_structure(const std::string &filename) {
  occ::timing::start(occ::timing::category::io);

  auto path = fs::path(filename);
  if (!fs::exists(path)) {
    throw std::runtime_error("Crystal file does not exist: " + filename);
  }

  occ::log::info("Reading crystal structure from {}", filename);
  auto crystal = occ::io::load_crystal(filename);

  occ::log::info("Loaded crystal with {} asymmetric atoms",
                 crystal.asymmetric_unit().size());
  occ::log::info("Space group: {}", crystal.space_group().symbol());

  occ::timing::stop(occ::timing::category::io);
  return crystal;
}

std::vector<double> calculate_charges(const Wavefunction &wfn,
                                      const std::string &charge_scheme) {
  occ::log::info("Calculating {} charges", charge_scheme);

  // Check if wavefunction is valid
  if (wfn.atoms.empty()) {
    occ::log::warn("Empty wavefunction, returning zero charges");
    return std::vector<double>();
  }

  Vec charges_vec;

  if (charge_scheme == "mulliken") {
    charges_vec = wfn.mulliken_charges();
  } else if (charge_scheme == "hirshfeld") {
    charges_vec =
        occ::dft::calculate_hirshfeld_charges(wfn.basis, wfn.mo, wfn.charge());
  } else if (charge_scheme == "chelpg") {
    charges_vec = occ::qm::chelpg_charges(wfn);
  } else {
    occ::log::error("Unknown charge scheme: {}. Using Mulliken.",
                    charge_scheme);
    charges_vec = wfn.mulliken_charges();
  }

  // Convert to std::vector<double> and print charges
  std::vector<double> charges(charges_vec.size());
  occ::log::info("{} charges:", charge_scheme);
  for (int i = 0; i < charges_vec.size(); i++) {
    charges[i] = charges_vec(i);
    occ::log::info("  Atom {:2d} ({}): {:8.4f}", i,
                   occ::core::Element(wfn.atoms[i].atomic_number).symbol(),
                   charges[i]);
  }

  return charges;
}

std::vector<occ::core::PointCharge> create_external_charges(
    const crystal::Crystal &crystal, const Vec &asymmetric_charges,
    const core::Molecule &target_molecule, double cutoff_radius) {

  occ::log::info("Creating external charge environment using crystal dimers");

  std::vector<occ::core::PointCharge> external_charges;

  // Get target molecule center for reference
  Mat3N target_pos = target_molecule.positions(); // already in Bohr
  Vec3 target_center = target_pos.rowwise().mean();

  occ::log::info("Target molecule center: ({:.3f}, {:.3f}, {:.3f}) Bohr",
                 target_center(0), target_center(1), target_center(2));

  // Get all dimers within cutoff using crystal methods
  auto crystal_dimers = crystal.symmetry_unique_dimers(cutoff_radius);

  occ::log::info("Found {} unique dimers within {:.2f} Å",
                 crystal_dimers.molecule_neighbors.size(), cutoff_radius);

  // Get the target molecule index directly from the molecule object
  int target_mol_idx = target_molecule.asymmetric_molecule_idx();

  occ::log::info("Target molecule is symmetry unique molecule index {}",
                 target_mol_idx);

  // Get neighbors of the target molecule and add their charges
  if (target_mol_idx < crystal_dimers.molecule_neighbors.size()) {
    const auto &neighbors = crystal_dimers.molecule_neighbors[target_mol_idx];

    occ::log::info("Target molecule has {} neighboring dimers",
                   neighbors.size());

    for (const auto &[dimer, unique_idx] : neighbors) {
      // Get the A and B molecules in the dimer for debugging
      const auto &mol_a = dimer.a();
      const auto &neighbor_mol = dimer.b();
      
      occ::log::debug("Processing dimer - unique_idx: {}", unique_idx);
      occ::log::debug("  Molecule A center: ({:.3f}, {:.3f}, {:.3f})", 
                      mol_a.centroid()(0), mol_a.centroid()(1), mol_a.centroid()(2));
      occ::log::debug("  Molecule B center: ({:.3f}, {:.3f}, {:.3f})", 
                      neighbor_mol.centroid()(0), neighbor_mol.centroid()(1), neighbor_mol.centroid()(2));
      
      // Check if molecule A is actually our target molecule
      Vec3 mol_a_center = mol_a.centroid();
      Vec3 target_to_mol_a = mol_a_center - target_center;
      occ::log::debug("  Distance from target to mol A: {:.3f} Bohr", target_to_mol_a.norm());
      
      // Add charges from each atom in the neighboring molecule
      auto neighbor_atoms = neighbor_mol.atoms(); // Get the atoms vector
      for (size_t atom_idx = 0; atom_idx < neighbor_mol.size(); atom_idx++) {
        const auto &atom = neighbor_atoms[atom_idx];
        Vec3 atom_pos_bohr(atom.x, atom.y,
                           atom.z); // atom positions already in Bohr

        // Get the asymmetric unit index for this atom to get its charge
        auto asym_indices = neighbor_mol.asymmetric_unit_idx();
        if (atom_idx < asym_indices.size()) {
          int asym_idx = asym_indices[atom_idx];
          double charge = asymmetric_charges[asym_idx];
          
          // Debug output for each charge
          Vec3 dist_vec = atom_pos_bohr - target_center;
          double distance_bohr = dist_vec.norm();
          double distance_angstrom = distance_bohr / occ::units::ANGSTROM_TO_BOHR;
          
          occ::log::debug("    Atom {}: pos=({:.3f}, {:.3f}, {:.3f}) Bohr, asym_idx={}, charge={:.3f}, dist={:.3f} Å", 
                         atom_idx, atom_pos_bohr(0), atom_pos_bohr(1), atom_pos_bohr(2), 
                         asym_idx, charge, distance_angstrom);

          external_charges.emplace_back(charge, atom_pos_bohr);
        }
      }
    }
  }

  occ::log::info("Created {} external point charges from neighboring molecules",
                 external_charges.size());

  // Print statistics about the external charges
  if (!external_charges.empty()) {
    double total_charge = 0.0;
    double min_dist = 1e6, max_dist = 0.0;
    for (const auto &pc : external_charges) {
      total_charge += pc.charge();
      double dist =
          (pc.position() - target_center).norm() / occ::units::ANGSTROM_TO_BOHR;
      min_dist = std::min(min_dist, dist);
      max_dist = std::max(max_dist, dist);
    }
    occ::log::info("External charge statistics:");
    occ::log::info("  Total charge: {:.4f} e", total_charge);
    occ::log::info("  Distance range: {:.2f} - {:.2f} Å", min_dist, max_dist);

    // Debug: print first few positions for comparison
    occ::log::debug("Target center (Bohr): ({:.3f}, {:.3f}, {:.3f})",
                    target_center(0), target_center(1), target_center(2));
    if (external_charges.size() > 0) {
      Vec3 first_pc_pos = external_charges[0].position();
      occ::log::debug(
          "First external charge pos (Bohr): ({:.3f}, {:.3f}, {:.3f})",
          first_pc_pos(0), first_pc_pos(1), first_pc_pos(2));
      Vec3 diff = first_pc_pos - target_center;
      occ::log::debug("Difference vector (Bohr): ({:.3f}, {:.3f}, {:.3f})",
                      diff(0), diff(1), diff(2));
      occ::log::debug("Distance (Bohr): {:.3f}, (Å): {:.3f}", diff.norm(),
                      diff.norm() / occ::units::ANGSTROM_TO_BOHR);
    }

    // Write external charges to XYZ file for visualization
    std::string xyz_filename = "external_charges.xyz";
    std::ofstream xyz_file(xyz_filename);
    if (xyz_file.is_open()) {
      // Write target molecule first (with different element symbols for
      // clarity)
      int total_atoms = target_molecule.size() + external_charges.size();
      xyz_file << total_atoms << "\n";
      xyz_file
          << "Target molecule (uppercase) + External charges (lowercase)\n";

      // Target molecule atoms (uppercase symbols)
      auto target_atoms = target_molecule.atoms(); // Get the atoms vector
      for (size_t i = 0; i < target_molecule.size(); i++) {
        const auto &atom = target_atoms[i];
        Vec3 pos_angstrom =
            Vec3(atom.x, atom.y, atom.z) / occ::units::ANGSTROM_TO_BOHR;
        std::string symbol = occ::core::Element(atom.atomic_number).symbol();
        xyz_file << symbol << " " << pos_angstrom(0) << " " << pos_angstrom(1)
                 << " " << pos_angstrom(2) << "\n";
      }

      // External charges (lowercase symbols, charge in comment)
      for (const auto &pc : external_charges) {
        Vec3 pos_angstrom = pc.position() / occ::units::ANGSTROM_TO_BOHR;
        // Use dummy element based on charge sign for visualization
        std::string symbol =
            (pc.charge() > 0) ? "he" : "ne"; // positive=he, negative=ne
        xyz_file << symbol << " " << pos_angstrom(0) << " " << pos_angstrom(1)
                 << " " << pos_angstrom(2) << "  # charge=" << pc.charge()
                 << "\n";
      }
      xyz_file.close();
      occ::log::info("External charges written to {}", xyz_filename);
    }
  }

  return external_charges;
}

Wavefunction perform_embedded_scf(
    const core::Molecule &molecule,
    const std::vector<occ::core::PointCharge> &external_charges,
    const std::vector<double> &molecular_charges, const EmbedConfig &config,
    const std::optional<Wavefunction> &initial_guess = {}, int net_charge = 0,
    int multiplicity = 1) {

  occ::log::info("Performing embedded SCF calculation with {} external charges",
                 external_charges.size());

  // Create basis set for this molecule
  auto basis = qm::AOBasis::load(molecule.atoms(), config.basis_name);
  basis.set_pure(config.basis_spherical);

  if (!external_charges.empty()) {
    // Create HF procedure
    qm::HartreeFock hf(basis);
    
    // Use appropriate wrapper based on configuration
    if (config.use_wolf_sum) {
      occ::log::info("Setting up SCF with Wolf sum external potential");
      qm::WolfSumCorrectedProcedure<qm::HartreeFock> wolf_hf(
          hf, external_charges, molecular_charges, config.wolf_alpha, config.wolf_cutoff);
      qm::SCF<qm::WolfSumCorrectedProcedure<qm::HartreeFock>> scf(wolf_hf, SpinorbitalKind::Restricted);
      
      // Set charge and multiplicity
      scf.set_charge_multiplicity(net_charge, multiplicity);
      
      // Set initial guess from previous cycle if available
      if (initial_guess.has_value()) {
        occ::log::info("Using wavefunction from previous cycle as initial guess");
        scf.set_initial_guess_from_wfn(initial_guess.value());
      }
      
      // Run SCF
      double energy = scf.compute_scf_energy();
      auto wfn = scf.wavefunction();
      
      occ::log::info("Embedded SCF with Wolf potential converged. Total energy: {:.8f} Hartree", energy);
      return wfn;
      
    } else {
      occ::log::info("Setting up SCF with point charge external potential");
      qm::PointChargeCorrectedProcedure<qm::HartreeFock> pc_hf(hf, external_charges);
      qm::SCF<qm::PointChargeCorrectedProcedure<qm::HartreeFock>> scf(pc_hf, SpinorbitalKind::Restricted);
      
      // Set charge and multiplicity
      scf.set_charge_multiplicity(net_charge, multiplicity);
      
      // Set initial guess from previous cycle if available
      if (initial_guess.has_value()) {
        occ::log::info("Using wavefunction from previous cycle as initial guess");
        scf.set_initial_guess_from_wfn(initial_guess.value());
      }
      
      // Run SCF
      double energy = scf.compute_scf_energy();
      auto wfn = scf.wavefunction();
      
      occ::log::info("Embedded SCF with point charges converged. Total energy: {:.8f} Hartree", energy);
      return wfn;
    }
  } else {
    // No external charges, just do gas phase
    OccInput embed_input;
    embed_input.method.name = config.method_name;
    embed_input.basis.name = config.basis_name;
    embed_input.basis.spherical = true;
    embed_input.electronic.charge = net_charge;
    embed_input.electronic.multiplicity = multiplicity;
    embed_input.electronic.spinorbital_kind = SpinorbitalKind::Restricted;
    embed_input.geometry.set_molecule(molecule);
    
    auto wfn = occ::driver::single_point(embed_input);
    occ::log::info("Gas phase SCF converged. Total energy: {:.8f} Hartree",
                   wfn.energy.total);
    return wfn;
  }
}

void run_self_consistent_embedding(const crystal::Crystal &crystal,
                                   const EmbedConfig &config) {

  occ::log::info("Starting self-consistent embedding calculation");

  // Get symmetry unique molecules or atoms
  std::vector<core::Molecule> unique_molecules =
      crystal.symmetry_unique_molecules();
  occ::log::info("Found {} symmetry unique molecules", unique_molecules.size());

  // Initialize charges for each unique molecule
  std::vector<std::vector<double>> molecular_charges(unique_molecules.size());
  std::vector<Wavefunction> wavefunctions(unique_molecules.size());
  std::vector<double> gas_phase_energies(unique_molecules.size());

  // Initial gas phase calculations to get starting charges
  occ::log::info("Performing initial gas phase calculations");
  for (size_t i = 0; i < unique_molecules.size(); i++) {
    const auto &mol = unique_molecules[i];

    // Get charge and multiplicity for this molecule/atom
    int net_charge =
        (i < config.net_charges.size()) ? config.net_charges[i] : 0;
    int multiplicity =
        (i < config.multiplicities.size()) ? config.multiplicities[i] : 1;

    occ::log::info("Gas phase calculation for {} {} (charge: {}, mult: {})",
                   config.atomic_mode ? "atom" : "molecule", i, net_charge,
                   multiplicity);

    // Set up OccInput for gas phase calculation
    OccInput gas_input;
    gas_input.method.name = config.method_name;
    gas_input.basis.name = config.basis_name;
    gas_input.basis.spherical = true;
    gas_input.electronic.charge = net_charge;
    gas_input.electronic.multiplicity = multiplicity;
    gas_input.electronic.spinorbital_kind = SpinorbitalKind::Restricted;
    gas_input.geometry.set_molecule(mol);

    // Perform gas phase SCF calculation
    auto gas_wfn = occ::driver::single_point(gas_input);
    wavefunctions[i] = gas_wfn;
    gas_phase_energies[i] = gas_wfn.energy.total;

    // Calculate initial charges
    molecular_charges[i] = calculate_charges(gas_wfn, config.charge_scheme);

    occ::log::info("Gas phase energy for molecule {}: {:.8f} Hartree", i,
                   gas_wfn.energy.total);
  }

  // Create asymmetric unit charges
  Vec asymmetric_charges = Vec::Zero(crystal.asymmetric_unit().size());
  for (size_t i = 0; i < unique_molecules.size(); i++) {
    const auto &mol = unique_molecules[i];
    const auto &asym_indices = mol.asymmetric_unit_idx();

    for (int j = 0; j < mol.size(); j++) {
      asymmetric_charges(asym_indices(j)) = molecular_charges[i][j];
    }
  }

  // Self-consistent embedding cycle
  std::vector<double> prev_energies(unique_molecules.size(), 0.0);

  for (int cycle = 0; cycle < config.max_embed_cycles; cycle++) {
    occ::log::info("Embedding cycle {}", cycle + 1);

    bool charges_converged = true;
    bool energies_converged = true;

    for (size_t i = 0; i < unique_molecules.size(); i++) {
      const auto &mol = unique_molecules[i];

      // Create external charges for this molecule
      auto external_charges = create_external_charges(
          crystal, asymmetric_charges, mol, config.wolf_cutoff);

      // Get charge and multiplicity for this molecule/atom
      int net_charge =
          (i < config.net_charges.size()) ? config.net_charges[i] : 0;
      int multiplicity =
          (i < config.multiplicities.size()) ? config.multiplicities[i] : 1;

      // Perform embedded SCF (use previous wavefunction as guess if available)
      std::optional<Wavefunction> initial_guess;
      if (cycle > 0) {
        initial_guess = wavefunctions[i];
      }
      auto embedded_wfn =
          perform_embedded_scf(mol, external_charges, molecular_charges[i],
                               config, initial_guess, net_charge, multiplicity);

      // Calculate new charges
      auto new_charges = calculate_charges(embedded_wfn, config.charge_scheme);

      // Check charge convergence
      double max_charge_change = 0.0;
      for (size_t j = 0; j < new_charges.size(); j++) {
        double change = std::abs(new_charges[j] - molecular_charges[i][j]);
        max_charge_change = std::max(max_charge_change, change);
      }

      if (max_charge_change > config.charge_convergence) {
        charges_converged = false;
      }

      // Check energy convergence
      double energy_change =
          std::abs(embedded_wfn.energy.total - prev_energies[i]);
      if (energy_change > config.energy_convergence) {
        energies_converged = false;
      }

      occ::log::info(
          "Molecule {} - Energy: {:.8f} Ha, Max charge change: {:.6f}", i,
          embedded_wfn.energy.total, max_charge_change);

      // Print charge changes for monitoring
      if (cycle > 0) {
        occ::log::info("Charge changes for molecule {}:", i);
        for (size_t j = 0; j < new_charges.size(); j++) {
          double change = new_charges[j] - molecular_charges[i][j];
          if (std::abs(change) > 0.01) { // Only print significant changes
            occ::log::info("  Atom {:2d}: {:.4f} -> {:.4f} (change: {:+.4f})",
                           j, molecular_charges[i][j], new_charges[j], change);
          }
        }
      }

      // Update charges and wavefunction
      molecular_charges[i] = new_charges;
      wavefunctions[i] = embedded_wfn;
      prev_energies[i] = embedded_wfn.energy.total;

      // Update asymmetric charges immediately for next molecules in this cycle
      const auto &asym_indices = mol.asymmetric_unit_idx();
      for (int j = 0; j < mol.size(); j++) {
        asymmetric_charges(asym_indices(j)) = new_charges[j];
      }
    }

    if (charges_converged && energies_converged) {
      occ::log::info("Embedding converged after {} cycles", cycle + 1);
      break;
    }

    if (cycle == config.max_embed_cycles - 1) {
      occ::log::warn("Embedding did not converge within {} cycles",
                     config.max_embed_cycles);
    }
  }

  // Write output files
  for (size_t i = 0; i < unique_molecules.size(); i++) {
    std::string filename =
        fmt::format("{}_{}.owf.json", config.output_prefix, i);
    occ::log::info("Writing embedded wavefunction to {}", filename);
    wavefunctions[i].save(filename);
  }

  // Print energy differences summary
  occ::log::info("Energy differences (Embedded - Gas phase):");
  for (size_t i = 0; i < unique_molecules.size(); i++) {
    double energy_diff = wavefunctions[i].energy.total - gas_phase_energies[i];
    occ::log::info("  Molecule {}: {:.8f} Hartree ({:.2f} kJ/mol)", 
                   i, energy_diff, energy_diff * 2625.50);
  }

  occ::log::info("Self-consistent embedding calculation completed");
}

CLI::App *add_embed_subcommand(CLI::App &app) {
  auto config = std::make_shared<EmbedConfig>();

  CLI::App *embed =
      app.add_subcommand("embed", "Perform embedded SCF calculations on "
                                  "symmetry unique molecules in a crystal");
  embed->fallthrough();

  // Required CIF input
  CLI::Option *cif_option = embed->add_option("cif", config->cif_filename,
                                              "CIF crystal structure file");
  cif_option->check(CLI::ExistingFile);
  cif_option->required();

  // Optional parameters
  embed->add_option("--method", config->method_name,
                    "quantum mechanical method");
  embed->add_option("--basis", config->basis_name, "basis set name");
  embed->add_option("--charge-scheme", config->charge_scheme,
                    "charge analysis scheme (mulliken, hirshfeld, chelpg)");
  embed->add_flag("--wolf,!--no-wolf", config->use_wolf_sum,
                  "use Wolf sum instead of regular point charges");
  embed->add_flag("--atomic", config->atomic_mode,
                  "treat each atom as separate molecule (for ionic crystals)");
  embed->add_option("--net-charges", config->net_charges,
                    "net charges for each molecule/atom (comma-separated)");
  embed->add_option("--multiplicities", config->multiplicities,
                    "multiplicities for each molecule/atom (comma-separated)");
  embed->add_option("--wolf-alpha", config->wolf_alpha,
                    "Wolf sum damping parameter");
  embed->add_option("--wolf-cutoff", config->wolf_cutoff,
                    "Wolf sum cutoff radius");
  embed->add_option("--max-cycles", config->max_embed_cycles,
                    "maximum embedding cycles");
  embed->add_option("--output-prefix", config->output_prefix,
                    "output file prefix");

  embed->callback([config]() { run_embed_subcommand(*config); });

  return embed;
}

void run_embed_subcommand(const EmbedConfig &config) {
  occ::main::print_header();
  
  occ::log::info("Starting embedding calculation with settings:");
  occ::log::info("  Method: {}", config.method_name);
  occ::log::info("  Basis: {}", config.basis_name);
  occ::log::info("  Charge scheme: {}", config.charge_scheme);
  occ::log::info("  Mode: {}", config.atomic_mode ? "Atomic" : "Molecular");
  occ::log::info("  External potential: {}",
                 config.use_wolf_sum ? "Wolf sum" : "Point charges");
  if (config.use_wolf_sum) {
    occ::log::info("  Wolf alpha: {:.2f} Å⁻¹", config.wolf_alpha);
    occ::log::info("  Wolf cutoff: {:.1f} Å", config.wolf_cutoff);
  }
  if (!config.net_charges.empty()) {
    occ::log::info("  Net charges: [{}]", occ::util::join(config.net_charges, ", "));
  }
  if (!config.multiplicities.empty()) {
    occ::log::info("  Multiplicities: [{}]",
                   occ::util::join(config.multiplicities, ", "));
  }
  occ::log::info("  Max cycles: {}", config.max_embed_cycles);

  // Read crystal structure
  crystal::Crystal crystal = read_crystal_structure(config.cif_filename);
  crystal.set_connectivity_criteria(!config.atomic_mode);

  // Run the self-consistent embedding calculation
  run_self_consistent_embedding(crystal, config);
}

} // namespace occ::main
