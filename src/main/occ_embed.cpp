#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <fstream>
#include <numeric>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/progress.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/dft/hirshfeld.h>
#include <occ/dft/dft.h>
#include <occ/driver/single_point.h>
#include <occ/interaction/wolf.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/coulomb.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/interaction/wavefunction_transform.h>
#include <occ/io/load_geometry.h>
#include <occ/xdm/xdm.h>
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

// Helper function to compute electric field from point charges at given positions
Mat3N compute_point_charge_electric_field(
    const Vec &charges,
    const Mat3N &charge_positions,
    const Mat3N &target_positions) {

  size_t n_target = target_positions.cols();
  size_t n_charges = charge_positions.cols();

  Mat3N electric_field = Mat3N::Zero(3, n_target);

  for (size_t i = 0; i < n_target; i++) {
    Vec3 r_target = target_positions.col(i);

    for (size_t j = 0; j < n_charges; j++) {
      Vec3 r_charge = charge_positions.col(j);
      Vec3 r_vec = r_target - r_charge;
      double r = r_vec.norm();

      if (r < 1e-10) continue; // Skip self-interaction

      double q = charges(j);
      double r3 = r * r * r;

      // Classical Coulomb field: E = q * r_vec / r^3
      electric_field.col(i) += q * r_vec / r3;
    }
  }

  return electric_field;
}

// Compute electric field from a wavefunction at given positions
Mat3N compute_wavefunction_electric_field(
    const Wavefunction &wfn,
    const Mat3N &target_positions) {

  // Create HF object for computing fields
  qm::HartreeFock hf(wfn.basis);

  // Electronic contribution (from electron density)
  Mat3N E_electronic = hf.electronic_electric_field_contribution(wfn.mo, target_positions);

  // Nuclear contribution (from nuclei)
  Mat3N E_nuclear = hf.nuclear_electric_field_contribution(target_positions);

  // Debug: check for NaNs
  if (E_electronic.hasNaN()) {
    occ::log::warn("E_electronic has NaNs in wavefunction field calculation");
  }
  if (E_nuclear.hasNaN()) {
    occ::log::warn("E_nuclear has NaNs in wavefunction field calculation");
  }

  // Total field
  return E_electronic + E_nuclear;
}

struct PolarizationCorrectionResult {
  double correction_A{0.0};  // Correction for molecule A
  double correction_B{0.0};  // Correction for molecule B
  double total_correction{0.0};  // Total correction

  // Diagnostic information
  Vec field_mag_pc_A;        // |E_pc| at each atom in A (Coulomb)
  Vec field_mag_wolf_A;      // |E_wolf| at each atom in A (Wolf sum)
  Vec field_mag_wfn_A;       // |E_wfn| at each atom in A
  Vec field_mag_pc_B;        // |E_pc| at each atom in B (Coulomb)
  Vec field_mag_wolf_B;      // |E_wolf| at each atom in B (Wolf sum)
  Vec field_mag_wfn_B;       // |E_wfn| at each atom in B
  Vec per_atom_correction_A; // Per-atom corrections for A
  Vec per_atom_correction_B; // Per-atom corrections for B
};

// Compute polarization field correction for a dimer pair
PolarizationCorrectionResult compute_polarization_field_correction(
    Wavefunction wfn_A,
    Wavefunction wfn_B,
    const core::Molecule &mol_A,
    const core::Molecule &mol_B,
    const crystal::Crystal &crystal,
    const Vec &charges_A,
    const Mat3N &positions_A,
    const Vec &charges_B,
    const Mat3N &positions_B,
    const Vec &polarizabilities_A,
    const Vec &polarizabilities_B,
    double wolf_alpha,
    double wolf_cutoff) {

  PolarizationCorrectionResult result;

  size_t n_atoms_A = positions_A.cols();
  size_t n_atoms_B = positions_B.cols();

  // Transform wavefunctions to match dimer positions (avoid r=0 singularity)
  auto transform_A = interaction::transform::WavefunctionTransformer::calculate_transform(
      wfn_A, mol_A, crystal);
  auto transform_B = interaction::transform::WavefunctionTransformer::calculate_transform(
      wfn_B, mol_B, crystal);

  wfn_A = transform_A.wfn;
  wfn_B = transform_B.wfn;

  occ::log::debug("Transformed wfn_A RMSD: {:.6f} Å", transform_A.rmsd);
  occ::log::debug("Transformed wfn_B RMSD: {:.6f} Å", transform_B.rmsd);

  // Initialize diagnostic arrays
  result.field_mag_pc_A = Vec::Zero(n_atoms_A);
  result.field_mag_wolf_A = Vec::Zero(n_atoms_A);
  result.field_mag_wfn_A = Vec::Zero(n_atoms_A);
  result.field_mag_pc_B = Vec::Zero(n_atoms_B);
  result.field_mag_wolf_B = Vec::Zero(n_atoms_B);
  result.field_mag_wfn_B = Vec::Zero(n_atoms_B);
  result.per_atom_correction_A = Vec::Zero(n_atoms_A);
  result.per_atom_correction_B = Vec::Zero(n_atoms_B);

  // ==============================================
  // Correction for molecule A experiencing B's field
  // ==============================================

  // Compute field from B's point charges at A's atomic positions (Coulomb)
  Mat3N E_B_pc = compute_point_charge_electric_field(charges_B, positions_B, positions_A);

  // Compute field from B's point charges using Wolf sum (for comparison/diagnostics)
  interaction::WolfParameters wolf_params{wolf_cutoff, wolf_alpha};
  Mat3N E_B_wolf = interaction::wolf_electric_field(charges_B, positions_B, positions_A, wolf_params);

  // Compute field from B's wavefunction at A's atomic positions
  Mat3N E_B_wfn = compute_wavefunction_electric_field(wfn_B, positions_A);

  // Compute polarization correction for A
  // ΔE_pol_A = -0.5 * Σ_i α_i * |E_B_wfn - E_B_pc|²
  // This is a perturbative correction for swapping PC field → wavefunction field
  for (size_t i = 0; i < n_atoms_A; i++) {
    Vec3 E_pc = E_B_pc.col(i);
    Vec3 E_wolf = E_B_wolf.col(i);
    Vec3 E_wfn = E_B_wfn.col(i);

    // Store magnitudes for diagnostics
    result.field_mag_pc_A(i) = E_pc.norm();
    result.field_mag_wolf_A(i) = E_wolf.norm();
    result.field_mag_wfn_A(i) = E_wfn.norm();

    // Field difference (vector) - perturbation from PC → wavefunction
    Vec3 delta_E = E_wfn - E_pc;
    double delta_E_mag_sq = delta_E.squaredNorm();

    // Polarization correction for this atom
    double correction = -0.5 * polarizabilities_A(i) * delta_E_mag_sq;
    result.per_atom_correction_A(i) = correction;
    result.correction_A += correction;
  }

  // ==============================================
  // Correction for molecule B experiencing A's field
  // ==============================================

  // Compute field from A's point charges at B's atomic positions (Coulomb)
  Mat3N E_A_pc = compute_point_charge_electric_field(charges_A, positions_A, positions_B);

  // Compute field from A's point charges using Wolf sum (for comparison/diagnostics)
  Mat3N E_A_wolf = interaction::wolf_electric_field(charges_A, positions_A, positions_B, wolf_params);

  // Compute field from A's wavefunction at B's atomic positions
  Mat3N E_A_wfn = compute_wavefunction_electric_field(wfn_A, positions_B);

  // Compute polarization correction for B
  // ΔE_pol_B = -0.5 * Σ_i α_i * |E_A_wfn - E_A_pc|²
  // This is a perturbative correction for swapping PC field → wavefunction field
  for (size_t i = 0; i < n_atoms_B; i++) {
    Vec3 E_pc = E_A_pc.col(i);
    Vec3 E_wolf = E_A_wolf.col(i);
    Vec3 E_wfn = E_A_wfn.col(i);

    // Store magnitudes for diagnostics
    result.field_mag_pc_B(i) = E_pc.norm();
    result.field_mag_wolf_B(i) = E_wolf.norm();
    result.field_mag_wfn_B(i) = E_wfn.norm();

    // Field difference (vector) - perturbation from PC → wavefunction
    Vec3 delta_E = E_wfn - E_pc;
    double delta_E_mag_sq = delta_E.squaredNorm();

    // Polarization correction for this atom
    double correction = -0.5 * polarizabilities_B(i) * delta_E_mag_sq;
    result.per_atom_correction_B(i) = correction;
    result.correction_B += correction;
  }

  // Total correction
  result.total_correction = result.correction_A + result.correction_B;

  return result;
}

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
  } else if (charge_scheme == "hirshfeld" || charge_scheme == "xdm") {
    // Use XDM for both "hirshfeld" and "xdm" schemes
    // XDM provides consistent Hirshfeld charges + polarizabilities
    occ::xdm::XDM xdm(wfn.basis, wfn.charge());
    xdm.energy(wfn.mo); // This populates the charges
    charges_vec = xdm.hirshfeld_charges();
  } else if (charge_scheme == "chelpg") {
    charges_vec = occ::qm::chelpg_charges(wfn);
  } else {
    occ::log::error("Unknown charge scheme: {}. Using XDM Hirshfeld.",
                    charge_scheme);
    occ::xdm::XDM xdm(wfn.basis, wfn.charge());
    xdm.energy(wfn.mo);
    charges_vec = xdm.hirshfeld_charges();
  }

  // Convert to std::vector<double>
  std::vector<double> charges(charges_vec.size());
  for (int i = 0; i < charges_vec.size(); i++) {
    charges[i] = charges_vec(i);
  }

  // Constrain charges to sum to the correct net charge
  double charge_sum = std::accumulate(charges.begin(), charges.end(), 0.0);
  double expected_charge = static_cast<double>(wfn.charge());
  double charge_error = charge_sum - expected_charge;

  if (std::abs(charge_error) > 1e-6) {
    occ::log::warn("Unconstrained {} charges sum to {:.6f} (expected {:.0f}), applying constraint",
                   charge_scheme, charge_sum, expected_charge);
    // Distribute error evenly across all atoms
    double correction = charge_error / charges.size();
    for (auto &q : charges) {
      q -= correction;
    }
    double new_sum = std::accumulate(charges.begin(), charges.end(), 0.0);
    occ::log::info("After constraint: charges sum to {:.10f}", new_sum);
  }

  // Print charges
  occ::log::info("{} charges (constrained to sum to {:.0f}):", charge_scheme, expected_charge);
  for (int i = 0; i < charges.size(); i++) {
    occ::log::info("  Atom {:2d} ({}): {:8.4f}", i,
                   occ::core::Element(wfn.atoms[i].atomic_number).symbol(),
                   charges[i]);
  }

  return charges;
}

struct NeighborChargeInfo {
  size_t neighbor_idx;
  Vec charges;        // Charges for this neighbor's atoms
  Mat3N positions;    // Positions for this neighbor's atoms (Bohr)
};

struct ExternalChargesWithNeighbors {
  std::vector<occ::core::PointCharge> all_charges;
  std::vector<NeighborChargeInfo> per_neighbor_info;
};

ExternalChargesWithNeighbors create_external_charges_with_neighbors(
    const crystal::Crystal &crystal, const Vec &asymmetric_charges,
    const core::Molecule &target_molecule, double cutoff_radius) {

  occ::log::info("Creating external charge environment using crystal dimers");

  ExternalChargesWithNeighbors result;
  std::vector<occ::core::PointCharge> &external_charges = result.all_charges;

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

    size_t neighbor_count = 0;
    for (const auto &[dimer, unique_idx] : neighbors) {
      // Get the A and B molecules in the dimer for debugging
      const auto &mol_a = dimer.a();
      const auto &neighbor_mol = dimer.b();

      // Prepare per-neighbor info
      NeighborChargeInfo neighbor_info;
      neighbor_info.neighbor_idx = neighbor_count;
      neighbor_info.charges = Vec(neighbor_mol.size());
      neighbor_info.positions = Mat3N(3, neighbor_mol.size());

      occ::log::debug("Processing dimer - unique_idx: {}", unique_idx);
      occ::log::debug("  Molecule A center: ({:.3f}, {:.3f}, {:.3f})",
                      mol_a.centroid()(0), mol_a.centroid()(1),
                      mol_a.centroid()(2));
      occ::log::debug("  Molecule B center: ({:.3f}, {:.3f}, {:.3f})",
                      neighbor_mol.centroid()(0), neighbor_mol.centroid()(1),
                      neighbor_mol.centroid()(2));

      // Check if molecule A is actually our target molecule
      Vec3 mol_a_center = mol_a.centroid();
      Vec3 target_to_mol_a = mol_a_center - target_center;
      occ::log::debug("  Distance from target to mol A: {:.3f} Bohr",
                      target_to_mol_a.norm());

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
          double distance_angstrom =
              distance_bohr / occ::units::ANGSTROM_TO_BOHR;

          occ::log::debug("    Atom {}: pos=({:.3f}, {:.3f}, {:.3f}) Bohr, "
                          "asym_idx={}, charge={:.3f}, dist={:.3f} Å",
                          atom_idx, atom_pos_bohr(0), atom_pos_bohr(1),
                          atom_pos_bohr(2), asym_idx, charge,
                          distance_angstrom);

          external_charges.emplace_back(charge, atom_pos_bohr);

          // Store in per-neighbor info
          neighbor_info.charges(atom_idx) = charge;
          neighbor_info.positions.col(atom_idx) = atom_pos_bohr;
        }
      }

      result.per_neighbor_info.push_back(neighbor_info);
      neighbor_count++;
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

  occ::log::info("Tracked {} unique neighbors", result.per_neighbor_info.size());
  return result;
}

// Backward compatibility wrapper
std::vector<occ::core::PointCharge> create_external_charges(
    const crystal::Crystal &crystal, const Vec &asymmetric_charges,
    const core::Molecule &target_molecule, double cutoff_radius) {
  return create_external_charges_with_neighbors(crystal, asymmetric_charges,
                                                 target_molecule, cutoff_radius)
      .all_charges;
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
    // Determine if we're using DFT or HF
    bool is_dft = (config.method_name != "hf" && config.method_name != "rhf" &&
                   config.method_name != "uhf");

    if (is_dft) {
      // Create DFT procedure
      dft::DFT dft_proc(config.method_name, basis);

      // Use appropriate wrapper based on configuration
      if (config.use_wolf_sum) {
        occ::log::info("Setting up DFT SCF with Wolf sum external potential");
        qm::WolfSumCorrectedProcedure<dft::DFT> wolf_dft(
            dft_proc, external_charges, molecular_charges, config.wolf_alpha,
            config.wolf_cutoff);
        qm::SCF<qm::WolfSumCorrectedProcedure<dft::DFT>> scf(
            wolf_dft, SpinorbitalKind::Restricted);

        // Set charge and multiplicity
        scf.set_charge_multiplicity(net_charge, multiplicity);

        // Set initial guess from previous cycle if available
        if (initial_guess.has_value()) {
          occ::log::info(
              "Using wavefunction from previous cycle as initial guess");
          scf.set_initial_guess_from_wfn(initial_guess.value());
        }

        // Run SCF
        double energy = scf.compute_scf_energy();
        auto wfn = scf.wavefunction();

        occ::log::info("Embedded DFT SCF with Wolf potential converged. Total "
                       "energy: {:.8f} Hartree",
                       energy);
        return wfn;

      } else {
        occ::log::info("Setting up DFT SCF with point charge external potential");
        qm::PointChargeCorrectedProcedure<dft::DFT> pc_dft(
            dft_proc, external_charges);
        qm::SCF<qm::PointChargeCorrectedProcedure<dft::DFT>> scf(
            pc_dft, SpinorbitalKind::Restricted);

        // Set charge and multiplicity
        scf.set_charge_multiplicity(net_charge, multiplicity);

        // Set initial guess from previous cycle if available
        if (initial_guess.has_value()) {
          occ::log::info(
              "Using wavefunction from previous cycle as initial guess");
          scf.set_initial_guess_from_wfn(initial_guess.value());
        }

        // Run SCF
        double energy = scf.compute_scf_energy();
        auto wfn = scf.wavefunction();

        occ::log::info("Embedded DFT SCF with point charges converged. Total energy: "
                       "{:.8f} Hartree",
                       energy);
        return wfn;
      }
    } else {
      // Create HF procedure
      qm::HartreeFock hf(basis);

      // Use appropriate wrapper based on configuration
      if (config.use_wolf_sum) {
        occ::log::info("Setting up HF SCF with Wolf sum external potential");
        qm::WolfSumCorrectedProcedure<qm::HartreeFock> wolf_hf(
            hf, external_charges, molecular_charges, config.wolf_alpha,
            config.wolf_cutoff);
        qm::SCF<qm::WolfSumCorrectedProcedure<qm::HartreeFock>> scf(
            wolf_hf, SpinorbitalKind::Restricted);

        // Set charge and multiplicity
        scf.set_charge_multiplicity(net_charge, multiplicity);

        // Set initial guess from previous cycle if available
        if (initial_guess.has_value()) {
          occ::log::info(
              "Using wavefunction from previous cycle as initial guess");
          scf.set_initial_guess_from_wfn(initial_guess.value());
        }

        // Run SCF
        double energy = scf.compute_scf_energy();
        auto wfn = scf.wavefunction();

        occ::log::info("Embedded HF SCF with Wolf potential converged. Total "
                       "energy: {:.8f} Hartree",
                       energy);
        return wfn;

      } else {
        occ::log::info("Setting up HF SCF with point charge external potential");
        qm::PointChargeCorrectedProcedure<qm::HartreeFock> pc_hf(
            hf, external_charges);
        qm::SCF<qm::PointChargeCorrectedProcedure<qm::HartreeFock>> scf(
            pc_hf, SpinorbitalKind::Restricted);

        // Set charge and multiplicity
        scf.set_charge_multiplicity(net_charge, multiplicity);

        // Set initial guess from previous cycle if available
        if (initial_guess.has_value()) {
          occ::log::info(
              "Using wavefunction from previous cycle as initial guess");
          scf.set_initial_guess_from_wfn(initial_guess.value());
        }

        // Run SCF
        double energy = scf.compute_scf_energy();
        auto wfn = scf.wavefunction();

        occ::log::info("Embedded HF SCF with point charges converged. Total energy: "
                       "{:.8f} Hartree",
                       energy);
        return wfn;
      }
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
  std::vector<Wavefunction> gas_phase_wavefunctions(unique_molecules.size());
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
    gas_phase_wavefunctions[i] = gas_wfn;  // Save for later comparison
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

  // Compute effective pair energies and coupling terms after convergence
  occ::log::info("Computing pair energies and coupling terms");
  occ::log::info("Using pair energy radius: {:.2f} Å", config.pair_energy_radius);

  // Create CE energy models for both gas-phase and embedded wavefunctions
  interaction::CEEnergyModel ce_model_embedded(crystal, wavefunctions, wavefunctions);
  ce_model_embedded.set_model_name("ce-1p");

  interaction::CEEnergyModel ce_model_gas(crystal, gas_phase_wavefunctions, gas_phase_wavefunctions);
  ce_model_gas.set_model_name("ce-1p");

  // Get dimers from crystal (use smaller radius for CE calculations)
  auto crystal_dimers = crystal.symmetry_unique_dimers(config.pair_energy_radius);

  // Compute external potential energies for each embedded wavefunction
  // This is <ψ|V_ext|ψ> where V_ext is from point charges at all neighbor positions
  std::vector<double> external_potential_energies(unique_molecules.size(), 0.0);

  // Also store per-neighbor contributions for corrected pair energies
  // external_potential_per_neighbor[i][neighbor_idx] = V_ext contribution from that neighbor
  std::vector<std::vector<double>> external_potential_per_neighbor(unique_molecules.size());

  occ::log::info("Computing external potential energies for embedded wavefunctions");
  for (size_t i = 0; i < unique_molecules.size(); i++) {
    const auto &mol = unique_molecules[i];
    const auto &wfn = wavefunctions[i];

    // Get point charges at all neighbor positions for this molecule
    auto external_info = create_external_charges_with_neighbors(
        crystal, asymmetric_charges, mol, config.pair_energy_radius);

    if (!external_info.all_charges.empty()) {
      // Create basis for computing V_ext matrix
      auto basis = qm::AOBasis::load(mol.atoms(), config.basis_name);
      basis.set_pure(config.basis_spherical);
      qm::HartreeFock hf(basis);

      // Compute per-neighbor V_ext contributions (using linearity of expectation value)
      external_potential_per_neighbor[i].resize(external_info.per_neighbor_info.size());
      double V_ext_total = 0.0;

      for (size_t neighbor_idx = 0; neighbor_idx < external_info.per_neighbor_info.size(); neighbor_idx++) {
        const auto &neighbor_info = external_info.per_neighbor_info[neighbor_idx];

        // Build point charge list for this specific neighbor
        std::vector<occ::core::PointCharge> neighbor_charges;
        for (size_t j = 0; j < neighbor_info.charges.size(); j++) {
          neighbor_charges.emplace_back(neighbor_info.charges(j), neighbor_info.positions.col(j));
        }

        // Compute V_ext for this neighbor
        Mat V_ext_neighbor = hf.compute_point_charge_interaction_matrix(neighbor_charges);
        double V_ext_elec_neighbor = 2 * occ::qm::expectation(wfn.mo.kind, wfn.mo.D, V_ext_neighbor);
        double V_ext_nuc_neighbor = hf.nuclear_point_charge_interaction_energy(neighbor_charges);
        double V_ext_neighbor_total = V_ext_elec_neighbor + V_ext_nuc_neighbor;

        external_potential_per_neighbor[i][neighbor_idx] = V_ext_neighbor_total;
        V_ext_total += V_ext_neighbor_total;
      }

      external_potential_energies[i] = V_ext_total;

      occ::log::info("  Molecule {}: V_ext(total) = {:.8f} Ha ({:.2f} kJ/mol) from {} neighbors",
                     i, V_ext_total, V_ext_total * 2625.5, external_info.per_neighbor_info.size());

      // Write point charges to file for debugging
      std::string pc_filename = fmt::format("point_charges_mol_{}.xyz", i);
      std::ofstream pc_file(pc_filename);
      if (pc_file.is_open()) {
        pc_file << external_info.all_charges.size() << "\n";
        pc_file << fmt::format("Point charges for molecule {} (total charge: {:.4f})\n",
                               i, std::accumulate(external_info.all_charges.begin(),
                                                 external_info.all_charges.end(), 0.0,
                                                 [](double sum, const auto &pc) { return sum + pc.charge(); }));
        for (const auto &pc : external_info.all_charges) {
          auto pos = pc.position();
          // Format: x y z charge
          pc_file << fmt::format("{:.6f} {:.6f} {:.6f} {:.6f}\n",
                                pos(0) / occ::units::ANGSTROM_TO_BOHR,
                                pos(1) / occ::units::ANGSTROM_TO_BOHR,
                                pos(2) / occ::units::ANGSTROM_TO_BOHR,
                                pc.charge());
        }
        pc_file.close();
        occ::log::info("  Wrote {} point charges to {}", external_info.all_charges.size(), pc_filename);
      }
    }
  }

  for (size_t i = 0; i < unique_molecules.size(); i++) {
    const auto &mol = unique_molecules[i];
    auto &wfn_a = wavefunctions[i];

    occ::log::info("Processing molecule {} pair energies", i);

    // Get neighbors for this molecule
    if (i >= crystal_dimers.molecule_neighbors.size()) {
      occ::log::warn("No neighbors found for molecule {}", i);
      continue;
    }

    const auto &neighbors = crystal_dimers.molecule_neighbors[i];
    occ::log::info("  Found {} neighboring dimers", neighbors.size());

    // Compute XDM for polarizabilities (for coupling)
    occ::xdm::XDM xdm_a(wfn_a.basis, wfn_a.charge());
    xdm_a.energy(wfn_a.mo);
    Vec alpha_a = xdm_a.polarizabilities();
    wfn_a.xdm_polarizabilities = alpha_a;
    wfn_a.have_xdm_parameters = true;

    // Track electric fields for coupling
    std::vector<Mat3N> E_fields;
    Mat3N mol_a_pos = mol.positions();

    // Compute pair energies
    double total_ce_embedded = 0.0;
    double total_ce_gas = 0.0;
    double total_wolf_classical = 0.0;

    // Track CE components
    interaction::CEEnergyComponents sum_ce_embedded;
    interaction::CEEnergyComponents sum_ce_gas;

    // Track polarization field corrections
    double total_polarization_correction = 0.0;
    std::vector<double> polarization_corrections_per_pair;

    // Setup progress tracking
    size_t total_neighbors = neighbors.size();
    occ::core::ProgressTracker progress(total_neighbors);

    // Detailed energy accounting for first pair
    bool first_pair = true;

    size_t pair_count = 0;
    for (const auto &[dimer, unique_idx] : neighbors) {
      const auto &mol_b = dimer.b();
      int mol_b_idx = mol_b.asymmetric_molecule_idx();

      // Get wavefunction for molecule B
      auto &wfn_b = wavefunctions[mol_b_idx];

      // Ensure wfn_b has XDM parameters
      if (!wfn_b.have_xdm_parameters) {
        occ::xdm::XDM xdm_b(wfn_b.basis, wfn_b.charge());
        xdm_b.energy(wfn_b.mo);
        wfn_b.xdm_polarizabilities = xdm_b.polarizabilities();
        wfn_b.have_xdm_parameters = true;
      }

      // Compute CE-1p energy with EMBEDDED wavefunctions
      auto ce_embedded = ce_model_embedded.compute_energy(dimer);

      // Compute CE-1p energy with GAS-PHASE wavefunctions
      auto ce_gas = ce_model_gas.compute_energy(dimer);

      // Compute Wolf classical Coulomb for this dimer
      double wolf_classical = interaction::coulomb_interaction_energy_asym_charges(
          dimer, asymmetric_charges);

      // Compute polarization field correction for this pair
      // Get charges and positions for molecule A
      Vec charges_a(mol.size());
      Mat3N pos_a(3, mol.size());
      auto asym_indices_a = mol.asymmetric_unit_idx();
      auto mol_a_atoms = mol.atoms();
      for (size_t j = 0; j < mol.size(); j++) {
        charges_a(j) = asymmetric_charges(asym_indices_a(j));
        const auto &atom = mol_a_atoms[j];
        pos_a.col(j) = Vec3(atom.x, atom.y, atom.z);
      }

      // Get charges and positions for molecule B
      Vec charges_b(mol_b.size());
      Mat3N pos_b(3, mol_b.size());
      auto asym_indices_b = mol_b.asymmetric_unit_idx();
      auto mol_b_atoms = mol_b.atoms();
      for (size_t j = 0; j < mol_b.size(); j++) {
        charges_b(j) = asymmetric_charges(asym_indices_b(j));
        const auto &atom = mol_b_atoms[j];
        pos_b.col(j) = Vec3(atom.x, atom.y, atom.z);
      }

      // Compute polarization correction
      auto pol_correction = compute_polarization_field_correction(
          wfn_a, wfn_b, mol, mol_b, crystal,
          charges_a, pos_a, charges_b, pos_b,
          alpha_a, wfn_b.xdm_polarizabilities,
          config.wolf_alpha, config.wolf_cutoff);

      total_polarization_correction += pol_correction.total_correction;
      polarization_corrections_per_pair.push_back(pol_correction.total_correction);

      // Detailed accounting for first pair
      if (first_pair) {
        occ::log::info("");
        occ::log::info("  Detailed energy accounting for first pair [{}, {}]:", i, mol_b_idx);
        occ::log::info("  Dimer: {}", dimer.name());
        occ::log::info("  Distance: {:.3f} Å", dimer.nearest_distance());
        occ::log::info("");
        occ::log::info("  Monomer energies:");
        occ::log::info("    E_A (gas):      {:.8f} Ha", gas_phase_energies[i]);
        occ::log::info("    E_A (embedded): {:.8f} Ha", wfn_a.energy.total);
        occ::log::info("    E_B (gas):      {:.8f} Ha", gas_phase_energies[mol_b_idx]);
        occ::log::info("    E_B (embedded): {:.8f} Ha", wfn_b.energy.total);
        occ::log::info("");
        occ::log::info("  CE interaction energies (E_AB - E_A - E_B):");
        occ::log::info("    CE (gas wfns):      {:.8f} Ha ({:.2f} kJ/mol)",
                       ce_gas.total, ce_gas.total * 2625.5);
        occ::log::info("      Coulomb:          {:.8f} Ha", ce_gas.coulomb);
        occ::log::info("      Exchange:         {:.8f} Ha", ce_gas.exchange);
        occ::log::info("      Repulsion:        {:.8f} Ha", ce_gas.repulsion);
        occ::log::info("      Dispersion:       {:.8f} Ha", ce_gas.dispersion);
        occ::log::info("");
        occ::log::info("    CE (embedded wfns): {:.8f} Ha ({:.2f} kJ/mol)",
                       ce_embedded.total, ce_embedded.total * 2625.5);
        occ::log::info("      Coulomb:          {:.8f} Ha", ce_embedded.coulomb);
        occ::log::info("      Exchange:         {:.8f} Ha", ce_embedded.exchange);
        occ::log::info("      Repulsion:        {:.8f} Ha", ce_embedded.repulsion);
        occ::log::info("      Dispersion:       {:.8f} Ha", ce_embedded.dispersion);
        occ::log::info("");
        occ::log::info("  Wolf classical (point charges): {:.8f} Ha ({:.2f} kJ/mol)",
                       wolf_classical, wolf_classical * 2625.5);
        occ::log::info("");
        occ::log::info("  Polarization field correction:");
        occ::log::info("    ΔE_pol (A<-B):  {:.8f} Ha ({:.2f} kJ/mol)",
                       pol_correction.correction_A, pol_correction.correction_A * 2625.5);
        occ::log::info("    ΔE_pol (B<-A):  {:.8f} Ha ({:.2f} kJ/mol)",
                       pol_correction.correction_B, pol_correction.correction_B * 2625.5);
        occ::log::info("    ΔE_pol (total): {:.8f} Ha ({:.2f} kJ/mol)",
                       pol_correction.total_correction, pol_correction.total_correction * 2625.5);
        occ::log::info("");
        occ::log::info("  Electric field details for molecule A (experiencing B's field):");
        occ::log::info("    (All fields in a.u.)");
        for (size_t atom_idx = 0; atom_idx < pos_a.cols(); atom_idx++) {
          occ::log::info("    Atom {:2d}: |E_pc|={:.6f}, |E_wolf|={:.6f}, |E_wfn|={:.6f}, ΔE_pol={:.8f} Ha ({:.4f} kJ/mol)",
                         atom_idx,
                         pol_correction.field_mag_pc_A(atom_idx),
                         pol_correction.field_mag_wolf_A(atom_idx),
                         pol_correction.field_mag_wfn_A(atom_idx),
                         pol_correction.per_atom_correction_A(atom_idx),
                         pol_correction.per_atom_correction_A(atom_idx) * 2625.5);
        }
        occ::log::info("");
        occ::log::info("  Electric field details for molecule B (experiencing A's field):");
        occ::log::info("    (All fields in a.u.)");
        for (size_t atom_idx = 0; atom_idx < pos_b.cols(); atom_idx++) {
          occ::log::info("    Atom {:2d}: |E_pc|={:.6f}, |E_wolf|={:.6f}, |E_wfn|={:.6f}, ΔE_pol={:.8f} Ha ({:.4f} kJ/mol)",
                         atom_idx,
                         pol_correction.field_mag_pc_B(atom_idx),
                         pol_correction.field_mag_wolf_B(atom_idx),
                         pol_correction.field_mag_wfn_B(atom_idx),
                         pol_correction.per_atom_correction_B(atom_idx),
                         pol_correction.per_atom_correction_B(atom_idx) * 2625.5);
        }
        occ::log::info("");
        first_pair = false;
      }

      total_ce_embedded += ce_embedded.total;
      total_ce_gas += ce_gas.total;
      total_wolf_classical += wolf_classical;

      // Accumulate CE components
      sum_ce_embedded += ce_embedded;
      sum_ce_gas += ce_gas;

      occ::log::debug("  Pair [{},{}]: CE_embed={:.6f}, CE_gas={:.6f}, Wolf={:.6f}, Pol={:.6f} Ha",
                      i, mol_b_idx, ce_embedded.total, ce_gas.total, wolf_classical,
                      pol_correction.total_correction);

      // Compute electric field from this neighbor for coupling
      interaction::WolfParameters wolf_params{config.wolf_cutoff, config.wolf_alpha};
      Mat3N E_field = interaction::wolf_electric_field(
          charges_b, pos_b, mol_a_pos, wolf_params);
      E_fields.push_back(E_field);

      // Update progress
      pair_count++;
      progress.update(pair_count, total_neighbors,
                      fmt::format("Mol {} | E[{}|{}]: {}", i,
                                  i, mol_b_idx, dimer.name()));
    }

    // Calculate total embedding energy
    double embed_energy = wfn_a.energy.total - gas_phase_energies[i];

    // Get CE model scale factors
    double polarization_scale = ce_model_embedded.polarization_scale_factor();

    // Compute polarization correction statistics
    double min_pol_correction = *std::min_element(polarization_corrections_per_pair.begin(),
                                                   polarization_corrections_per_pair.end());
    double max_pol_correction = *std::max_element(polarization_corrections_per_pair.begin(),
                                                   polarization_corrections_per_pair.end());
    double mean_pol_correction = total_polarization_correction / polarization_corrections_per_pair.size();

    // Lattice energy calculation (factor of 0.5 for double counting in sums)
    // Include the polarization field correction
    double latt_corrected = embed_energy + 0.5 * total_ce_embedded - external_potential_energies[i];

    // Remove CE polarization term since it's already included in the model
    double ce_pol_correction = 0.5 * polarization_scale * sum_ce_embedded.polarization;

    // Add polarization field correction (factor of 0.5 for double counting)
    double pol_field_correction = 0.5 * total_polarization_correction;

    double latt_final = latt_corrected - ce_pol_correction + pol_field_correction;

    // Summary for this molecule
    occ::log::info("");
    occ::log::info("========================================");
    occ::log::info("Molecule {} Lattice Energy Calculation", i);
    occ::log::info("========================================");
    occ::log::info("");
    occ::log::info("  Configuration:");
    occ::log::info("    Neighbors:           {} (r < {:.2f} Å)", total_neighbors, config.pair_energy_radius);
    occ::log::info("    Wolf cutoff:         {:.2f} Bohr", config.wolf_cutoff);
    occ::log::info("    Wolf alpha:          {:.4f} Bohr⁻¹", config.wolf_alpha);
    occ::log::info("    k_pol:               {:.4f}", polarization_scale);
    occ::log::info("");
    occ::log::info("  CE Energy Components (summed over all pairs):");
    occ::log::info("    {:<20} {:>15} {:>15} {:>15}", "Component", "Gas (kJ/mol)", "Embedded (kJ/mol)", "Δ (kJ/mol)");
    occ::log::info("    {:<20} {:>15} {:>15} {:>15}", std::string(20, '-'), std::string(15, '-'), std::string(15, '-'), std::string(15, '-'));
    occ::log::info("    {:<20} {:>15.2f} {:>15.2f} {:>15.2f}", "Coulomb",
                   sum_ce_gas.coulomb_kjmol(), sum_ce_embedded.coulomb_kjmol(),
                   (sum_ce_embedded.coulomb - sum_ce_gas.coulomb) * 2625.5);
    occ::log::info("    {:<20} {:>15.2f} {:>15.2f} {:>15.2f}", "Exchange",
                   sum_ce_gas.exchange_kjmol(), sum_ce_embedded.exchange_kjmol(),
                   (sum_ce_embedded.exchange - sum_ce_gas.exchange) * 2625.5);
    occ::log::info("    {:<20} {:>15.2f} {:>15.2f} {:>15.2f}", "Repulsion",
                   sum_ce_gas.repulsion_kjmol(), sum_ce_embedded.repulsion_kjmol(),
                   (sum_ce_embedded.repulsion - sum_ce_gas.repulsion) * 2625.5);
    occ::log::info("    {:<20} {:>15.2f} {:>15.2f} {:>15.2f}", "Polarization",
                   sum_ce_gas.polarization_kjmol(), sum_ce_embedded.polarization_kjmol(),
                   (sum_ce_embedded.polarization - sum_ce_gas.polarization) * 2625.5);
    occ::log::info("    {:<20} {:>15.2f} {:>15.2f} {:>15.2f}", "Dispersion",
                   sum_ce_gas.dispersion_kjmol(), sum_ce_embedded.dispersion_kjmol(),
                   (sum_ce_embedded.dispersion - sum_ce_gas.dispersion) * 2625.5);
    occ::log::info("    {:<20} {:>15} {:>15} {:>15}", std::string(20, '-'), std::string(15, '-'), std::string(15, '-'), std::string(15, '-'));
    occ::log::info("    {:<20} {:>15.2f} {:>15.2f} {:>15.2f}", "Total",
                   sum_ce_gas.total_kjmol(), sum_ce_embedded.total_kjmol(),
                   (sum_ce_embedded.total - sum_ce_gas.total) * 2625.5);
    occ::log::info("");
    occ::log::info("  Polarization Field Corrections:");
    occ::log::info("    Total correction:    {:>12.6f} Ha  ({:>10.2f} kJ/mol)",
                   total_polarization_correction, total_polarization_correction * 2625.5);
    occ::log::info("    Per-pair statistics:");
    occ::log::info("      Min:               {:>12.6f} Ha  ({:>10.2f} kJ/mol)",
                   min_pol_correction, min_pol_correction * 2625.5);
    occ::log::info("      Max:               {:>12.6f} Ha  ({:>10.2f} kJ/mol)",
                   max_pol_correction, max_pol_correction * 2625.5);
    occ::log::info("      Mean:              {:>12.6f} Ha  ({:>10.2f} kJ/mol)",
                   mean_pol_correction, mean_pol_correction * 2625.5);
    occ::log::info("    0.5 × Σ ΔE_pol:      {:>12.6f} Ha  ({:>10.2f} kJ/mol)",
                   pol_field_correction, pol_field_correction * 2625.5);
    occ::log::info("");
    occ::log::info("  Input components:");
    occ::log::info("    ΔE_embed             = {:>12.6f} Ha  ({:>10.2f} kJ/mol)",
                   embed_energy, embed_energy * 2625.5);
    occ::log::info("    0.5 × ΣE_CE(emb)     = {:>12.6f} Ha  ({:>10.2f} kJ/mol)",
                   0.5 * total_ce_embedded, 0.5 * total_ce_embedded * 2625.5);
    occ::log::info("    V_ext                = {:>12.6f} Ha  ({:>10.2f} kJ/mol)",
                   external_potential_energies[i], external_potential_energies[i] * 2625.5);
    occ::log::info("    0.5×k_pol×ΣCE_pol    = {:>12.6f} Ha  ({:>10.2f} kJ/mol)",
                   ce_pol_correction, ce_pol_correction * 2625.5);
    occ::log::info("    0.5 × Σ ΔE_pol       = {:>12.6f} Ha  ({:>10.2f} kJ/mol)",
                   pol_field_correction, pol_field_correction * 2625.5);
    occ::log::info("");
    occ::log::info("  Formula: E_latt = ΔE_embed + 0.5×ΣE_CE(emb) - V_ext - 0.5×k_pol×ΣCE_pol(emb) + 0.5×ΣΔE_pol");
    occ::log::info("");
    occ::log::info("  LATTICE ENERGY       = {:>12.6f} Ha  ({:>10.2f} kJ/mol)",
                   latt_final, latt_final * 2625.5);
    occ::log::info("");
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
    occ::log::info("  Molecule {}: {:.8f} Hartree ({:.2f} kJ/mol)", i,
                   energy_diff, energy_diff * 2625.50);
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
  embed->add_option("--charges,--net-charges", config->net_charges,
                    "net charges for each molecule/atom (comma-separated)");
  embed->add_option("--multiplicities", config->multiplicities,
                    "multiplicities for each molecule/atom (comma-separated)");
  embed->add_option("--wolf-alpha", config->wolf_alpha,
                    "Wolf sum damping parameter");
  embed->add_option("--wolf-cutoff", config->wolf_cutoff,
                    "Wolf sum cutoff radius (Angstroms)");
  embed->add_option("--pair-radius", config->pair_energy_radius,
                    "Radius for computing CE pair energies (Angstroms)");
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
    occ::log::info("  Net charges: [{}]",
                   occ::util::join(config.net_charges, ", "));
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
