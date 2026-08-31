#include <filesystem>
#include <fmt/os.h>
#include <occ/cg/distance_partition.h>
#include <occ/cg/solvation_data.h>
#include <occ/driver/crystal_growth.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/interaction/lattice_energy.h>
#include <occ/interaction/xtb_energy_model.h>
#include <occ/xtb/smd_xtb.h>
#include <occ/xtb/xtb_calculator.h>

namespace fs = std::filesystem;
using occ::interaction::CEEnergyModel;
using occ::interaction::LatticeConvergenceSettings;
using occ::interaction::LatticeEnergyCalculator;
using occ::interaction::XTBEnergyModel;

namespace occ::driver {

std::vector<AssignedEnergy> assign_interaction_terms_to_nearest_neighbours(
    const crystal::CrystalDimers::MoleculeNeighbors &neighbors,
    const std::vector<double> &dimer_energies, double cg_radius) {
  double total_taken{0.0};
  std::vector<AssignedEnergy> crystal_contributions(neighbors.size());
  for (size_t k1 = 0; k1 < crystal_contributions.size(); k1++) {
    const auto &[dimerk1, unique_dimer_idx] = neighbors[k1];
    if (dimerk1.nearest_distance() <= cg_radius)
      continue;
    crystal_contributions[k1].is_nn = false;
    auto v = dimerk1.v_ab().normalized();

    // skip if not contributing
    if (dimer_energies[unique_dimer_idx] == 0.0)
      continue;

    total_taken += dimer_energies[unique_dimer_idx];
    double total_dp = 0.0;
    size_t number_interactions = 0;
    for (size_t k2 = 0; k2 < crystal_contributions.size(); k2++) {
      const auto &[dimerk2, unique_index_k2] = neighbors[k2];
      if (dimerk2.nearest_distance() > cg_radius)
        continue;
      if (k1 == k2)
        continue;
      auto v2 = dimerk2.v_ab().normalized();
      double dp = v.dot(v2);
      if (dp <= 0.0)
        continue;
      total_dp += dp;
      number_interactions++;
    }
    for (size_t k2 = 0; k2 < crystal_contributions.size(); k2++) {
      const auto &[dimerk2, unique_index_k2] = neighbors[k2];
      if (dimerk2.nearest_distance() > cg_radius)
        continue;
      if (k1 == k2)
        continue;
      auto v2 = dimerk2.v_ab().normalized();
      double dp = v.dot(v2);
      if (dp <= 0.0)
        continue;
      crystal_contributions[k2].is_nn = true;
      crystal_contributions[k2].energy +=
          (dp / total_dp) * dimer_energies[unique_dimer_idx];
    }
  }
  double total_reassigned{0.0};
  for (size_t k1 = 0; k1 < crystal_contributions.size(); k1++) {
    if (!crystal_contributions[k1].is_nn)
      continue;
    occ::log::debug("{}: {:.3f}", k1, crystal_contributions[k1].energy);
    total_reassigned += crystal_contributions[k1].energy;
  }
  occ::log::debug("Total taken from non-nearest neighbors: {:.3f} kJ/mol",
                  total_taken);
  occ::log::debug("Total assigned to nearest neighbors: {:.3f} kJ/mol",
                  total_reassigned);
  return crystal_contributions;
}

std::vector<occ::Vec3>
calculate_net_dipole(const WavefunctionList &wavefunctions,
                     const crystal::CrystalDimers &crystal_dimers) {
  std::vector<occ::Vec3> dipoles;
  std::vector<occ::Vec> partial_charges;
  for (const auto &wfn : wavefunctions) {
    partial_charges.push_back(wfn.mulliken_charges());
  }
  for (size_t idx = 0; idx < crystal_dimers.molecule_neighbors.size(); idx++) {
    occ::Vec3 dipole = occ::Vec3::Zero(3);
    size_t j = 0;
    for (const auto &[dimer, unique_idx] :
         crystal_dimers.molecule_neighbors[idx]) {
      occ::Vec3 center_a = dimer.a().center_of_mass();
      if (j == 0) {
        const auto &charges =
            partial_charges[dimer.a().asymmetric_molecule_idx()];
        dipole.array() +=
            ((dimer.a().positions().colwise() - center_a).array() *
             charges.array())
                .rowwise()
                .sum();
      }
      const auto &charges =
          partial_charges[dimer.b().asymmetric_molecule_idx()];
      const auto &pos_b = dimer.b().positions();
      dipole.array() += ((pos_b.colwise() - center_a).array() * charges.array())
                            .rowwise()
                            .sum();
      j++;
    }
    dipoles.push_back(dipole / occ::units::BOHR_TO_ANGSTROM);
  }
  return dipoles;
}

inline Wavefunction
load_or_calculate_wavefunction(const Molecule &mol, const std::string &name,
                               const std::string &energy_model) {
  fs::path json_path(fmt::format("{}.owf.json", name));
  if (fs::exists(json_path)) {
    occ::log::info("Loading wavefunction from {}", json_path.string());
    return Wavefunction::load(json_path.string());
  }

  auto parameterized_model =
      occ::interaction::ce_model_from_string(energy_model);

  occ::io::OccInput input;
  input.method.name = parameterized_model.method;
  input.basis.name = parameterized_model.basis;
  input.geometry.set_molecule(mol);
  input.electronic.charge = mol.charge();
  input.electronic.multiplicity = mol.multiplicity();

  auto wfn = occ::driver::single_point(input);

  wfn.save(json_path.string());
  return wfn;
}

inline WavefunctionList
calculate_wavefunctions(const std::string &basename,
                        const std::vector<Molecule> &molecules,
                        const std::string &energy_model) {
  WavefunctionList wavefunctions;
  size_t index = 0;
  for (const auto &m : molecules) {
    occ::log::info(
        "Geometry for molecule {} ({})\n{:3s} {:^10s} {:^10s} {:^10s}", index,
        m.name(), "sym", "x", "y", "z");
    for (const auto &atom : m.atoms()) {
      occ::log::info("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                     core::Element(atom.atomic_number).symbol(), atom.x, atom.y,
                     atom.z);
    }
    std::string name = fmt::format("{}_{}", basename, index);
    wavefunctions.emplace_back(
        load_or_calculate_wavefunction(m, name, energy_model));
    index++;
  }
  return wavefunctions;
}

inline void compute_monomer_energies(const std::string &basename,
                                     WavefunctionList &wavefunctions) {
  size_t idx = 0;

  for (auto &wfn : wavefunctions) {
    fs::path monomer_energies_path(
        fmt::format("{}_{}_monomer_energies.json", basename, idx));
    if (fs::exists(monomer_energies_path)) {
      occ::log::debug("Loading monomer energies from {}",
                      monomer_energies_path.string());
      std::ifstream ifs(monomer_energies_path.string());
      wfn.energy = nlohmann::json::parse(ifs).get<occ::qm::Energy>();
    } else {
      occ::log::flush();
      qm::HartreeFock hf(wfn.basis);
      occ::interaction::CEMonomerCalculationParameters params;
      params.Schwarz = hf.compute_schwarz_ints();
      occ::interaction::compute_ce_model_energies(wfn, hf, params);
      occ::log::debug("Writing monomer energies to {}",
                      monomer_energies_path.string());
      std::ofstream ofs(monomer_energies_path.string());
      nlohmann::json j = wfn.energy;
      ofs << j;
    }
    idx++;
  }
}

inline void write_energy_summary(double total,
                                 const occ::core::Molecule &molecule,
                                 double solvation_free_energy,
                                 double total_interaction_energy) {
  double Gr = molecule.rotational_free_energy(298);
  occ::core::MolecularPointGroup pg(molecule);
  occ::log::debug("Molecule point group = {}, symmetry number = {}",
                  pg.point_group_string(), pg.symmetry_number());
  double Gt = molecule.translational_free_energy(298);
  double molar_mass = molecule.molar_mass();

  constexpr double R = 8.31446261815324;
  constexpr double RT = 298 * R / 1000;
  occ::log::warn("Free energy estimates at T = 298 K, P = 1 atm., "
                 "units: kJ/mol");
  occ::log::warn("-------------------------------------------------------");
  occ::log::warn("lattice energy (crystal)             {: 9.3f}  (E_lat)",
                 0.5 * total);
  Gr += RT * std::log(pg.symmetry_number());
  occ::log::warn("rotational free energy (molecule)    {: 9.3f}  (E_rot)", Gr);
  occ::log::warn("translational free energy (molecule) {: 9.3f}  (E_trans)",
                 Gt);
  // includes concentration shift
  double dG_solv = solvation_free_energy + 1.89 / occ::units::KJ_TO_KCAL;
  occ::log::warn("solvation free energy (molecule)     {: 9.3f}  (E_solv)",
                 dG_solv);
  double dH_sub = -0.5 * total - 2 * RT;
  occ::log::warn("dH sublimation                       {: 9.3f}", dH_sub);
  double dS_sub = Gr + Gt;
  occ::log::warn("dS sublimation                       {: 9.3f}", dS_sub);
  double dG_sub = dH_sub + dS_sub;
  occ::log::warn("dG sublimation                       {: 9.3f}", dG_sub);
  double dG_solubility = dG_solv + dG_sub;
  occ::log::warn("dG solution                          {: 9.3f}",
                 dG_solubility);
  double equilibrium_constant = std::exp(-dG_solubility / RT);
  occ::log::warn("equilibrium_constant                 {: 9.2e}",
                 equilibrium_constant);
  occ::log::warn("log S                                {: 9.3f}",
                 std::log10(equilibrium_constant));
  occ::log::warn("solubility (g/L)                     {: 9.2e}",
                 equilibrium_constant * molar_mass * 1000);
  occ::log::warn("Total E_int                          {: 9.3f}",
                 total_interaction_energy);
}

inline void write_xyz_neighbors(
    const std::string &filename,
    const crystal::CrystalDimers::MoleculeNeighbors &neighbors) {
  auto neigh = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                              fmt::file::CREATE);

  size_t natom = std::accumulate(
      neighbors.begin(), neighbors.end(), 0,
      [](size_t a, const auto &d) { return a + d.dimer.b().size(); });

  neigh.print("{}\nel x y z idx\n", natom);

  size_t j = 0;
  for (const auto &[dimer, unique_idx] : neighbors) {
    auto pos = dimer.b().positions();
    auto els = dimer.b().elements();
    for (size_t a = 0; a < dimer.b().size(); a++) {
      neigh.print("{:.3s} {:12.5f} {:12.5f} {:12.5f} {:5d}\n", els[a].symbol(),
                  pos(0, a), pos(1, a), pos(2, a), j);
    }
    j++;
  }
}

namespace {

/// Inputs to the solvated neighbour loop that do not depend on how the dimer
/// energies were produced.
struct SolvatedNeighborInputs {
  const crystal::Crystal &crystal;
  const crystal::CrystalDimers::MoleculeNeighbors &full_neighbors;
  const crystal::CrystalDimers::MoleculeNeighbors &nearest_neighbors;
  const cg::SolvationData &solvation;
  std::string molname;
  size_t num_unique_dimers{0};
  bool antisymmetrize{true};
  bool write_surface_files{true};
  double inner_radius{3.8};
  double solution_term{0.0}; ///< kJ/mol
  bool print_descriptors{false};
};

/// One dimer's crystal-side energy, supplied by the calculator.
struct DimerCrystalEnergy {
  bool computed{true};
  double total{0.0};                  ///< kJ/mol
  cg::CGEnergyComponents breakdown{}; ///< model-specific extra components
};

/// Partition the solvation surface over the neighbours and assemble the
/// per-dimer results.
///
/// Everything here is independent of the energy model: `crystal_energy(idx)`
/// supplies the crystal-side energy for unique dimer `idx`, and the rest —
/// partitioning, the solvation term, nearest-neighbour assignment, the result
/// records and the totals — is shared. Adding a solvation channel or an energy
/// model touches this once rather than once per calculator.
template <typename EnergyFn>
cg::MoleculeResult assemble_solvated_neighbors(
    const SolvatedNeighborInputs &in, EnergyFn &&crystal_energy,
    cg::DimerResults &interactions, cg::DimerResults &interactions_crystal,
    std::vector<cg::SolvationContribution> &breakdown_out) {

  cg::SolventSurfacePartitioner partitioner(in.crystal, in.full_neighbors);
  partitioner.set_should_antisymmetrize(in.antisymmetrize);
  partitioner.set_basename(in.molname);
  partitioner.set_use_normalized_distance(false);
  partitioner.set_should_write_surface_files(in.write_surface_files);
  auto breakdown = partitioner.partition(in.nearest_neighbors, in.solvation);

  // Indexed by unique dimer index, so an uncomputed dimer leaves a zero in
  // place rather than shifting everything after it.
  std::vector<double> dimer_energy_vals(in.num_unique_dimers, 0.0);
  for (size_t k = 0; k < in.num_unique_dimers; k++) {
    const auto energy = crystal_energy(k);
    dimer_energy_vals[k] = energy.computed ? energy.total : 0.0;
  }

  auto crystal_contributions = assign_interaction_terms_to_nearest_neighbours(
      in.full_neighbors, dimer_energy_vals, in.inner_radius);
  interactions.reserve(in.full_neighbors.size());

  occ::log::warn("Neighbors for asymmetric molecule {}", in.molname);
  occ::log::warn("nn {:>3s} {:>5s} {:>5s} {:<28s} "
                 "{:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}",
                 "id", "Rn", "Rc", "Label", "E_crys", "ES_AB", "ES_BA", "E_S",
                 "E_nn", "E_int");
  occ::log::warn(std::string(95, '='));

  static constexpr const char *row_fmt =
      " {} {:>3d} {:>5.2f} {:>5.2f} {:<28s} {: 7.2f} "
      "{: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f}";

  cg::MoleculeResult results;
  results.total.solution_term = in.solution_term;

  struct DescriptorRow {
    int unique_idx;
    std::string label;
    cg::CGEnergyComponents values;
  };
  std::vector<DescriptorRow> descriptor_rows;

  size_t j = 0;
  for (const auto &[dimer, unique_idx] : in.full_neighbors) {
    const auto energy = crystal_energy(unique_idx);
    if (!energy.computed) {
      interactions.push_back(cg::DimerResult{dimer, false, unique_idx});
      interactions_crystal.push_back(cg::DimerResult{dimer, false, unique_idx});
      j++;
      continue;
    }

    const auto &contribution = breakdown[j];
    cg::DimerSolventTerm solvent_term;
    solvent_term.ab =
        contribution.forward_energy() * occ::units::AU_TO_KJ_PER_MOL;
    solvent_term.ba =
        contribution.reverse_energy() * occ::units::AU_TO_KJ_PER_MOL;
    solvent_term.total =
        contribution.total_energy() * occ::units::AU_TO_KJ_PER_MOL;

    const double e_nn = crystal_contributions[j].energy;
    const bool is_nearest_neighbor = crystal_contributions[j].is_nn;
    results.total.crystal_energy += energy.total;

    double interaction_energy = solvent_term.total - energy.total - e_nn;

    if (is_nearest_neighbor) {
      results.total.interaction_energy += interaction_energy;

      auto components = energy.breakdown;
      components[cg::components::crystal_nn] = e_nn;
      components[cg::components::crystal_total] = energy.total;
      components[cg::components::solvation_ab] = solvent_term.ab;
      components[cg::components::solvation_ba] = solvent_term.ba;
      components[cg::components::solvation_total] = solvent_term.total;
      components[cg::components::total] = interaction_energy;

      // Whatever non-energy channels the model partitioned ride along with
      // the energies, so a new descriptor needs no change here.
      cg::CGEnergyComponents descriptors;
      for (const auto &channel : contribution.descriptor_channels())
        descriptors[channel] = contribution.descriptor(channel).total();
      if (in.print_descriptors)
        descriptor_rows.push_back({unique_idx, dimer.name(), descriptors});

      interactions.push_back(cg::DimerResult{dimer, true, unique_idx,
                                             std::move(components),
                                             std::move(descriptors)});

      auto crystal_components = energy.breakdown;
      crystal_components[cg::components::crystal_nn] = e_nn;
      crystal_components[cg::components::crystal_total] = energy.total;
      crystal_components[cg::components::total] = energy.total + e_nn;
      interactions_crystal.push_back(cg::DimerResult{
          dimer, true, unique_idx, std::move(crystal_components)});
    } else {
      interactions.push_back(cg::DimerResult{dimer, false, unique_idx});
      interactions_crystal.push_back(cg::DimerResult{dimer, false, unique_idx});
      interaction_energy = 0;
    }

    const double rn = dimer.nearest_distance();
    const double rc = dimer.centroid_distance();
    if (is_nearest_neighbor) {
      occ::log::warn(fmt::runtime(row_fmt), "|", unique_idx, rn, rc,
                     dimer.name(), energy.total, solvent_term.ab,
                     solvent_term.ba, solvent_term.total, e_nn,
                     interaction_energy);
    } else {
      occ::log::debug(fmt::runtime(row_fmt), " ", unique_idx, rn, rc,
                      dimer.name(), energy.total, solvent_term.ab,
                      solvent_term.ba, solvent_term.total, e_nn,
                      interaction_energy);
    }

    results.add_dimer_result(interactions.back());
    j++;
  }

  // Per-contact descriptors, one column per channel the model produced.
  if (in.print_descriptors && !descriptor_rows.empty()) {
    std::vector<std::string> channels;
    for (const auto &[name, value] : descriptor_rows.front().values)
      channels.push_back(name);
    std::sort(channels.begin(), channels.end());

    std::string header = fmt::format("{:>4s} {:<28s}", "id", "Label");
    for (const auto &channel : channels)
      header += fmt::format(" {:>16s}", channel);
    occ::log::warn("");
    occ::log::warn("Solvation descriptors per nearest-neighbour contact");
    occ::log::warn("{}", header);
    occ::log::warn(std::string(header.size(), '='));

    for (const auto &row : descriptor_rows) {
      std::string line = fmt::format("{:>4d} {:<28s}", row.unique_idx,
                                     row.label);
      for (const auto &channel : channels) {
        const auto it = row.values.find(channel);
        line += fmt::format(" {:16.4f}",
                            it == row.values.end() ? 0.0 : it->second);
      }
      occ::log::warn("{}", line);
    }
  }

  // Per-contact descriptors are model-specific, so report whatever came
  // through rather than a fixed set.
  if (!results.descriptors.empty()) {
    occ::log::warn("Solvation descriptors summed over nearest neighbours:");
    std::vector<std::string> names;
    for (const auto &[name, value] : results.descriptors)
      names.push_back(name);
    std::sort(names.begin(), names.end());
    for (const auto &name : names)
      occ::log::warn("  {:<24s} {: 12.4f}", name, results.descriptors[name]);
  }

  breakdown_out = std::move(breakdown);
  return results;
}

} // namespace

CrystalGrowthCalculator::CrystalGrowthCalculator(
    const crystal::Crystal &crystal,
    const CrystalGrowthCalculatorOptions &options)
    : m_crystal(crystal), m_molecules(m_crystal.symmetry_unique_molecules()),
      m_options(options), m_interaction_energies(m_molecules.size()),
      m_crystal_interaction_energies(m_molecules.size()) {
  const auto N = m_molecules.size();
  occ::log::info("Found {} symmetry unique molecule{}\n{:<5s} {:<5s} {:>32s}",
                 N, N > 1 ? "s" : "", "index", "label", "formula");
  for (int i = 0; i < N; i++) {
    const auto &mol = m_molecules[i];
    occ::log::info("{:<5d} {:<5s} {:>32s}", i, mol.name(),
                   occ::core::chemical_formula(mol.elements()));
  }
}

void CrystalGrowthCalculator::set_molecule_charges(
    const std::vector<int> &charges) {
  if (charges.size() != m_molecules.size()) {
    throw std::runtime_error(
        fmt::format("Require {} charges to be specified, found {}",
                    m_molecules.size(), charges.size()));
  }
  for (int i = 0; i < charges.size(); i++) {
    m_molecules[i].set_charge(charges[i]);
  }
}

CEModelCrystalGrowthCalculator::CEModelCrystalGrowthCalculator(
    const crystal::Crystal &crystal,
    const CrystalGrowthCalculatorOptions &options)
    : CrystalGrowthCalculator(crystal, options) {}

/*
void CEModelCrystalGrowthCalculator::dipole_correction() {
  auto dipoles = calculate_net_dipole(m_gas_phase_wavefunctions, m_full_dimers);
  double V =
      4.0 * std::numbers::pi_v<double> * m_outer_radius * m_outer_radius * m_outer_radius / 3.0;
  for (int i = 0; i < dipoles.size(); i++) {
    const auto &dipole = dipoles[i];
    occ::log::debug("Net dipole for molecule shell {} = ({:.3f} {:.3f} {:.3f})",
                    i, dipole(0), dipole(1), dipole(2));
    double e = -2 * std::numbers::pi_v<double> * dipole.squaredNorm() / (3 * V) *
               occ::units::AU_TO_KJ_PER_MOL;
    occ::log::debug("Energy = {:.6f} kJ/mol ({:.3f} per molecule)", e,
                    e / (2 * m_full_dimers.molecule_neighbors[i].size()));
  }
}
*/

void CEModelCrystalGrowthCalculator::init_monomer_energies() {
  const auto &opts = options();
  {
    occ::timing::StopWatch sw;
    sw.start();
    m_gas_phase_wavefunctions =
        calculate_wavefunctions(opts.basename, m_molecules, opts.energy_model);
    sw.stop();

    occ::log::info("Gas phase wavefunctions took {:.6f} seconds", sw.read());
  }
  {
    auto parameterized_model =
        occ::interaction::ce_model_from_string(opts.energy_model);
    occ::timing::StopWatch sw;
    sw.start();

    CGSolvationSettings settings;
    settings.method = parameterized_model.method;
    settings.basis = parameterized_model.basis;
    // Dissolving a crystal, the cell gives the condensed-phase volume per
    // molecule directly; openCOSMO-RS uses it for its reference-state term.
    if (const auto n = m_crystal.unit_cell_molecules().size(); n > 0)
      settings.volume_per_molecule = m_crystal.volume() / static_cast<double>(n);

    auto spec = SolventSpec::parse(opts.solvent);
    spec.temperature = opts.temperature;

    auto model = make_cg_solvation_model(opts.solvation_model, settings);
    model->validate(spec);
    if (opts.wavefunction_choice == WavefunctionChoice::Solvated &&
        !model->supports_solvated_wavefunctions()) {
      throw std::runtime_error(fmt::format(
          "{} cannot supply solvent-polarised wavefunctions (its reference is "
          "the ideal conductor); use --wavefunction-choice gas",
          model->name()));
    }
    occ::log::info("Solvation model: {} in '{}'", model->name(),
                   spec.to_string());

    auto result = model->compute(opts.basename, m_molecules,
                                 m_gas_phase_wavefunctions, spec);
    m_solvated_surface_properties = std::move(result.surfaces);
    m_solvated_wavefunctions = std::move(result.wavefunctions);

    sw.stop();
    occ::log::info("Solution phase wavefunctions took {:.6f} seconds",
                   sw.read());
  }
  occ::timing::StopWatch sw;
  sw.start();
  occ::log::info("Computing monomer energies for gas phase");
  compute_monomer_energies(opts.basename, m_gas_phase_wavefunctions);
  occ::log::info("Computing monomer energies for solution phase");
  compute_monomer_energies(fmt::format("{}_{}", opts.basename, opts.solvent_tag),
                           m_solvated_wavefunctions);
  sw.stop();
  occ::log::info("Computing monomer energies took {:.6f} seconds", sw.read());
}

void CEModelCrystalGrowthCalculator::converge_lattice_energy() {
  const std::string wfn_choice = "gas";
  const auto &opts = options();
  occ::log::info("Computing crystal interactions using {} wavefunctions",
                 wfn_choice);

  LatticeConvergenceSettings convergence_settings;
  convergence_settings.model_name = opts.energy_model;
  convergence_settings.max_radius = opts.outer_radius;
  convergence_settings.wolf_sum = opts.use_wolf_sum;
  convergence_settings.crystal_field_polarization =
      opts.use_crystal_polarization;

  auto energy_model = std::make_unique<CEEnergyModel>(
      m_crystal, inner_wavefunctions(), outer_wavefunctions());
  energy_model->set_model_name(opts.energy_model);

  LatticeEnergyCalculator calculator(std::move(energy_model), m_crystal,
                                     opts.basename, convergence_settings);

  auto result = calculator.compute();

  m_full_dimers = result.dimers;
  m_dimer_energies = result.energy_components;

  m_nearest_dimers = m_crystal.symmetry_unique_dimers(opts.inner_radius);

  if (m_full_dimers.unique_dimers.size() < 1) {
    occ::log::error("No dimers found using neighbour radius {:.3f}",
                    opts.outer_radius);
    exit(0);
  }
}

cg::MoleculeResult
CEModelCrystalGrowthCalculator::process_neighbors_for_symmetry_unique_molecule(
    int i, const std::string &molname) {
  const auto &opts = options();

  SolvatedNeighborInputs in{crystal(),
                            m_full_dimers.molecule_neighbors[i],
                            m_nearest_dimers.molecule_neighbors[i],
                            m_solvated_surface_properties[i],
                            molname,
                            m_dimer_energies.size(),
                            opts.use_asymmetric_partition,
                            opts.write_debug_output_files,
                            opts.inner_radius,
                            m_solvated_surface_properties[i]
                                    .total_solvation_energy *
                                occ::units::AU_TO_KJ_PER_MOL,
                            opts.print_solvation_descriptors};

  auto crystal_energy = [this](size_t idx) {
    const auto &e = m_dimer_energies[idx];
    DimerCrystalEnergy out;
    out.computed = e.is_computed;
    if (!out.computed)
      return out;
    out.total = e.total_kjmol();
    out.breakdown = {
        {cg::components::coulomb, e.coulomb_kjmol()},
        {cg::components::polarization, e.polarization_kjmol()},
        {cg::components::repulsion, e.repulsion_kjmol()},
        {cg::components::exchange, e.exchange_kjmol()},
        {cg::components::dispersion, e.dispersion_kjmol()},
    };
    return out;
  };

  std::vector<cg::SolvationContribution> breakdown;
  auto results = assemble_solvated_neighbors(
      in, crystal_energy, m_interaction_energies[i],
      m_crystal_interaction_energies[i], breakdown);
  m_solvation_breakdowns.push_back(std::move(breakdown));
  return results;
}

cg::CrystalGrowthResult
CEModelCrystalGrowthCalculator::evaluate_molecular_surroundings() {
  const auto &opts = options();
  cg::CrystalGrowthResult result;

  m_solution_terms = std::vector<double>(m_molecules.size(), 0.0);
  for (size_t i = 0; i < m_molecules.size(); i++) {
    auto mol_dimer_results = process_neighbors_for_symmetry_unique_molecule(
        i, fmt::format("{}_{}_{}", opts.basename, i, opts.solvent_tag));

    result.molecule_results.push_back(mol_dimer_results);

    m_solution_terms[i] = mol_dimer_results.total.solution_term;
    m_lattice_energies.push_back(mol_dimer_results.total.crystal_energy);
    write_energy_summary(mol_dimer_results.total.crystal_energy, m_molecules[i],
                         mol_dimer_results.total.solution_term,
                         mol_dimer_results.total.interaction_energy);

    if (opts.write_debug_output_files) {
      // write neighbors file for molecule i
      std::string neighbors_filename =
          fmt::format("{}_{}_neighbors.xyz", opts.basename, i);
      write_xyz_neighbors(neighbors_filename,
                          m_full_dimers.molecule_neighbors[i]);
    }
  }
  return result;
}

XTBCrystalGrowthCalculator::XTBCrystalGrowthCalculator(
    const crystal::Crystal &crystal,
    const CrystalGrowthCalculatorOptions &options)
    : CrystalGrowthCalculator(crystal, options) {

  occ::log::info("XTB solvation model: {}", options.xtb_solvation_model);
}

void XTBCrystalGrowthCalculator::converge_lattice_energy() {
  occ::log::info("Computing crystal interactions using xtb");

  const auto &opts = options();
  occ::interaction::LatticeConvergenceSettings convergence_settings;
  convergence_settings.wolf_sum = opts.use_wolf_sum;
  convergence_settings.max_radius = opts.outer_radius;

  m_full_dimers = m_crystal.symmetry_unique_dimers(opts.outer_radius);
  std::vector<interaction::CEEnergyComponents> energies;

  LatticeEnergyCalculator calculator(
      std::make_unique<XTBEnergyModel>(m_crystal), m_crystal, opts.basename,
      convergence_settings);

  auto result = calculator.compute();

  m_full_dimers = result.dimers;

  for (const auto &e : result.energy_components) {
    m_dimer_energies.push_back(e.total_kjmol());
  }

  m_nearest_dimers = m_crystal.symmetry_unique_dimers(opts.inner_radius);

  if (m_full_dimers.unique_dimers.size() < 1) {
    occ::log::error("No dimers found using neighbour radius {:.3f}",
                    opts.outer_radius);
    exit(0);
  }
}

occ::cg::CrystalGrowthResult
XTBCrystalGrowthCalculator::evaluate_molecular_surroundings() {
  const auto &opts = options();
  occ::cg::CrystalGrowthResult result;

  m_solution_terms = std::vector<double>(m_molecules.size(), 0.0);
  for (size_t i = 0; i < m_molecules.size(); i++) {
    auto mol_dimer_results = process_neighbors_for_symmetry_unique_molecule(
        i, fmt::format("{}_{}_{}", opts.basename, i, opts.solvent_tag));

    result.molecule_results.push_back(mol_dimer_results);

    m_solution_terms[i] = mol_dimer_results.total.solution_term;
    m_lattice_energies.push_back(mol_dimer_results.total.crystal_energy);
    occ::driver::write_energy_summary(
        mol_dimer_results.total.crystal_energy, m_molecules[i],
        mol_dimer_results.total.solution_term,
        mol_dimer_results.total.interaction_energy);
  }
  return result;
}

void XTBCrystalGrowthCalculator::init_monomer_energies() {
  occ::timing::StopWatch sw_gas;
  occ::timing::StopWatch sw_solv;
  const auto &opts = options();

  m_solvated_surface_properties.clear();
  m_solvated_surface_properties.reserve(m_molecules.size());

  size_t index = 0;
  for (const auto &m : m_molecules) {
    occ::log::info("Molecule ({})\n{:3s} {:^10s} {:^10s} {:^10s}", index, "sym",
                   "x", "y", "z");
    for (const auto &atom : m.atoms()) {
      occ::log::info("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                     core::Element(atom.atomic_number).symbol(), atom.x, atom.y,
                     atom.z);
    }

    double e_gas, e_solv;

    // Gas phase via the in-tree GFN2 backend.
    {
      occ::xtb::XtbCalculator xtb(m);
      sw_gas.start();
      e_gas = xtb.single_point_energy();
      sw_gas.stop();
      m_gas_phase_energies.push_back(e_gas);
    }

    // Solvated monomer: in-tree GFN2 + SMD model, harvesting the per-element
    // surface so the partitioner can attribute solvation energy to neighbours
    // exactly the way the CE/QM path does. We deliberately don't compute
    // solvated *dimers* — the per-monomer surfaces partitioned over the
    // crystal neighbour list replace that.
    {
      occ::xtb::XtbCalculator xtb(m);
      auto smd =
          std::make_shared<occ::xtb::SmdSolvationModel>(opts.solvent);
      xtb.set_solvation_model(smd);
      occ::log::info("Solvation: {} (in-tree SmdSolvationModel)", opts.solvent);
      sw_solv.start();
      e_solv = xtb.single_point_energy();
      sw_solv.stop();
      m_solvated_energies.push_back(e_solv);

      const auto &res = xtb.last_result();
      if (res.solvation_surfaces) {
        m_solvated_surface_properties.push_back(
            cg::from_xtb_surfaces(*res.solvation_surfaces));
      } else {
        // Shouldn't happen with a real SMD model; fall back to an empty
        // surface so the partitioner sees length-zero coulomb/cds vectors.
        m_solvated_surface_properties.emplace_back();
      }
    }

    occ::log::info("Solvation free energy: {:12.6f} (E(solv) = "
                   "{:12.6f}, E(gas) = {:12.6f})\n",
                   e_solv - e_gas, e_solv, e_gas);
    index++;
  }
  occ::log::info("Gas phase calculations took {:.6f} seconds", sw_gas.read());
  occ::log::info("Solution phase calculations took {:.6f} seconds",
                 sw_solv.read());
}

cg::MoleculeResult
XTBCrystalGrowthCalculator::process_neighbors_for_symmetry_unique_molecule(
    int i, const std::string &molname) {
  const auto &opts = options();

  SolvatedNeighborInputs in{crystal(),
                            m_full_dimers.molecule_neighbors[i],
                            m_nearest_dimers.molecule_neighbors[i],
                            m_solvated_surface_properties[i],
                            molname,
                            m_dimer_energies.size(),
                            opts.use_asymmetric_partition,
                            opts.write_debug_output_files,
                            opts.inner_radius,
                            (m_solvated_energies[i] - m_gas_phase_energies[i]) *
                                occ::units::AU_TO_KJ_PER_MOL,
                            opts.print_solvation_descriptors};

  auto crystal_energy = [this](size_t idx) {
    DimerCrystalEnergy out;
    out.total = m_dimer_energies[idx];
    return out;
  };

  std::vector<cg::SolvationContribution> breakdown;
  auto results = assemble_solvated_neighbors(
      in, crystal_energy, m_interaction_energies[i],
      m_crystal_interaction_energies[i], breakdown);
  m_solvation_breakdowns.push_back(std::move(breakdown));
  return results;
}

DummyCrystalGrowthCalculator::DummyCrystalGrowthCalculator(
    const crystal::Crystal &crystal,
    const CrystalGrowthCalculatorOptions &options)
    : CrystalGrowthCalculator(crystal, options) {}

void DummyCrystalGrowthCalculator::init_monomer_energies() {
  occ::log::info("Dummy calculator - no monomer energies to initialize");
  // No wavefunctions needed for dummy calculator
}

void DummyCrystalGrowthCalculator::converge_lattice_energy() {
  const auto &opts = options();

  // Generate dimers using the same radius as other calculators
  m_full_dimers = m_crystal.symmetry_unique_dimers(opts.inner_radius);
  m_nearest_dimers = m_crystal.symmetry_unique_dimers(opts.inner_radius);

  if (m_full_dimers.unique_dimers.size() < 1) {
    occ::log::error("No dimers found using neighbour radius {:.3f}",
                    opts.outer_radius);
    exit(0);
  }

  // Create dummy energy components based on 1/r
  m_dimer_energies.clear();
  m_dimer_energies.reserve(m_full_dimers.unique_dimers.size());

  for (const auto &dimer : m_full_dimers.unique_dimers) {
    double r = dimer.nearest_distance();
    double dummy_energy = 0.0;
    if (r > 0 && r < opts.inner_radius) {
      dummy_energy = 10.0 / r;
    }

    // Create dummy energy component
    cg::PairEnergies::value_type dummy_component;
    dummy_component.is_computed = true;
    // Set the total energy - you may need to adjust this based on your
    // PairEnergies structure This assumes there's a way to set the total energy
    // value

    m_dimer_energies.push_back(dummy_component);

    occ::log::debug("Dummy dimer energy for distance {:.3f}: {:.3f}", r,
                    dummy_energy);
  }

  occ::log::info("Generated {} dummy dimer energies", m_dimer_energies.size());
}

cg::MoleculeResult
DummyCrystalGrowthCalculator::process_neighbors_for_symmetry_unique_molecule(
    int i, const std::string &molname) {

  const auto &opts = options();
  const auto &full_neighbors = m_full_dimers.molecule_neighbors[i];
  const auto &nearest_neighbors = m_nearest_dimers.molecule_neighbors[i];
  auto &interactions = m_interaction_energies[i];
  auto &interactions_crystal = m_crystal_interaction_energies[i];

  // Create dummy energy values based on 1/r
  std::vector<double> dimer_energy_vals;
  for (size_t idx = 0; idx < m_full_dimers.unique_dimers.size(); idx++) {
    const auto &dimer = m_full_dimers.unique_dimers[idx];
    double r = dimer.nearest_distance();
    double dummy_energy = 0.0;
    if (r > 0 && r < opts.inner_radius) {
      dummy_energy = 10.0 / r;
    }
    dimer_energy_vals.push_back(dummy_energy);
  }

  auto crystal_contributions = assign_interaction_terms_to_nearest_neighbours(
      full_neighbors, dimer_energy_vals, opts.inner_radius);

  interactions.reserve(full_neighbors.size());

  occ::log::warn("Dummy neighbors for asymmetric molecule {}", molname);
  occ::log::warn("nn {:>3s} {:>5s} {:>5s} {:<28s} {:>7s} {:>7s}", "id", "Rn",
                 "Rc", "Label", "E_dummy", "E_int");
  occ::log::warn(std::string(70, '='));

  cg::MoleculeResult dimer_energy_results;
  auto &total = dimer_energy_results.total;

  // Dummy solution term
  total.solution_term = 0.0; // No solvation for dummy calculator

  size_t j = 0;
  for (const auto &[dimer, unique_idx] : full_neighbors) {
    auto dimer_name = dimer.name();
    double rn = dimer.nearest_distance();
    double rc = dimer.centroid_distance();
    double crystal_contribution = crystal_contributions[j].energy;
    bool is_nearest_neighbor = crystal_contributions[j].is_nn;

    double dummy_energy = dimer_energy_vals[unique_idx];
    double interaction_energy =
        is_nearest_neighbor ? dummy_energy + crystal_contribution : 0.0;

    total.crystal_energy += dummy_energy;

    if (is_nearest_neighbor) {
      total.interaction_energy += interaction_energy;

      interactions.push_back(cg::DimerResult{
          dimer,
          true,
          unique_idx,
          {
              {cg::components::crystal_nn, crystal_contribution},
              {cg::components::crystal_total, dummy_energy},
              {cg::components::total, interaction_energy},
          }});

      interactions_crystal.push_back(cg::DimerResult{
          dimer,
          true,
          unique_idx,
          {
              {cg::components::crystal_total, dummy_energy},
              {cg::components::total, dummy_energy + crystal_contribution},
          }});

      occ::log::warn(" {} {:>3d} {:>5.2f} {:>5.2f} {:<28s} {:>7.2f} {:>7.2f}",
                     '|', unique_idx, rn, rc, dimer_name, dummy_energy,
                     interaction_energy);
    } else {
      interactions.push_back(cg::DimerResult{dimer, false, unique_idx});
      interactions_crystal.push_back(cg::DimerResult{dimer, false, unique_idx});

      occ::log::debug(" {} {:>3d} {:>5.2f} {:>5.2f} {:<28s} {:>7.2f} {:>7.2f}",
                      ' ', unique_idx, rn, rc, dimer_name, dummy_energy, 0.0);
    }

    dimer_energy_results.add_dimer_result(interactions.back());
    j++;
  }

  return dimer_energy_results;
}

cg::CrystalGrowthResult
DummyCrystalGrowthCalculator::evaluate_molecular_surroundings() {
  const auto &opts = options();
  cg::CrystalGrowthResult result;

  m_solution_terms = std::vector<double>(m_molecules.size(), 0.0);

  for (size_t i = 0; i < m_molecules.size(); i++) {
    auto mol_dimer_results = process_neighbors_for_symmetry_unique_molecule(
        i, fmt::format("{}_{}_dummy", opts.basename, i));

    result.molecule_results.push_back(mol_dimer_results);

    m_solution_terms[i] = mol_dimer_results.total.solution_term;
    m_lattice_energies.push_back(mol_dimer_results.total.crystal_energy);

    occ::log::info("Dummy calculation for molecule {} - energy numbers are "
                   "largely meaningless (10.0/nearest_distance in angs)",
                   i);
    occ::log::info("Total interaction energy = {:.3f}",
                   mol_dimer_results.total.interaction_energy);

    if (opts.write_debug_output_files) {
      // write neighbors file for molecule i
      std::string neighbors_filename =
          fmt::format("{}_{}_neighbors_dummy.xyz", opts.basename, i);
      write_xyz_neighbors(neighbors_filename,
                          m_full_dimers.molecule_neighbors[i]);
    }
  }

  return result;
}

} // namespace occ::driver
