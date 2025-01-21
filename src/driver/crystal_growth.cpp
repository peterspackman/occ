#include <filesystem>
#include <fmt/os.h>
#include <occ/driver/crystal_growth.h>
#include <occ/interaction/ce_energy_model.h>
#include <occ/interaction/lattice_energy.h>
#include <occ/interaction/xtb_energy_model.h>
#include <occ/xtb/xtb_wrapper.h>

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
      std::cout << std::flush;
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
      4.0 * M_PI * m_outer_radius * m_outer_radius * m_outer_radius / 3.0;
  for (int i = 0; i < dipoles.size(); i++) {
    const auto &dipole = dipoles[i];
    occ::log::debug("Net dipole for molecule shell {} = ({:.3f} {:.3f} {:.3f})",
                    i, dipole(0), dipole(1), dipole(2));
    double e = -2 * M_PI * dipole.squaredNorm() / (3 * V) *
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

    cg::SMDSettings smd_settings;
    smd_settings.method = parameterized_model.method;
    smd_settings.basis = parameterized_model.basis;

    cg::SMDCalculator smd_calc(opts.basename, m_molecules,
                               m_gas_phase_wavefunctions, opts.solvent,
                               smd_settings);
    auto result = smd_calc.calculate();
    m_solvated_surface_properties = result.surfaces;
    m_solvated_wavefunctions = result.wavefunctions;

    sw.stop();
    occ::log::info("Solution phase wavefunctions took {:.6f} seconds",
                   sw.read());
  }
  occ::timing::StopWatch sw;
  sw.start();
  occ::log::info("Computing monomer energies for gas phase");
  compute_monomer_energies(opts.basename, m_gas_phase_wavefunctions);
  occ::log::info("Computing monomer energies for solution phase");
  compute_monomer_energies(fmt::format("{}_{}", opts.basename, opts.solvent),
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

  const auto &surface_properties = m_solvated_surface_properties[i];
  const auto &full_neighbors = m_full_dimers.molecule_neighbors[i];
  const auto &nearest_neighbors = m_nearest_dimers.molecule_neighbors[i];
  auto &interactions = m_interaction_energies[i];
  auto &interactions_crystal = m_crystal_interaction_energies[i];

  constexpr bool use_dnorm = false;

  cg::SolventSurfacePartitioner p(crystal(), full_neighbors);
  p.set_should_antisymmetrize(opts.use_asymmetric_partition);
  p.set_basename(molname);
  p.set_use_normalized_distance(use_dnorm);
  auto solvation_breakdown = p.partition(nearest_neighbors, surface_properties);

  std::vector<double> dimer_energy_vals;
  for (const auto &de : m_dimer_energies) {
    if (!de.is_computed)
      dimer_energy_vals.push_back(0.0);
    dimer_energy_vals.push_back(de.total_kjmol());
  }
  auto crystal_contributions = assign_interaction_terms_to_nearest_neighbours(
      full_neighbors, dimer_energy_vals, opts.inner_radius);
  interactions.reserve(full_neighbors.size());

  occ::log::warn("Neighbors for asymmetric molecule {}", molname);

  occ::log::warn("nn {:>3s} {:>5s} {:>5s} {:<28s} "
                 "{:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}",
                 "id", "Rn", "Rc", "Label", "E_crys", "ES_AB", "ES_BA", "E_S",
                 "E_nn", "E_int");

  occ::log::warn(std::string(95, '='));

  size_t j = 0;
  cg::MoleculeResult dimer_energy_results;
  auto &total = dimer_energy_results.total;

  size_t num_neighbors = std::accumulate(
      crystal_contributions.begin(), crystal_contributions.end(), 0,
      [](size_t a, const AssignedEnergy &x) { return x.is_nn ? a + 1 : a; });

  total.solution_term =
      surface_properties.total_solvation_energy * occ::units::AU_TO_KJ_PER_MOL;

  const std::string row_fmt_string =
      " {} {:>3d} {:>5.2f} {:>5.2f} {:<28s} {: 7.2f} "
      "{: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f}";

  for (const auto &[dimer, unique_idx] : full_neighbors) {
    const auto &e = m_dimer_energies[unique_idx];
    if (!e.is_computed) {
      interactions.push_back(cg::DimerResult{dimer, false, unique_idx});
      interactions_crystal.push_back(cg::DimerResult{dimer, false, unique_idx});
      j++;
      continue;
    }
    const auto &solvent_neighbor_contribution = solvation_breakdown[j];
    auto dimer_name = dimer.name();
    double rn = dimer.nearest_distance();
    double rc = dimer.centroid_distance();
    double crystal_contribution = crystal_contributions[j].energy;
    bool is_nearest_neighbor = crystal_contributions[j].is_nn;

    occ::Vec3 v_ab = dimer.v_ab();

    const double e_nn = crystal_contributions[j].energy;
    const double e_crys = e.total_kjmol();
    const double e_coul = e.coulomb_kjmol();
    const double e_rep = e.repulsion_kjmol();
    const double e_exch = e.exchange_kjmol();
    const double e_pol = e.polarization_kjmol();
    const double e_disp = e.dispersion_kjmol();

    total.crystal_energy += e_crys;

    cg::DimerSolventTerm solvent_term;
    solvent_term.ab = (solvent_neighbor_contribution.coulomb().forward +
                       solvent_neighbor_contribution.cds().forward) *
                      occ::units::AU_TO_KJ_PER_MOL;
    solvent_term.ba = (solvent_neighbor_contribution.coulomb().reverse +
                       solvent_neighbor_contribution.cds().reverse) *
                      occ::units::AU_TO_KJ_PER_MOL;

    solvent_term.total = solvent_neighbor_contribution.total_energy() *
                         occ::units::AU_TO_KJ_PER_MOL;

    double interaction_energy = solvent_term.total - e_crys - e_nn;

    if (is_nearest_neighbor) {
      total.interaction_energy += interaction_energy;
      interactions.push_back(cg::DimerResult{
          dimer,
          true,
          unique_idx,
          {
              {cg::components::crystal_nn, e_nn},
              {cg::components::coulomb, e_coul},
              {cg::components::polarization, e_pol},
              {cg::components::repulsion, e_rep},
              {cg::components::exchange, e_exch},
              {cg::components::dispersion, e_disp},
              {cg::components::crystal_total, e_crys},
              {cg::components::solvation_ab, solvent_term.ab},
              {cg::components::solvation_ba, solvent_term.ba},
              {cg::components::solvation_total, solvent_term.total},
              {cg::components::total, interaction_energy},
          }});

      interactions_crystal.push_back(
          cg::DimerResult{dimer,
                          true,
                          unique_idx,
                          {
                              {cg::components::crystal_nn, e_nn},
                              {cg::components::coulomb, e_coul},
                              {cg::components::polarization, e_pol},
                              {cg::components::repulsion, e_rep},
                              {cg::components::exchange, e_exch},
                              {cg::components::dispersion, e_disp},
                              {cg::components::crystal_total, e_crys},
                              {cg::components::total, e_crys + e_nn},
                          }});
    } else {
      interactions.push_back(cg::DimerResult{dimer, false, unique_idx});
      interactions_crystal.push_back(cg::DimerResult{dimer, false, unique_idx});
      interaction_energy = 0;
    }

    if (is_nearest_neighbor) {
      occ::log::warn(fmt::runtime(row_fmt_string), "|", unique_idx, rn, rc,
                     dimer_name, e.total_kjmol(), solvent_term.ab,
                     solvent_term.ba, solvent_term.total, crystal_contribution,
                     interaction_energy);
    } else {
      occ::log::debug(fmt::runtime(row_fmt_string), " ", unique_idx, rn, rc,
                      dimer_name, e.total_kjmol(), solvent_term.ab,
                      solvent_term.ba, solvent_term.total, crystal_contribution,
                      interaction_energy);
    }
    dimer_energy_results.add_dimer_result(interactions.back());
    j++;
  }
  m_solvation_breakdowns.push_back(solvation_breakdown);
  return dimer_energy_results;
}

cg::CrystalGrowthResult
CEModelCrystalGrowthCalculator::evaluate_molecular_surroundings() {
  const auto &opts = options();
  cg::CrystalGrowthResult result;

  m_solution_terms = std::vector<double>(m_molecules.size(), 0.0);
  for (size_t i = 0; i < m_molecules.size(); i++) {
    auto mol_dimer_results = process_neighbors_for_symmetry_unique_molecule(
        i, fmt::format("{}_{}_{}", opts.basename, i, opts.solvent));

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

  // calculate solvated dimers contribution
  size_t unique_idx = 0;
  m_solvated_dimer_energies =
      std::vector<double>(m_full_dimers.unique_dimers.size(), 0.0);
  occ::log::info("Computing solvated dimer energies for nearest neighbors");
  for (const auto &dimer : m_full_dimers.unique_dimers) {
    m_solvated_dimer_energies[unique_idx] = 0.0;
    if (dimer.nearest_distance() <= 3.8) {
      occ::xtb::XTBCalculator xtb(dimer);
      xtb.set_solvent(opts.solvent);
      xtb.set_solvation_model(opts.xtb_solvation_model);
      int a_idx = dimer.a().asymmetric_molecule_idx();
      int b_idx = dimer.b().asymmetric_molecule_idx();
      m_solvated_dimer_energies[unique_idx] = xtb.single_point_energy() -
                                              m_solvated_energies[a_idx] -
                                              m_solvated_energies[b_idx];
    }
    occ::log::debug("Computed solvated dimer energy {} = {}", unique_idx,
                    m_solvated_dimer_energies[unique_idx]);
    unique_idx++;
  }
}

occ::cg::CrystalGrowthResult
XTBCrystalGrowthCalculator::evaluate_molecular_surroundings() {
  const auto &opts = options();
  occ::cg::CrystalGrowthResult result;

  m_solution_terms = std::vector<double>(m_molecules.size(), 0.0);
  for (size_t i = 0; i < m_molecules.size(); i++) {
    auto mol_dimer_results = process_neighbors_for_symmetry_unique_molecule(
        i, fmt::format("{}_{}_{}", opts.basename, i, opts.solvent));

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
    {
      occ::xtb::XTBCalculator xtb(m);
      sw_gas.start();
      e_gas = xtb.single_point_energy();
      sw_gas.stop();
      m_gas_phase_energies.push_back(e_gas);
      m_partial_charges.push_back(xtb.partial_charges());
    }
    {
      occ::xtb::XTBCalculator xtb(m);
      xtb.set_solvent(opts.solvent);
      occ::log::info("Solvation: {} using {}", opts.solvent,
                     opts.xtb_solvation_model);
      xtb.set_solvation_model(opts.xtb_solvation_model);
      sw_solv.start();
      e_solv = xtb.single_point_energy();
      sw_solv.stop();
      m_solvated_energies.push_back(e_solv);
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

  const auto &full_neighbors = m_full_dimers.molecule_neighbors[i];
  const auto &nearest_neighbors = m_nearest_dimers.molecule_neighbors[i];
  auto &interactions = m_interaction_energies[i];
  auto &interactions_crystal = m_crystal_interaction_energies[i];

  auto crystal_contributions =
      occ::driver::assign_interaction_terms_to_nearest_neighbours(
          full_neighbors, m_dimer_energies, opts.inner_radius);
  interactions.reserve(full_neighbors.size());

  size_t num_neighbors = std::accumulate(
      crystal_contributions.begin(), crystal_contributions.end(), 0,
      [](size_t a, const occ::driver::AssignedEnergy &x) {
        return x.is_nn ? a + 1 : a;
      });

  cg::MoleculeResult dimer_energy_results;
  auto &total = dimer_energy_results.total;

  total.solution_term = (m_solvated_energies[i] - m_gas_phase_energies[i]) *
                        occ::units::AU_TO_KJ_PER_MOL;

  double dimers_solv_total = 0.0;
  {
    size_t j = 0;
    for (const auto &[dimer, unique_idx] : full_neighbors) {
      dimers_solv_total += m_solvated_dimer_energies[unique_idx];
      j++;
    }
    dimers_solv_total *= occ::units::AU_TO_KJ_PER_MOL;
  }
  double dimers_solv_scale_factor = total.solution_term * 2 / dimers_solv_total;

  occ::log::debug("Total dimers solvation: {} vs {}", dimers_solv_total,
                  total.solution_term);

  occ::log::warn("Neighbors for asymmetric molecule {}", molname);

  occ::log::warn("nn {:>3s} {:>7s} {:>7s} {:<28s} "
                 "{:>7s} {:>7s} {:>7s} {:>7s}",
                 "id", "Rn", "Rc", "Label", "E_crys", "E_solv", "E_nn",
                 "E_int");
  occ::log::warn(std::string(83, '='));

  size_t j = 0;
  for (const auto &[dimer, unique_idx] : full_neighbors) {
    double e = m_dimer_energies[unique_idx];
    auto dimer_name = dimer.name();
    double rn = dimer.nearest_distance();
    double rc = dimer.centroid_distance();
    double crystal_contribution = crystal_contributions[j].energy;
    bool is_nearest_neighbor = crystal_contributions[j].is_nn;

    occ::Vec3 v_ab = dimer.v_ab();

    total.crystal_energy += e;

    cg::DimerSolventTerm solvent_term;
    solvent_term.total = m_solvated_dimer_energies[unique_idx] *
                         occ::units::AU_TO_KJ_PER_MOL *
                         dimers_solv_scale_factor;

    double interaction_energy =
        solvent_term.total - e - -crystal_contributions[j].energy;

    if (is_nearest_neighbor) {
      total.interaction_energy += interaction_energy;
      interactions.push_back(cg::DimerResult{
          dimer,
          true,
          unique_idx,
          {
              {cg::components::crystal_nn, crystal_contributions[j].energy},
              {cg::components::crystal_total, e},
              {cg::components::solvation_total, solvent_term.total},
              {cg::components::total, interaction_energy},
          }});

      interactions_crystal.push_back(cg::DimerResult{
          dimer,
          true,
          unique_idx,
          {{cg::components::crystal_total, e},
           {cg::components::total, e + crystal_contributions[j].energy}}});
    } else {
      interactions.push_back(cg::DimerResult{dimer, false, unique_idx});
      interactions_crystal.push_back(cg::DimerResult{dimer, false, unique_idx});
      interaction_energy = 0;
    }

    if (is_nearest_neighbor) {
      occ::log::warn(" {} {:>3d} {: 7.2f} {: 7.2f} {:<28s} {: 7.2f} {: 7.2f} "
                     "{: 7.2f} {: 7.2f}",
                     '|', unique_idx, rn, rc, dimer_name, e, solvent_term.total,
                     crystal_contribution, interaction_energy);

    } else {
      occ::log::debug(" {} {:>3d} {: 7.2f} {: 7.2f} {:<28s} {: 7.2f} {: "
                      "7.2f} {: 7.2f} {: 7.2f}",
                      ' ', unique_idx, rn, rc, dimer_name, e,
                      solvent_term.total, crystal_contribution,
                      interaction_energy);
    }

    dimer_energy_results.add_dimer_result(interactions.back());
    j++;
  }
  return dimer_energy_results;
}

} // namespace occ::driver
