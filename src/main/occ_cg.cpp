#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <filesystem>
#include <fmt/os.h>
#include <fstream>
#include <occ/core/kabsch.h>
#include <occ/core/log.h>
#include <occ/core/point_group.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/dft/dft.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/polarization.h>
#include <occ/io/cifparser.h>
#include <occ/io/core_json.h>
#include <occ/io/crystalgrower.h>
#include <occ/io/eigen_json.h>
#include <occ/io/kmcpp.h>
#include <occ/io/occ_input.h>
#include <occ/io/wavefunction_json.h>
#include <occ/io/xyz.h>
#include <occ/main/crystal_surface_energy.h>
#include <occ/main/pair_energy.h>
#include <occ/main/single_point.h>
#include <occ/main/solvation_partition.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/solvation_correction.h>
#include <occ/xtb/xtb_wrapper.h>
#include <scn/scn.h>

namespace fs = std::filesystem;
using occ::core::Dimer;
using occ::core::Element;
using occ::core::Molecule;
using occ::crystal::Crystal;
using occ::crystal::CrystalDimers;
using occ::crystal::SymmetryOperation;
using occ::qm::HartreeFock;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
using occ::scf::SCF;
using occ::units::AU_TO_KJ_PER_MOL;
using occ::units::BOHR_TO_ANGSTROM;
using SolventNeighborContributionList =
    std::vector<occ::main::SolventNeighborContribution>;
using occ::interaction::CEEnergyComponents;
using occ::main::SolvatedSurfaceProperties;

using WavefunctionList = std::vector<Wavefunction>;
using DimerEnergies = std::vector<CEEnergyComponents>;

Crystal read_crystal(const std::string &filename) {
    occ::io::CifParser parser;
    return parser.parse_crystal(filename).value();
}

Wavefunction calculate_wavefunction(const Molecule &mol,
                                    const std::string &name,
                                    const std::string &energy_model) {
    fs::path json_path(fmt::format("{}.owf.json", name));
    if (fs::exists(json_path)) {
        occ::log::info("Loading gas phase wavefunction from {}",
                       json_path.string());
        using occ::io::JsonWavefunctionReader;
        JsonWavefunctionReader json_wfn_reader(json_path.string());
        return json_wfn_reader.wavefunction();
    }

    auto parameterized_model =
        occ::interaction::ce_model_from_string(energy_model);

    occ::io::OccInput input;
    input.method.name = parameterized_model.method;
    input.basis.name = parameterized_model.basis;
    input.geometry.set_molecule(mol);
    input.electronic.charge = mol.charge();
    input.electronic.multiplicity = mol.multiplicity();
    auto wfn = occ::main::single_point_calculation(input);

    occ::io::JsonWavefunctionWriter writer;
    writer.write(wfn, json_path.string());
    return wfn;
}

WavefunctionList calculate_wavefunctions(const std::string &basename,
                                         const std::vector<Molecule> &molecules,
                                         const std::string &energy_model) {
    WavefunctionList wavefunctions;
    size_t index = 0;
    for (const auto &m : molecules) {
        occ::log::info("Molecule ({})\n{:3s} {:^10s} {:^10s} {:^10s}", index,
                       "sym", "x", "y", "z");
        for (const auto &atom : m.atoms()) {
            occ::log::info("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                           Element(atom.atomic_number).symbol(), atom.x, atom.y,
                           atom.z);
        }
        std::string name = fmt::format("{}_{}", basename, index);
        wavefunctions.emplace_back(
            calculate_wavefunction(m, name, energy_model));
        index++;
    }
    return wavefunctions;
}

std::vector<occ::Vec3>
calculate_net_dipole(const WavefunctionList &wavefunctions,
                     const CrystalDimers &crystal_dimers) {
    std::vector<occ::Vec3> dipoles;
    std::vector<occ::Vec> partial_charges;
    for (const auto &wfn : wavefunctions) {
        partial_charges.push_back(wfn.mulliken_charges());
    }
    for (size_t idx = 0; idx < crystal_dimers.molecule_neighbors.size();
         idx++) {
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
            dipole.array() +=
                ((pos_b.colwise() - center_a).array() * charges.array())
                    .rowwise()
                    .sum();
            j++;
        }
        dipoles.push_back(dipole / BOHR_TO_ANGSTROM);
    }
    return dipoles;
}

void compute_monomer_energies(const std::string &basename,
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
            HartreeFock hf(wfn.basis);
            occ::interaction::CEMonomerCalculationParameters params;
            params.precision = 1e-8;
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

void write_xyz_neighbors(const std::string &filename,
                         const CrystalDimers::MoleculeNeighbors &neighbors) {
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
            neigh.print("{:.3s} {:12.5f} {:12.5f} {:12.5f} {:5d}\n",
                        els[a].symbol(), pos(0, a), pos(1, a), pos(2, a), j);
        }
        j++;
    }
}

void write_cg_structure_file(const std::string &filename,
                             const Crystal &crystal,
                             const CrystalDimers &uc_dimers) {
    occ::log::info("Writing crystalgrower structure file to '{}'", filename);
    occ::io::crystalgrower::StructureWriter cg_structure_writer(filename);
    cg_structure_writer.write(crystal, uc_dimers);
}

void write_cg_net_file(const std::string &filename, const Crystal &crystal,
                       const CrystalDimers &uc_dimers) {
    occ::log::info("Writing crystalgrower net file to '{}'", filename);
    occ::io::crystalgrower::NetWriter cg_net_writer(filename);
    cg_net_writer.write(crystal, uc_dimers);
}

void write_kmcpp_input_file(const std::string &filename, const Crystal &crystal,
                            const CrystalDimers &uc_dimers,
                            const std::vector<double> &solution_terms) {
    occ::log::info("Writing kmcpp structure file to '{}'", filename);
    occ::io::kmcpp::InputWriter kmcpp_structure_writer(filename);
    kmcpp_structure_writer.write(crystal, uc_dimers, solution_terms);
}

struct AssignedEnergy {
    bool is_nn{true};
    double energy{0.0};
};

std::vector<AssignedEnergy> assign_interaction_terms_to_nearest_neighbours(
    const CrystalDimers::MoleculeNeighbors &neighbors,
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

void write_energy_summary(double total, const Molecule &molecule,
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
    occ::log::warn("rotational free energy (molecule)    {: 9.3f}  (E_rot)",
                   Gr);
    occ::log::warn("translational free energy (molecule) {: 9.3f}  (E_trans)",
                   Gt);
    // includes concentration shift
    double dG_solv = solvation_free_energy * occ::units::AU_TO_KJ_PER_MOL +
                     1.89 / occ::units::KJ_TO_KCAL;
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

std::vector<double> map_unique_interactions_to_uc_molecules(
    const Crystal &crystal, const CrystalDimers &dimers,
    CrystalDimers &uc_dimers, const std::vector<double> &solution_terms,
    const std::vector<std::vector<double>> &interaction_energies_vec) {

    auto &uc_neighbors = uc_dimers.molecule_neighbors;
    const auto &mol_neighbors = dimers.molecule_neighbors;

    std::vector<double> solution_terms_uc(uc_neighbors.size());
    // map interactions surrounding UC molecules to symmetry unique
    // interactions
    for (size_t i = 0; i < uc_neighbors.size(); i++) {
        const auto &m = crystal.unit_cell_molecules()[i];
        size_t asym_idx = m.asymmetric_molecule_idx();
        solution_terms_uc[i] = solution_terms[asym_idx];

        const auto &m_asym = crystal.symmetry_unique_molecules()[asym_idx];
        auto &unit_cell_neighbors = uc_neighbors[i];

        occ::log::debug("Molecule {} has {} neighbours within {:.3f}", i,
                        unit_cell_neighbors.size(), uc_dimers.radius);
        occ::log::debug("Unit cell index = {}, asymmetric index = {}", i,
                        asym_idx);
        int s_int = m.asymmetric_unit_symop()(0);

        SymmetryOperation symop(s_int);

        const auto &rotation = symop.rotation();
        occ::log::debug("Asymmetric unit symop: {} (has handedness change: {})",
                        symop.to_string(), rotation.determinant() < 0);

        size_t j = 0;
        const auto &asymmetric_neighbors = mol_neighbors[asym_idx];
        const auto &interaction_energies = interaction_energies_vec[asym_idx];
        occ::log::debug(
            "Num asym neighbors = {}, num interaction energies = {}",
            asymmetric_neighbors.size(), interaction_energies.size());

        occ::log::debug("Neighbors for unit cell molecule {} ({})", i,
                        unit_cell_neighbors.size());
        occ::log::debug("{:<7s} {:>7s} {:>10s} {:>7s} {:>7s}", "N", "b", "Tb",
                        "E_int", "R");
        for (auto &[dimer, unique_idx] : unit_cell_neighbors) {

            auto shift_b = dimer.b().cell_shift();
            auto idx_b = dimer.b().unit_cell_molecule_idx();

            size_t idx{0};
            bool match_type{false};
            for (idx = 0; idx < asymmetric_neighbors.size(); idx++) {
                const auto &d_a = asymmetric_neighbors[idx].dimer;
                if (dimer.equivalent(d_a)) {
                    break;
                }
                if (dimer.equivalent_under_rotation(d_a, rotation)) {
                    match_type = true;
                    break;
                }
            }
            if (idx >= asymmetric_neighbors.size()) {
                throw std::runtime_error(
                    fmt::format("No matching interaction found for uc_mol "
                                "= {}, dimer = {}\n",
                                i, j));
            }
            if (idx >= interaction_energies.size()) {
                throw std::runtime_error(
                    "Matching interaction index exceeds number of known "
                    "interactions energies");
            }
            double rn = dimer.nearest_distance();
            double rc = dimer.centroid_distance();

            double e_int = interaction_energies[idx];

            dimer.set_interaction_energy(e_int);
            dimer.set_interaction_id(idx);
            occ::log::debug(
                "{:<7d} {:>7d} {:>10s} {:7.2f} {:7.3f} {}", j, idx_b,
                fmt::format("{},{},{}", shift_b[0], shift_b[1], shift_b[2]),
                e_int, rc, match_type);
            j++;
        }
    }
    return solution_terms_uc;
}

class CEModelCrystalGrowthCalculator {
  public:
    enum class WavefunctionChoice { GasPhase, Solvated };

    CEModelCrystalGrowthCalculator(const Crystal &crystal,
                                   const std::string &solvent)
        : m_crystal(crystal),
          m_molecules(m_crystal.symmetry_unique_molecules()),
          m_solvent(solvent), m_interaction_energies(m_molecules.size()),
          m_crystal_interaction_energies(m_molecules.size()) {
        occ::log::info("found {} symmetry unique molecules:\n{:<10s} {:>32s}",
                       m_molecules.size(), "index", "name");
        for (int i = 0; i < m_molecules.size(); i++) {
            occ::log::info("{:<4d} {:>32s}", i, m_molecules[i].name());
        }
    }

    void set_basename(const std::string &basename) { m_basename = basename; }
    void set_wavefunction_choice(WavefunctionChoice choice) {
        m_wfn_choice = choice;
    }

    void set_output_verbosity(bool output) { m_output = output; }

    void set_energy_model(const std::string &model) { m_model = model; }

    inline auto &gas_phase_wavefunctions() { return m_gas_phase_wavefunctions; }
    inline auto &solvated_wavefunctions() { return m_solvated_wavefunctions; }
    inline auto &inner_wavefunctions() {
        switch (m_wfn_choice) {
        case WavefunctionChoice::Solvated:
            return solvated_wavefunctions();
        default:
            return gas_phase_wavefunctions();
        }
    }

    inline auto &outer_wavefunctions() {
        switch (m_wfn_choice) {
        case WavefunctionChoice::Solvated:
            return solvated_wavefunctions();
        default:
            return gas_phase_wavefunctions();
        }
    }
    inline auto &solvated_surface_properties() {
        return m_solvated_surface_properties;
    }

    inline void dipole_correction() {
        auto dipoles =
            calculate_net_dipole(m_gas_phase_wavefunctions, m_full_dimers);
        double V =
            4.0 * M_PI * m_outer_radius * m_outer_radius * m_outer_radius / 3.0;
        for (int i = 0; i < dipoles.size(); i++) {
            const auto &dipole = dipoles[i];
            fmt::print(
                "Net dipole for molecule shell {} = ({:.3f} {:.3f} {:.3f})\n",
                i, dipole(0), dipole(1), dipole(2));
            double e = -2 * M_PI * dipole.squaredNorm() / (3 * V) *
                       occ::units::AU_TO_KJ_PER_MOL;
            fmt::print("Energy = {:.6f} ({:.3f} per molecule)\n", e,
                       e / (2 * m_full_dimers.molecule_neighbors[i].size()));
        }
    }

    inline auto &crystal() { return m_crystal; }
    inline const auto &name() { return m_basename; }
    inline const auto &solvent() { return m_solvent; }
    inline const auto &molecules() { return m_molecules; }

    inline auto &nearest_dimers() { return m_nearest_dimers; }
    inline auto &full_dimers() { return m_full_dimers; }
    inline auto &dimer_energies() { return m_dimer_energies; }

    inline auto &solution_terms() { return m_solution_terms; }
    inline auto &interaction_energies() { return m_interaction_energies; }
    inline auto &crystal_interaction_energies() {
        return m_crystal_interaction_energies;
    }

    inline void set_use_wolf_sum(bool value) { m_use_wolf_sum = value; }
    inline void set_use_crystal_polarization(bool value) {
        m_use_crystal_polarization = value;
    }

    void init_monomer_energies() {
        occ::timing::StopWatch sw;
        sw.start();
        occ::log::info("Computing monomer energies for gas phase");
        compute_monomer_energies(m_basename, m_gas_phase_wavefunctions);
        occ::log::info("Computing monomer energies for solution phase");
        compute_monomer_energies(fmt::format("{}_{}", m_basename, m_solvent),
                                 m_solvated_wavefunctions);
        sw.stop();
        occ::log::info("Computing monomer energies took {:.6f} seconds",
                       sw.read());
    }

    void init_wavefunctions() {
        {
            occ::timing::StopWatch sw;
            sw.start();
            m_gas_phase_wavefunctions =
                calculate_wavefunctions(m_basename, m_molecules, m_model);
            sw.stop();

            occ::log::info("Gas phase wavefunctions took {:.6f} seconds",
                           sw.read());
        }
        {
            auto parameterized_model =
                occ::interaction::ce_model_from_string(m_model);
            occ::timing::StopWatch sw;
            sw.start();
            std::tie(m_solvated_surface_properties, m_solvated_wavefunctions) =
                occ::main::calculate_solvated_surfaces(
                    m_basename, m_molecules, m_gas_phase_wavefunctions,
                    m_solvent, parameterized_model.method,
                    parameterized_model.basis);
            sw.stop();
            occ::log::info("Solution phase wavefunctions took {:.6f} seconds",
                           sw.read());
        }
        init_monomer_energies();
    }

    void converge_lattice_energy(double inner_radius, double outer_radius) {
        const std::string wfn_choice = "gas";
        occ::log::info("Computing crystal interactions using {} wavefunctions",
                       wfn_choice);

        occ::main::LatticeConvergenceSettings convergence_settings;
        convergence_settings.model_name = m_model;
        outer_radius = std::max(outer_radius, inner_radius);
        convergence_settings.max_radius = outer_radius;
        convergence_settings.wolf_sum = m_use_wolf_sum;
        convergence_settings.crystal_field_polarization =
            m_use_crystal_polarization;
        m_inner_radius = inner_radius;
        m_outer_radius = outer_radius;

        auto result = occ::main::converged_lattice_energies(
            m_crystal, inner_wavefunctions(), outer_wavefunctions(), m_basename,
            convergence_settings);
        m_full_dimers = result.dimers;
        m_dimer_energies = result.energy_components;

        m_nearest_dimers = m_crystal.symmetry_unique_dimers(inner_radius);

        if (m_full_dimers.unique_dimers.size() < 1) {
            occ::log::error("No dimers found using neighbour radius {:.3f}",
                            outer_radius);
            exit(0);
        }
    }

    void set_molecule_charges(const std::vector<int> &charges) {
        if (charges.size() != m_molecules.size()) {
            throw std::runtime_error(
                fmt::format("Require {} charges to be specified, found {}",
                            m_molecules.size(), charges.size()));
        }
        for (int i = 0; i < charges.size(); i++) {
            m_molecules[i].set_charge(charges[i]);
        }
    }

    std::tuple<CEEnergyComponents, double, double>
    process_neighbors_for_symmetry_unique_molecule(int i,
                                                   const std::string &molname) {

        const auto &surface_properties = m_solvated_surface_properties[i];
        const auto &full_neighbors = m_full_dimers.molecule_neighbors[i];
        const auto &nearest_neighbors = m_nearest_dimers.molecule_neighbors[i];
        auto &interactions = m_interaction_energies[i];
        auto &interactions_crystal = m_crystal_interaction_energies[i];

        const occ::main::SolvationPartitionScheme partition_scheme =
            occ::main::SolvationPartitionScheme::NearestAtom;

        auto solvation_breakdown = occ::main::partition_solvent_surface(
            partition_scheme, m_crystal, molname, outer_wavefunctions(),
            surface_properties, full_neighbors, nearest_neighbors, m_solvent);

        std::vector<double> dimer_energy_vals;
        for (const auto &de : m_dimer_energies) {
            if (!de.is_computed)
                dimer_energy_vals.push_back(0.0);
            dimer_energy_vals.push_back(de.total_kjmol());
        }
        auto crystal_contributions =
            assign_interaction_terms_to_nearest_neighbours(
                full_neighbors, dimer_energy_vals, m_inner_radius);
        interactions.reserve(full_neighbors.size());

        occ::log::warn("Neighbors for asymmetric molecule {}", molname);

        occ::log::warn("nn {:>7s} {:>7s} {:>20s} "
                       "{:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}",
                       "Rn", "Rc", "Symop", "E_crys", "ES_AB", "ES_BA", "E_S",
                       "E_nn", "E_int");
        occ::log::warn(std::string(88, '='));

        size_t j = 0;
        CEEnergyComponents total;

        size_t num_neighbors = std::accumulate(
            crystal_contributions.begin(), crystal_contributions.end(), 0,
            [](size_t a, const AssignedEnergy &x) {
                return x.is_nn ? a + 1 : a;
            });

        double solution_term =
            (surface_properties.dg_gas + surface_properties.dg_correction) *
            AU_TO_KJ_PER_MOL;

        double total_interaction_energy{0.0};
        const std::string row_fmt_string =
            " {} {:>7.2f} {:>7.2f} {:>20s} {: 7.2f} "
            "{: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f}";

        for (const auto &[dimer, unique_idx] : full_neighbors) {
            const auto &e = m_dimer_energies[unique_idx];
            if (!e.is_computed) {
                interactions.push_back(0.0);
                interactions_crystal.push_back(0.0);
                j++;
                continue;
            }
            const auto &solvent_neighbor_contribution = solvation_breakdown[j];
            auto symmetry_string = m_crystal.dimer_symmetry_string(dimer);
            double rn = dimer.nearest_distance();
            double rc = dimer.centroid_distance();
            double crystal_contribution = crystal_contributions[j].energy;
            bool is_nearest_neighbor = crystal_contributions[j].is_nn;

            occ::Vec3 v_ab = dimer.v_ab();

            total = total + e;

            double interaction_energy =
                solvent_neighbor_contribution.total_kjmol() - e.total_kjmol() -
                crystal_contributions[j].energy;

            if (is_nearest_neighbor) {
                total_interaction_energy += interaction_energy;
                interactions.push_back(interaction_energy);
                interactions_crystal.push_back(e.total_kjmol() +
                                               crystal_contributions[j].energy);
            } else {
                interactions.push_back(0.0);
                interactions_crystal.push_back(0.0);
                interaction_energy = 0;
            }

            double solvent_contribution_ab =
                (solvent_neighbor_contribution.coulomb.ab +
                 solvent_neighbor_contribution.cds.ab) *
                AU_TO_KJ_PER_MOL;
            double solvent_contribution_ba =
                (solvent_neighbor_contribution.coulomb.ba +
                 solvent_neighbor_contribution.cds.ba) *
                AU_TO_KJ_PER_MOL;

            if (is_nearest_neighbor) {
                occ::log::warn(row_fmt_string, "|", rn, rc, symmetry_string,
                               e.total_kjmol(), solvent_contribution_ab,
                               solvent_contribution_ba,
                               solvent_neighbor_contribution.total_kjmol(),
                               crystal_contribution, interaction_energy);
            } else {
                occ::log::debug(row_fmt_string, " ", rn, rc, symmetry_string,
                                e.total_kjmol(), solvent_contribution_ab,
                                solvent_contribution_ba,
                                solvent_neighbor_contribution.total_kjmol(),
                                crystal_contribution, interaction_energy);
            }
            j++;
        }
        m_solvation_breakdowns.push_back(solvation_breakdown);
        return {total, total_interaction_energy, solution_term};
    }

    void evaluate_molecular_surroundings() {

        m_solution_terms = std::vector<double>(m_molecules.size(), 0.0);
        for (size_t i = 0; i < m_molecules.size(); i++) {
            auto [mol_total, mol_total_interaction_energy, solution_term] =
                process_neighbors_for_symmetry_unique_molecule(
                    i, fmt::format("{}_{}_{}", m_basename, i, m_solvent));

            m_solution_terms[i] = solution_term;
            m_lattice_energies.push_back(mol_total.total_kjmol());
            write_energy_summary(mol_total.total_kjmol(), m_molecules[i],
                                 m_solvated_surface_properties[i].esolv,
                                 mol_total_interaction_energy);

            if (m_output) {
                // write neighbors file for molecule i
                std::string neighbors_filename =
                    fmt::format("{}_{}_neighbors.xyz", m_basename, i);
                write_xyz_neighbors(neighbors_filename,
                                    m_full_dimers.molecule_neighbors[i]);
            }
        }
    }

    const auto &lattice_energies() const { return m_lattice_energies; }

  private:
    bool m_output{true};
    bool m_use_wolf_sum{false};
    bool m_use_crystal_polarization{false};
    WavefunctionChoice m_wfn_choice{WavefunctionChoice::GasPhase};
    Crystal m_crystal;
    std::vector<occ::core::Molecule> m_molecules;
    std::vector<double> m_lattice_energies;
    std::string m_solvent;
    std::string m_model;
    std::string m_basename;
    WavefunctionList m_gas_phase_wavefunctions;
    WavefunctionList m_solvated_wavefunctions;
    std::vector<SolvatedSurfaceProperties> m_solvated_surface_properties;
    CrystalDimers m_full_dimers;
    DimerEnergies m_dimer_energies;
    CrystalDimers m_nearest_dimers;
    std::vector<SolventNeighborContributionList> m_solvation_breakdowns;
    std::vector<std::vector<double>> m_interaction_energies;
    std::vector<std::vector<double>> m_crystal_interaction_energies;
    std::vector<double> m_solution_terms;
    double m_inner_radius{0.0}, m_outer_radius{0.0};
};

class XTBCrystalGrowthCalculator {
  public:
    XTBCrystalGrowthCalculator(const Crystal &crystal,
                               const std::string &solvent)
        : m_crystal(crystal),
          m_molecules(m_crystal.symmetry_unique_molecules()),
          m_solvent(solvent), m_interaction_energies(m_molecules.size()),
          m_crystal_interaction_energies(m_molecules.size()) {
        init_monomer_energies();
    }

    inline auto &crystal() { return m_crystal; }
    inline const auto &name() { return m_basename; }
    inline const auto &solvent() { return m_solvent; }
    inline const auto &molecules() { return m_molecules; }

    inline auto &nearest_dimers() { return m_nearest_dimers; }
    inline auto &full_dimers() { return m_full_dimers; }
    inline auto &dimer_energies() { return m_dimer_energies; }

    inline auto &solution_terms() { return m_solution_terms; }
    inline auto &interaction_energies() { return m_interaction_energies; }
    inline auto &crystal_interaction_energies() {
        return m_crystal_interaction_energies;
    }

    inline void set_use_wolf_sum(bool value) { m_use_wolf_sum = value; }

    void set_basename(const std::string &basename) { m_basename = basename; }

    void set_molecule_charges(const std::vector<int> &charges) {
        if (charges.size() != m_molecules.size()) {
            throw std::runtime_error(
                fmt::format("Require {} charges to be specified, found {}",
                            m_molecules.size(), charges.size()));
        }
        for (int i = 0; i < charges.size(); i++) {
            m_molecules[i].set_charge(charges[i]);
        }
    }

    void converge_lattice_energy(double inner_radius, double outer_radius) {
        occ::log::info("Computing crystal interactions using xtb");

        occ::main::LatticeConvergenceSettings convergence_settings;
        outer_radius = std::max(outer_radius, inner_radius);
        m_inner_radius = inner_radius;
        m_outer_radius = outer_radius;
        convergence_settings.wolf_sum = m_use_wolf_sum;
        convergence_settings.max_radius = outer_radius;

        m_full_dimers = m_crystal.symmetry_unique_dimers(outer_radius);
        std::vector<CEEnergyComponents> energies;
        auto result = occ::main::converged_xtb_lattice_energies(
            m_crystal, m_basename, convergence_settings);
        m_full_dimers = result.dimers;

        for (const auto &e : result.energy_components) {
            m_dimer_energies.push_back(e.total_kjmol());
        }

        m_nearest_dimers = m_crystal.symmetry_unique_dimers(inner_radius);

        if (m_full_dimers.unique_dimers.size() < 1) {
            occ::log::error("No dimers found using neighbour radius {:.3f}",
                            outer_radius);
            exit(0);
        }

        // calculate solvated dimers contribution
        size_t unique_idx = 0;
        m_solvated_dimer_energies =
            std::vector<double>(m_full_dimers.unique_dimers.size(), 0.0);
        for (const auto &dimer : m_full_dimers.unique_dimers) {
            m_solvated_dimer_energies[unique_idx] = 0.0;
            if (dimer.nearest_distance() <= 3.8) {
                occ::xtb::XTBCalculator xtb(dimer);
                xtb.set_solvent(m_solvent);
                int a_idx = dimer.a().asymmetric_molecule_idx();
                int b_idx = dimer.a().asymmetric_molecule_idx();
                m_solvated_dimer_energies[unique_idx] =
                    xtb.single_point_energy() - m_solvated_energies[a_idx] -
                    m_solvated_energies[b_idx];
            }
            occ::log::debug("Computed solvated dimer energy {} = {}",
                            unique_idx, m_solvated_dimer_energies[unique_idx]);
            unique_idx++;
        }
    }

    void evaluate_molecular_surroundings() {

        m_solution_terms = std::vector<double>(m_molecules.size(), 0.0);
        for (size_t i = 0; i < m_molecules.size(); i++) {
            auto [mol_total, mol_total_interaction_energy, solution_term] =
                process_neighbors_for_symmetry_unique_molecule(
                    i, fmt::format("{}_{}_{}", m_basename, i, m_solvent));

            m_solution_terms[i] = solution_term * AU_TO_KJ_PER_MOL;
            m_lattice_energies.push_back(mol_total);
            write_energy_summary(mol_total, m_molecules[i], solution_term,
                                 mol_total_interaction_energy);
        }
    }

  private:
    std::tuple<double, double, double>
    process_neighbors_for_symmetry_unique_molecule(int i,
                                                   const std::string &molname) {

        const auto &full_neighbors = m_full_dimers.molecule_neighbors[i];
        const auto &nearest_neighbors = m_nearest_dimers.molecule_neighbors[i];
        auto &interactions = m_interaction_energies[i];
        auto &interactions_crystal = m_crystal_interaction_energies[i];

        auto crystal_contributions =
            assign_interaction_terms_to_nearest_neighbours(
                full_neighbors, m_dimer_energies, m_inner_radius);
        interactions.reserve(full_neighbors.size());

        size_t num_neighbors = std::accumulate(
            crystal_contributions.begin(), crystal_contributions.end(), 0,
            [](size_t a, const AssignedEnergy &x) {
                return x.is_nn ? a + 1 : a;
            });

        double solution_term =
            (m_solvated_energies[i] - m_gas_phase_energies[i]);

        double total_interaction_energy{0.0};
        const std::string row_fmt_string =
            " {} {:>7.2f} {:>7.2f} {:>20s} {: 7.2f} "
            "{: 7.2f} {: 7.2f} {: 7.2f}";

        double dimers_solv_total = 0.0;
        {
            size_t j = 0;
            for (const auto &[dimer, unique_idx] : full_neighbors) {
                dimers_solv_total += m_solvated_dimer_energies[unique_idx];
                j++;
            }
        }
        double dimers_solv_scale_factor = solution_term / dimers_solv_total;

        occ::log::debug("Total dimers solvation: {} vs {}", dimers_solv_total,
                        solution_term);

        occ::log::warn("Neighbors for asymmetric molecule {}", molname);

        occ::log::warn("nn {:>7s} {:>7s} {:>20s} "
                       "{:>7s} {:>7s} {:>7s} {:>7s}",
                       "Rn", "Rc", "Symop", "E_crys", "E_solv", "E_nn",
                       "E_int");
        occ::log::warn(std::string(88, '='));

        size_t j = 0;
        double total = 0.0;
        for (const auto &[dimer, unique_idx] : full_neighbors) {
            double e = m_dimer_energies[unique_idx];
            auto symmetry_string = m_crystal.dimer_symmetry_string(dimer);
            double rn = dimer.nearest_distance();
            double rc = dimer.centroid_distance();
            double crystal_contribution = crystal_contributions[j].energy;
            bool is_nearest_neighbor = crystal_contributions[j].is_nn;

            occ::Vec3 v_ab = dimer.v_ab();

            total += e;

            double solv_ab_ba = m_solvated_dimer_energies[unique_idx] *
                                AU_TO_KJ_PER_MOL * dimers_solv_scale_factor;

            double interaction_energy =
                solv_ab_ba - e - -crystal_contributions[j].energy;

            if (is_nearest_neighbor) {
                total_interaction_energy += interaction_energy;
                interactions.push_back(interaction_energy);
                interactions_crystal.push_back(e +
                                               crystal_contributions[j].energy);
            } else {
                interactions.push_back(0.0);
                interactions_crystal.push_back(0.0);
                interaction_energy = 0;
            }

            if (is_nearest_neighbor) {
                occ::log::warn(row_fmt_string, "|", rn, rc, symmetry_string, e,
                               solv_ab_ba, crystal_contribution,
                               interaction_energy);
            } else {
                occ::log::debug(row_fmt_string, " ", rn, rc, symmetry_string, e,
                                solv_ab_ba, crystal_contribution,
                                interaction_energy);
            }
            j++;
        }
        return {total, total_interaction_energy, solution_term};
    }

    void init_monomer_energies() {
        size_t index = 0;
        for (const auto &m : m_molecules) {
            occ::log::info("Molecule ({})\n{:3s} {:^10s} {:^10s} {:^10s}",
                           index, "sym", "x", "y", "z");
            for (const auto &atom : m.atoms()) {
                occ::log::info("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                               Element(atom.atomic_number).symbol(), atom.x,
                               atom.y, atom.z);
            }

            occ::xtb::XTBCalculator xtb(m);
            double e_gas = xtb.single_point_energy();
            m_gas_phase_energies.push_back(e_gas);
            m_partial_charges.push_back(xtb.partial_charges());
            xtb.set_solvent(m_solvent);
            double e_solv = xtb.single_point_energy();
            m_solvated_energies.push_back(e_solv);
            occ::log::info("Solvation free energy: {:12.6f}\n", e_solv - e_gas);
            index++;
        }
    }
    const auto &lattice_energies() const { return m_lattice_energies; }

    Crystal m_crystal;
    std::vector<occ::core::Molecule> m_molecules;
    std::vector<double> m_lattice_energies;
    std::string m_solvent;
    std::string m_basename;
    std::vector<double> m_gas_phase_energies;
    std::vector<occ::Vec> m_partial_charges;
    std::vector<double> m_solvated_energies;
    CrystalDimers m_full_dimers;
    std::vector<double> m_dimer_energies;
    std::vector<double> m_solvated_dimer_energies;
    CrystalDimers m_nearest_dimers;
    std::vector<std::vector<double>> m_interaction_energies;
    std::vector<std::vector<double>> m_crystal_interaction_energies;
    std::vector<double> m_solution_terms;
    double m_inner_radius{0.0}, m_outer_radius{0.0};
    bool m_use_wolf_sum{false};
};

void list_available_solvents() {
    fmt::print("{: <32s} {:>10s} {:>10s} {:>10s} {:>10s} {:>10s} "
               "{:>10s} {:>10s}\n",
               "Solvent", "n (293K)", "acidity", "basicity", "gamma",
               "dielectric", "aromatic", "%F,Cl,Br");
    fmt::print("{:-<110s}\n", "");
    for (const auto &solvent : occ::solvent::smd_solvent_parameters) {
        const auto &param = solvent.second;
        fmt::print("{:<32s} {:10.4f} {:10.4f} {:10.4f} {:10.4f} "
                   "{:10.4f} {:10.4f} {:10.4f}\n",
                   solvent.first, param.refractive_index_293K, param.acidity,
                   param.basicity, param.gamma, param.dielectric,
                   param.aromaticity, param.electronegative_halogenicity);
    }
}

int main(int argc, char **argv) {
    CLI::App app(
        "occ-cg - Interactions of molecules with neighbours in a crystal");
    std::string cif_filename{""}, charge_string{""}, verbosity{"warn"},
        solvent{"water"}, wfn_choice{"gas"};

    int threads{1};
    int max_facets{0};
    double radius{3.8}, cg_radius{3.8};
    double bond_tolerance{0.4};
    bool write_dump_files{false}, spherical{false};
    bool write_kmcpp_file{false};
    bool use_xtb{false};
    bool list_solvents{false};
    bool crystal_is_atomic{false};
    std::string model_name{"ce-b3lyp"};

    CLI::Option *input_option =
        app.add_option("input", cif_filename, "input CIF");
    input_option->required();
    app.add_option("-t,--threads", threads, "number of threads");
    app.add_option("-r,--radius", radius,
                   "maximum radius (Angstroms) for neighbours");
    app.add_option("-m,--model", model_name, "energy model");
    app.add_option("-c,--cg-radius", cg_radius,
                   "maximum radius (Angstroms) for nearest neighbours in CG "
                   "file (must be <= radius)");
    app.add_option("-s,--solvent", solvent, "solvent name");
    app.add_option("--charges", charge_string, "system net charge");
    app.add_option("-v,--verbosity", verbosity, "logging verbosity");
    app.add_option("-w,--wavefunction-choice", wfn_choice,
                   "Choice of wavefunctions");
    app.add_flag("--write-kmcpp", write_kmcpp_file,
                 "write out an input file for kmcpp program");
    app.add_flag("--xtb", use_xtb, "use xtb for interaction energies");
    app.add_flag("-d,--dump", write_dump_files, "Write dump files");
    app.add_flag("--atomic", crystal_is_atomic,
                 "Crystal is atomic (i.e. no bonds)");
    app.add_option(
        "--surface-energies", max_facets,
        "Calculate surface energies and write .gmf morphology files");
    app.add_flag("--list-available-solvents", list_solvents,
                 "List available solvents and exit");

    app.add_option("--covalent-bond-tolerance", bond_tolerance,
                   "tolerance for covalent bond detection (angstroms)");

    CLI11_PARSE(app, argc, argv);

    occ::log::setup_logging(verbosity);
    occ::core::set_bond_tolerance(bond_tolerance);

    if (list_solvents) {
        list_available_solvents();
        exit(0);
    }
    occ::timing::StopWatch global_timer;
    global_timer.start();

    occ::parallel::set_num_threads(std::max(1, threads));
#ifdef _OPENMP
    std::string thread_type = "OpenMP";
#else
    std::string thread_type = "std";
#endif
    occ::log::info("parallelization: {} {} threads, {} eigen threads",
                   occ::parallel::get_num_threads(), thread_type,
                   Eigen::nbThreads());
    const std::string error_format =
        "Exception:\n    {}\nTerminating program.\n";
    try {
        std::string basename = fs::path(cif_filename).stem().string();
        Crystal c_symm = read_crystal(cif_filename);

        if (crystal_is_atomic) {
            c_symm.set_connectivity_criteria(false);
        }

        if (use_xtb) {

            auto calc = XTBCrystalGrowthCalculator(c_symm, solvent);
            calc.set_basename(basename);
            if (!charge_string.empty()) {
                std::vector<int> charges;
                auto tokens = occ::util::tokenize(charge_string, ",");
                for (const auto &token : tokens) {
                    charges.push_back(std::stoi(token));
                }
                calc.set_molecule_charges(charges);
                calc.set_use_wolf_sum(true);
            }
            calc.converge_lattice_energy(cg_radius, radius);
            calc.evaluate_molecular_surroundings();

            auto uc_dimers = calc.crystal().unit_cell_dimers(cg_radius);
            auto uc_dimers_vacuum = uc_dimers;
            write_cg_structure_file(fmt::format("{}_cg.txt", basename),
                                    calc.crystal(), uc_dimers);

            auto solution_terms_uc = map_unique_interactions_to_uc_molecules(
                calc.crystal(), calc.full_dimers(), uc_dimers,
                calc.solution_terms(), calc.interaction_energies());

            auto solution_terms_uc_throwaway =
                map_unique_interactions_to_uc_molecules(
                    calc.crystal(), calc.full_dimers(), uc_dimers_vacuum,
                    calc.solution_terms(), calc.crystal_interaction_energies());

            if (write_kmcpp_file) {
                write_kmcpp_input_file(fmt::format("{}_kmcpp.json", basename),
                                       calc.crystal(), uc_dimers,
                                       solution_terms_uc);
            }
            write_cg_net_file(fmt::format("{}_net.txt", basename),
                              calc.crystal(), uc_dimers);

            if (max_facets > 0) {
                fmt::print("Crystal surface energies (solvated)\n");
                auto surface_energies =
                    occ::main::calculate_crystal_surface_energies(
                        fmt::format("{}_{}.gmf", basename, solvent),
                        calc.crystal(), uc_dimers, max_facets, 1);
                fmt::print("Crystal surface energies (vacuum)\n");
                auto vacuum_surface_energies =
                    occ::main::calculate_crystal_surface_energies(
                        fmt::format("{}_vacuum.gmf", basename), calc.crystal(),
                        uc_dimers_vacuum, max_facets, -1);

                nlohmann::json j;
                j["surface_energies"] = surface_energies;
                j["vacuum"] = calc.crystal_interaction_energies();
                j["solvated"] = calc.interaction_energies();
                std::ofstream destination(
                    fmt::format("{}_surface_energies.json", basename));
                destination << j.dump(2);
            }

        } else {
            auto calc = CEModelCrystalGrowthCalculator(c_symm, solvent);
            // Setup calculator parameters
            calc.set_basename(basename);
            calc.set_output_verbosity(write_dump_files);
            calc.set_energy_model(model_name);
            if (!charge_string.empty()) {
                std::vector<int> charges;
                auto tokens = occ::util::tokenize(charge_string, ",");
                for (const auto &token : tokens) {
                    charges.push_back(std::stoi(token));
                }
                calc.set_molecule_charges(charges);
                calc.set_use_wolf_sum(true);
                calc.set_use_crystal_polarization(true);
            }

            calc.set_wavefunction_choice(
                wfn_choice == "gas" ? CEModelCrystalGrowthCalculator::
                                          WavefunctionChoice::GasPhase
                                    : CEModelCrystalGrowthCalculator::
                                          WavefunctionChoice::Solvated);

            calc.init_wavefunctions();
            calc.converge_lattice_energy(cg_radius, radius);
            calc.evaluate_molecular_surroundings();

            auto uc_dimers = calc.crystal().unit_cell_dimers(cg_radius);
            auto uc_dimers_vacuum = uc_dimers;
            write_cg_structure_file(fmt::format("{}_cg.txt", basename),
                                    calc.crystal(), uc_dimers);

            auto solution_terms_uc = map_unique_interactions_to_uc_molecules(
                calc.crystal(), calc.full_dimers(), uc_dimers,
                calc.solution_terms(), calc.interaction_energies());

            auto solution_terms_uc_throwaway =
                map_unique_interactions_to_uc_molecules(
                    calc.crystal(), calc.full_dimers(), uc_dimers_vacuum,
                    calc.solution_terms(), calc.crystal_interaction_energies());

            if (write_kmcpp_file) {
                write_kmcpp_input_file(fmt::format("{}_kmcpp.json", basename),
                                       calc.crystal(), uc_dimers,
                                       solution_terms_uc);
            }

            auto write_uc_json = [](const occ::crystal::Crystal &crystal,
                                    const occ::crystal::CrystalDimers &dimers) {
                nlohmann::json j;
                j["title"] = "title";
                const auto &uc_molecules = crystal.unit_cell_molecules();
                j["unique_sites"] = uc_molecules.size();
                const auto &neighbors = dimers.molecule_neighbors;
                nlohmann::json molecules_json;
                molecules_json["kind"] = "atoms";
                j["lattice_vectors"] = crystal.unit_cell().direct();
                molecules_json["elements"] = {};
                molecules_json["positions"] = {};
                j["neighbor_offsets"] = {};
                size_t uc_idx_a = 0;
                j["neighbor_energies"] = {};
                for (const auto &mol : uc_molecules) {
                    nlohmann::json molj = mol;
                    molecules_json["elements"].push_back(molj["elements"]);
                    molecules_json["positions"].push_back(molj["positions"]);
                    j["neighbor_energies"].push_back(nlohmann::json::array({}));
                    std::vector<std::vector<int>> shifts;
                    for (const auto &[n, unique_idx] : neighbors[uc_idx_a]) {
                        const auto uc_shift = n.b().cell_shift();
                        const auto uc_idx_b = n.b().unit_cell_molecule_idx();
                        shifts.push_back(
                            {uc_shift[0], uc_shift[1], uc_shift[2], uc_idx_b});
                    }
                    j["neighbor_offsets"][uc_idx_a] = shifts;
                    const auto &neighbors_a = neighbors[uc_idx_a];
                    for (const auto &[n, unique_idx] : neighbors_a) {
                        j["neighbor_energies"][uc_idx_a].push_back(
                            n.interaction_energy() / occ::units::KJ_TO_KCAL);
                    }
                    uc_idx_a++;
                }
                j["molecules"] = molecules_json;

                std::ofstream dest("uc_interactions.json");
                dest << j.dump(2);
            };

            if (max_facets > 0) {
                fmt::print("Crystal surface energies (solvated)\n");
                auto surface_energies =
                    occ::main::calculate_crystal_surface_energies(
                        fmt::format("{}_{}.gmf", basename, solvent),
                        calc.crystal(), uc_dimers, max_facets, 1);
                fmt::print("Crystal surface energies (vacuum)\n");
                auto vacuum_surface_energies =
                    occ::main::calculate_crystal_surface_energies(
                        fmt::format("{}_vacuum.gmf", basename), calc.crystal(),
                        uc_dimers_vacuum, max_facets, -1);

                nlohmann::json j;
                j["surface_energies"] = surface_energies;
                j["vacuum"] = calc.crystal_interaction_energies();
                j["solvated"] = calc.interaction_energies();
                std::ofstream destination(
                    fmt::format("{}_surface_energies.json", basename));
                destination << j.dump(2);
            }

            write_uc_json(calc.crystal(), uc_dimers);

            write_cg_net_file(fmt::format("{}_{}_net.txt", basename, solvent),
                              calc.crystal(), uc_dimers);

            calc.dipole_correction();
        }

    } catch (const char *ex) {
        fmt::print(error_format, ex);
        return 1;
    } catch (std::string &ex) {
        fmt::print(error_format, ex);
        return 1;
    } catch (std::exception &ex) {
        fmt::print(error_format, ex.what());
        return 1;
    } catch (...) {
        occ::log::critical("Exception:\n- Unknown...\n");
        return 1;
    }

    global_timer.stop();
    occ::log::info("Program exiting successfully after {:.6f} seconds",
                   global_timer.read());

    return 0;
}
