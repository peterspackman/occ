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
#include <occ/io/load_geometry.h>
#include <occ/io/core_json.h>
#include <occ/io/crystalgrower.h>
#include <occ/io/eigen_json.h>
#include <occ/io/kmcpp.h>
#include <occ/io/occ_input.h>
#include <occ/io/wavefunction_json.h>
#include <occ/io/xyz.h>
#include <occ/main/crystal_surface_energy.h>
#include <occ/main/occ_cg.h>
#include <occ/main/pair_energy.h>
#include <occ/main/single_point.h>
#include <occ/main/solvation_partition.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/solvation_correction.h>
#include <occ/xtb/xtb_wrapper.h>

namespace fs = std::filesystem;
using occ::core::Element;
using occ::core::Molecule;
using occ::crystal::Crystal;
using occ::crystal::CrystalDimers;
using occ::crystal::SymmetryOperation;
using occ::qm::HartreeFock;
using occ::qm::Wavefunction;
using occ::units::AU_TO_KJ_PER_MOL;
using occ::units::BOHR_TO_ANGSTROM;
using SolventNeighborContributionList =
    std::vector<occ::main::SolventNeighborContribution>;
using occ::interaction::CEEnergyComponents;
using occ::main::SolvatedSurfaceProperties;

using WavefunctionList = std::vector<Wavefunction>;
using DimerEnergies = std::vector<CEEnergyComponents>;

enum class WavefunctionChoice { GasPhase, Solvated };

Wavefunction load_or_calculate_wavefunction(const Molecule &mol,
					    const std::string &name,
					    const std::string &energy_model) {
    fs::path json_path(fmt::format("{}.owf.json", name));
    if (fs::exists(json_path)) {
	occ::log::info("Loading wavefunction from {}", json_path.string());
	return Wavefunction::load(json_path);
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

    wfn.save(json_path.string());
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
            load_or_calculate_wavefunction(m, name, energy_model));
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
	    occ::log::debug("Atomic numbers\n{}\n", m_molecules[i].atomic_numbers());
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
            occ::log::debug(
                "Net dipole for molecule shell {} = ({:.3f} {:.3f} {:.3f})", i,
                dipole(0), dipole(1), dipole(2));
            double e = -2 * M_PI * dipole.squaredNorm() / (3 * V) *
                       occ::units::AU_TO_KJ_PER_MOL;
            occ::log::debug(
                "Energy = {:.6f} kJ/mol ({:.3f} per molecule)", e,
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

    std::tuple<occ::main::EnergyTotal, std::vector<occ::main::CGDimer>>
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

        occ::log::warn("nn {:>3s} {:>7s} {:>7s} {:>20s} "
                       "{:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}",
		       "id",
                       "Rn", "Rc", "Symop", "E_crys", "ES_AB", "ES_BA", "E_S",
                       "E_nn", "E_int");

        occ::log::warn(std::string(91, '='));

        size_t j = 0;
        occ::main::EnergyTotal total;
        std::vector<occ::main::CGDimer> dimer_energy_results;

        size_t num_neighbors = std::accumulate(
            crystal_contributions.begin(), crystal_contributions.end(), 0,
            [](size_t a, const AssignedEnergy &x) {
                return x.is_nn ? a + 1 : a;
            });

        total.solution_term = surface_properties.esolv * AU_TO_KJ_PER_MOL;

        const std::string row_fmt_string =
            " {} {:>3d} {:>7.2f} {:>7.2f} {:>20s} {: 7.2f} "
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

            total.crystal_energy += e.total_kjmol();

            double interaction_energy =
                solvent_neighbor_contribution.total_kjmol() - e.total_kjmol() -
                crystal_contributions[j].energy;

            if (is_nearest_neighbor) {
                total.interaction_energy += interaction_energy;
                interactions.push_back(interaction_energy);
                interactions_crystal.push_back(e.total_kjmol() +
                                               crystal_contributions[j].energy);
            } else {
                interactions.push_back(0.0);
                interactions_crystal.push_back(0.0);
                interaction_energy = 0;
            }

            occ::main::DimerSolventTerm solvent_term;
            solvent_term.ab = (solvent_neighbor_contribution.coulomb.ab +
                               solvent_neighbor_contribution.cds.ab) *
                              AU_TO_KJ_PER_MOL;
            solvent_term.ba = (solvent_neighbor_contribution.coulomb.ba +
                               solvent_neighbor_contribution.cds.ba) *
                              AU_TO_KJ_PER_MOL;

            solvent_term.total = solvent_neighbor_contribution.total_kjmol();

            if (is_nearest_neighbor) {
                occ::log::warn(fmt::runtime(row_fmt_string), "|", unique_idx, rn, rc,
                               symmetry_string, e.total_kjmol(),
                               solvent_term.ab, solvent_term.ba,
                               solvent_term.total, crystal_contribution,
                               interaction_energy);
            } else {
                occ::log::debug(fmt::runtime(row_fmt_string), " ", unique_idx, rn, rc,
                                symmetry_string, e.total_kjmol(),
                                solvent_term.ab, solvent_term.ba,
                                solvent_term.total, crystal_contribution,
                                interaction_energy);
            }
            dimer_energy_results.emplace_back(occ::main::CGDimer{
                dimer, unique_idx, interaction_energy, solvent_term,
                crystal_contribution, is_nearest_neighbor});
            j++;
        }
        m_solvation_breakdowns.push_back(solvation_breakdown);
        return {total, dimer_energy_results};
    }

    occ::main::CGResult evaluate_molecular_surroundings() {
        occ::main::CGResult result;

        m_solution_terms = std::vector<double>(m_molecules.size(), 0.0);
        for (size_t i = 0; i < m_molecules.size(); i++) {
            auto [mol_total, mol_dimer_results] =
                process_neighbors_for_symmetry_unique_molecule(
                    i, fmt::format("{}_{}_{}", m_basename, i, m_solvent));

            result.pair_energies.push_back(mol_dimer_results);
            result.total_energies.push_back(mol_total);

            m_solution_terms[i] = mol_total.solution_term;
            m_lattice_energies.push_back(mol_total.crystal_energy);
            write_energy_summary(mol_total.crystal_energy, m_molecules[i],
                                 mol_total.solution_term,
                                 mol_total.interaction_energy);

            if (m_output) {
                // write neighbors file for molecule i
                std::string neighbors_filename =
                    fmt::format("{}_{}_neighbors.xyz", m_basename, i);
                write_xyz_neighbors(neighbors_filename,
                                    m_full_dimers.molecule_neighbors[i]);
            }
        }
        return result;
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

    // do nothing
    inline void set_wavefunction_choice(WavefunctionChoice choice) {}
    inline void set_energy_model(const std::string &model) { m_model = model; }
    inline void set_use_crystal_polarization(bool value) {}
    void set_output_verbosity(bool output) { m_output = output; }

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
        occ::log::info(
            "Computing solvated dimer energies for nearest neighbors");
        for (const auto &dimer : m_full_dimers.unique_dimers) {
            m_solvated_dimer_energies[unique_idx] = 0.0;
            if (dimer.nearest_distance() <= 3.8) {
                occ::xtb::XTBCalculator xtb(dimer);
                xtb.set_solvent(m_solvent);
		xtb.set_solvation_model(m_solvation_model);
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

    occ::main::CGResult evaluate_molecular_surroundings() {
        occ::main::CGResult result;

        m_solution_terms = std::vector<double>(m_molecules.size(), 0.0);
        for (size_t i = 0; i < m_molecules.size(); i++) {
            auto [mol_total, mol_dimer_results] =
                process_neighbors_for_symmetry_unique_molecule(
                    i, fmt::format("{}_{}_{}", m_basename, i, m_solvent));

            result.pair_energies.push_back(mol_dimer_results);
            result.total_energies.push_back(mol_total);

            m_solution_terms[i] = mol_total.solution_term;
            m_lattice_energies.push_back(mol_total.crystal_energy);
            write_energy_summary(mol_total.crystal_energy, m_molecules[i],
                                 mol_total.solution_term,
                                 mol_total.interaction_energy);
        }
        return result;
    }

    void init_monomer_energies() {
        occ::timing::StopWatch sw_gas;
        occ::timing::StopWatch sw_solv;

        size_t index = 0;
        for (const auto &m : m_molecules) {
            occ::log::info("Molecule ({})\n{:3s} {:^10s} {:^10s} {:^10s}",
                           index, "sym", "x", "y", "z");
            for (const auto &atom : m.atoms()) {
                occ::log::info("{:^3s} {:10.6f} {:10.6f} {:10.6f}",
                               Element(atom.atomic_number).symbol(), atom.x,
                               atom.y, atom.z);
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
                xtb.set_solvent(m_solvent);
		xtb.set_solvation_model(m_solvation_model);
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
        occ::log::info("Gas phase calculations took {:.6f} seconds",
                       sw_gas.read());
        occ::log::info("Solution phase calculations took {:.6f} seconds",
                       sw_solv.read());
    }

    void set_solvation_model(const std::string &model) { m_solvation_model = model; }

  private:
    std::tuple<occ::main::EnergyTotal, std::vector<occ::main::CGDimer>>
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

        occ::main::EnergyTotal total;
        std::vector<occ::main::CGDimer> dimer_energy_results;
        total.solution_term =
            (m_solvated_energies[i] - m_gas_phase_energies[i]) *
            AU_TO_KJ_PER_MOL;


        double dimers_solv_total = 0.0;
        {
            size_t j = 0;
            for (const auto &[dimer, unique_idx] : full_neighbors) {
                dimers_solv_total += m_solvated_dimer_energies[unique_idx];
                j++;
            }
            dimers_solv_total *= AU_TO_KJ_PER_MOL;
        }
        double dimers_solv_scale_factor =
            total.solution_term * 2 / dimers_solv_total;

        occ::log::debug("Total dimers solvation: {} vs {}", dimers_solv_total,
                        total.solution_term);

        occ::log::warn("Neighbors for asymmetric molecule {}", molname);

        occ::log::warn("nn {:>3s} {:>7s} {:>7s} {:>20s} "
                       "{:>7s} {:>7s} {:>7s} {:>7s}",
		       "id",
                       "Rn", "Rc", "Symop", "E_crys", "E_solv", "E_nn",
                       "E_int");
        occ::log::warn(std::string(92, '='));

        size_t j = 0;
        for (const auto &[dimer, unique_idx] : full_neighbors) {
            double e = m_dimer_energies[unique_idx];
            auto symmetry_string = m_crystal.dimer_symmetry_string(dimer);
            double rn = dimer.nearest_distance();
            double rc = dimer.centroid_distance();
            double crystal_contribution = crystal_contributions[j].energy;
            bool is_nearest_neighbor = crystal_contributions[j].is_nn;

            occ::Vec3 v_ab = dimer.v_ab();

            total.crystal_energy += e;

            occ::main::DimerSolventTerm solvent_term;
            solvent_term.total = m_solvated_dimer_energies[unique_idx] *
                                 AU_TO_KJ_PER_MOL * dimers_solv_scale_factor;

            double interaction_energy =
                solvent_term.total - e - -crystal_contributions[j].energy;

            if (is_nearest_neighbor) {
                total.interaction_energy += interaction_energy;
                interactions.push_back(interaction_energy);
                interactions_crystal.push_back(e +
                                               crystal_contributions[j].energy);
            } else {
                interactions.push_back(0.0);
                interactions_crystal.push_back(0.0);
                interaction_energy = 0;
            }

	    if(is_nearest_neighbor) {
		occ::log::warn(" {} {:>3d} {: 7.2f} {: 7.2f} {:>20s} {: 7.2f} {: 7.2f} {: 7.2f} {: 8.2f}",
			       '|', unique_idx, rn, rc, symmetry_string, e, solvent_term.total,
			       crystal_contribution, interaction_energy);

	    }
	    else {
		occ::log::debug(" {} {:>3d} {: 7.2f} {: 7.2f} {:>20s} {: 7.2f} {: 7.2f} {: 7.2f} {: 8.2f}",
			       ' ', unique_idx, rn, rc, symmetry_string, e, solvent_term.total,
			       crystal_contribution, interaction_energy);
	    }

            dimer_energy_results.emplace_back(occ::main::CGDimer{
                dimer, unique_idx, interaction_energy, solvent_term,
                crystal_contribution, is_nearest_neighbor});
            j++;
        }
        return {total, dimer_energy_results};
    }

    const auto &lattice_energies() const { return m_lattice_energies; }

    Crystal m_crystal;
    std::vector<occ::core::Molecule> m_molecules;
    std::vector<double> m_lattice_energies;
    std::string m_solvent;
    std::string m_basename;
    std::string m_model{"gfn2-xtb"};
    std::string m_solvation_model{"cpcmx"};
    bool m_output{true};
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

namespace occ::main {

CLI::App *add_cg_subcommand(CLI::App &app) {
    CLI::App *cg =
        app.add_subcommand("cg", "compute crystal growth free energies");
    auto config = std::make_shared<CGConfig>();
    cg->add_option("input", config->lattice_settings.crystal_filename,
                   "input CIF")
        ->required();
    cg->add_option("-r,--radius", config->lattice_settings.max_radius,
                   "maximum radius (Angstroms) for neighbours");
    cg->add_option("-m,--model", config->lattice_settings.model_name,
                   "energy model");
    cg->add_option("-c,--cg-radius", config->cg_radius,
                   "maximum radius (Angstroms) for nearest neighbours in CG "
                   "file (must be <= radius)");
    cg->add_option("-s,--solvent", config->solvent, "solvent name");
    cg->add_option("--charges", config->charge_string, "system net charge");
    cg->add_option("-w,--wavefunction-choice", config->wavefunction_choice,
                   "Choice of wavefunctions");
    cg->add_flag("--write-kmcpp", config->write_kmcpp_file,
                 "write out an input file for kmcpp program");
    cg->add_flag("--xtb", config->use_xtb, "use xtb for interaction energies");
    cg->add_option("--xtb-solvation-model,--xtb_solvation_model", config->xtb_solvation_model, "solvation model for use with xtb interaction energies");
    cg->add_flag("-d,--dump", config->write_dump_files, "Write dump files");
    cg->add_flag("--atomic", config->crystal_is_atomic,
                 "Crystal is atomic (i.e. no bonds)");
    cg->add_option(
        "--surface-energies", config->max_facets,
        "Calculate surface energies and write .gmf morphology files");
    cg->add_flag("--list-available-solvents", config->list_solvents,
                 "List available solvents and exit");
    cg->fallthrough();
    cg->callback([config]() { run_cg_subcommand(*config); });
    return cg;
}

template <class Calculator> CGResult run_cg_impl(CGConfig const &config) {
    std::string basename =
        fs::path(config.lattice_settings.crystal_filename).stem().string();
    Crystal c_symm = occ::io::load_crystal(config.lattice_settings.crystal_filename);

    if (config.crystal_is_atomic) {
        c_symm.set_connectivity_criteria(false);
    }

    auto calc = Calculator(c_symm, config.solvent);
    // Setup calculator parameters
    calc.set_basename(basename);
    calc.set_output_verbosity(config.write_dump_files);
    calc.set_energy_model(config.lattice_settings.model_name);
    if (!config.charge_string.empty()) {
        std::vector<int> charges;
        auto tokens = occ::util::tokenize(config.charge_string, ",");
        for (const auto &token : tokens) {
            charges.push_back(std::stoi(token));
        }
        calc.set_molecule_charges(charges);
        calc.set_use_wolf_sum(true);
        calc.set_use_crystal_polarization(true);
    }

    if constexpr(std::is_same<Calculator, XTBCrystalGrowthCalculator>::value) {
	occ::log::info("XTB solvation model: {}", config.xtb_solvation_model);
	calc.set_solvation_model(config.xtb_solvation_model);
    }

    calc.set_wavefunction_choice(config.wavefunction_choice == "gas"
                                     ? WavefunctionChoice::GasPhase
                                     : WavefunctionChoice::Solvated);

    calc.init_monomer_energies();
    calc.converge_lattice_energy(config.cg_radius,
                                 config.lattice_settings.max_radius);

    CGResult result = calc.evaluate_molecular_surroundings();

    auto uc_dimers = calc.crystal().unit_cell_dimers(config.cg_radius);
    auto uc_dimers_vacuum = uc_dimers;
    write_cg_structure_file(fmt::format("{}_cg.txt", basename), calc.crystal(),
                            uc_dimers);

    auto solution_terms_uc = map_unique_interactions_to_uc_molecules(
        calc.crystal(), calc.full_dimers(), uc_dimers, calc.solution_terms(),
        calc.interaction_energies());

    auto solution_terms_uc_throwaway = map_unique_interactions_to_uc_molecules(
        calc.crystal(), calc.full_dimers(), uc_dimers_vacuum,
        calc.solution_terms(), calc.crystal_interaction_energies());

    if (config.write_kmcpp_file) {
        write_kmcpp_input_file(fmt::format("{}_kmcpp.json", basename),
                               calc.crystal(), uc_dimers, solution_terms_uc);
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

    if (config.max_facets > 0) {
        occ::log::info("Crystal surface energies (solvated)");
        auto surface_energies = occ::main::calculate_crystal_surface_energies(
            fmt::format("{}_{}.gmf", basename, config.solvent), calc.crystal(),
            uc_dimers, config.max_facets, 1);
        occ::log::info("Crystal surface energies (vacuum)");
        auto vacuum_surface_energies =
            occ::main::calculate_crystal_surface_energies(
                fmt::format("{}_vacuum.gmf", basename), calc.crystal(),
                uc_dimers_vacuum, config.max_facets, -1);

        nlohmann::json j;
        j["surface_energies"] = surface_energies;
        j["vacuum"] = calc.crystal_interaction_energies();
        j["solvated"] = calc.interaction_energies();
        std::string surf_energy_filename =
            fmt::format("{}_surface_energies.json", basename);
        occ::log::info("Writing surface energies to '{}'",
                       surf_energy_filename);
        std::ofstream destination(surf_energy_filename);
        destination << j.dump(2);
    }

    write_uc_json(calc.crystal(), uc_dimers);

    write_cg_net_file(fmt::format("{}_{}_net.txt", basename, config.solvent),
                      calc.crystal(), uc_dimers);

    // calc.dipole_correction();
    return result;
}

CGResult run_cg(CGConfig const &config) {
    CGResult result;
    std::string basename =
        fs::path(config.lattice_settings.crystal_filename).stem().string();
    Crystal c_symm = occ::io::load_crystal(config.lattice_settings.crystal_filename);

    if (config.crystal_is_atomic) {
        c_symm.set_connectivity_criteria(false);
    }

    if (config.use_xtb) {
        result = run_cg_impl<XTBCrystalGrowthCalculator>(config);
    } else {
        result = run_cg_impl<CEModelCrystalGrowthCalculator>(config);
    }
    return result;
}

void run_cg_subcommand(CGConfig const &config) {
    if (config.list_solvents) {
	occ::solvent::list_available_solvents();
        return;
    }
    (void)run_cg(config);
}

} // namespace occ::main
