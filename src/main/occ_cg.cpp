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
#include <occ/io/crystalgrower.h>
#include <occ/io/eigen_json.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/io/kmcpp.h>
#include <occ/io/occ_input.h>
#include <occ/io/xyz.h>
#include <occ/main/pair_energy.h>
#include <occ/main/single_point.h>
#include <occ/main/solvation_partition.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/solvation_correction.h>
#include <scn/scn.h>

namespace fs = std::filesystem;
using occ::core::Dimer;
using occ::core::Element;
using occ::core::Molecule;
using occ::crystal::Crystal;
using occ::crystal::SymmetryOperation;
using occ::qm::HartreeFock;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
using occ::scf::SCF;
using occ::units::AU_TO_KJ_PER_MOL;
using occ::units::BOHR_TO_ANGSTROM;

namespace occ::qm {
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Energy, coulomb, exchange, nuclear_repulsion,
                                   nuclear_attraction, kinetic, core, total)
}

Crystal read_crystal(const std::string &filename) {
    occ::io::CifParser parser;
    return parser.parse_crystal(filename).value();
}

Wavefunction calculate_wavefunction(const Molecule &mol,
                                    const std::string &name) {
    fs::path fchk_path(fmt::format("{}.fchk", name));
    if (fs::exists(fchk_path)) {
        occ::log::info("Loading gas phase wavefunction from {}",
                       fchk_path.string());
        using occ::io::FchkReader;
        FchkReader fchk(fchk_path.string());
        return Wavefunction(fchk);
    }

    const std::string method = "b3lyp";
    const std::string basis_name = "6-31G**";
    occ::io::OccInput input;
    input.method.name = method;
    input.basis.name = basis_name;
    input.geometry.set_molecule(mol);
    input.electronic.charge = mol.charge();
    input.electronic.multiplicity = mol.multiplicity();
    auto wfn = occ::main::single_point_calculation(input);

    occ::io::FchkWriter fchk(fchk_path.string());
    fchk.set_title(fmt::format("{} {}/{} generated by occ-ng",
                               fchk_path.stem().string(), method, basis_name));
    fchk.set_method(method);
    fchk.set_basis_name(basis_name);
    wfn.save(fchk);
    fchk.write();
    return wfn;
}

std::vector<Wavefunction>
calculate_wavefunctions(const std::string &basename,
                        const std::vector<Molecule> &molecules) {
    std::vector<Wavefunction> wfns;
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
        wfns.emplace_back(calculate_wavefunction(m, name));
        index++;
    }
    return wfns;
}

std::vector<occ::Vec3>
calculate_net_dipole(const std::vector<Wavefunction> &wfns,
                     const occ::crystal::CrystalDimers &crystal_dimers) {
    std::vector<occ::Vec3> dipoles;
    std::vector<occ::Vec> partial_charges;
    for (const auto &wfn : wfns) {
        partial_charges.push_back(wfn.mulliken_charges());
    }
    for (size_t idx = 0; idx < crystal_dimers.molecule_neighbors.size();
         idx++) {
        occ::Vec3 dipole = occ::Vec3::Zero(3);
        size_t j = 0;
        for (const auto &dimer : crystal_dimers.molecule_neighbors[idx]) {
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
                              std::vector<Wavefunction> &wfns) {
    size_t idx = 0;
    for (auto &wfn : wfns) {
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
            occ::Mat schwarz = hf.compute_schwarz_ints();
            occ::interaction::compute_ce_model_energies(wfn, hf, 1e-8, schwarz);
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
                         const std::vector<Dimer> &neighbors) {
    auto neigh = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                                fmt::file::CREATE);

    size_t natom = std::accumulate(
        neighbors.begin(), neighbors.end(), 0,
        [](size_t a, const auto &dimer) { return a + dimer.b().size(); });

    neigh.print("{}\nel x y z idx\n", natom);

    size_t j = 0;
    for (const auto &dimer : neighbors) {
        auto pos = dimer.b().positions();
        auto els = dimer.b().elements();
        for (size_t a = 0; a < dimer.b().size(); a++) {
            neigh.print("{:.3s} {:12.5f} {:12.5f} {:12.5f} {:5d}\n",
                        els[a].symbol(), pos(0, a), pos(1, a), pos(2, a), j);
        }
        j++;
    }
}

struct AssignedEnergy {
    bool is_nn{true};
    double energy{0.0};
};

std::vector<AssignedEnergy> assign_interaction_terms_to_nearest_neighbours(
    int mol_idx, const occ::crystal::CrystalDimers &crystal_dimers,
    const std::vector<occ::interaction::CEEnergyComponents> dimer_energies,
    double cg_radius) {
    double total_taken{0.0};
    const auto &n = crystal_dimers.molecule_neighbors[mol_idx];
    std::vector<AssignedEnergy> crystal_contributions(n.size());
    for (size_t k1 = 0; k1 < crystal_contributions.size(); k1++) {
        if (n[k1].nearest_distance() <= cg_radius)
            continue;
        crystal_contributions[k1].is_nn = false;
        auto v = n[k1].v_ab().normalized();
        auto unique_dimer_idx = crystal_dimers.unique_dimer_idx[mol_idx][k1];

        // skip if not contributing
        if (!dimer_energies[unique_dimer_idx].is_computed)
            continue;

        total_taken += dimer_energies[unique_dimer_idx].total_kjmol();
        double total_dp = 0.0;
        size_t number_interactions = 0;
        for (size_t k2 = 0; k2 < crystal_contributions.size(); k2++) {
            if (n[k2].nearest_distance() > cg_radius)
                continue;
            if (k1 == k2)
                continue;
            auto v2 = n[k2].v_ab().normalized();
            double dp = v.dot(v2);
            if (dp <= 0.0)
                continue;
            total_dp += dp;
            number_interactions++;
        }
        for (size_t k2 = 0; k2 < crystal_contributions.size(); k2++) {
            if (n[k2].nearest_distance() > cg_radius)
                continue;
            if (k1 == k2)
                continue;
            auto v2 = n[k2].v_ab().normalized();
            double dp = v.dot(v2);
            if (dp <= 0.0)
                continue;
            crystal_contributions[k2].is_nn = true;
            crystal_contributions[k2].energy +=
                (dp / total_dp) *
                dimer_energies[unique_dimer_idx].total_kjmol();
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

int main(int argc, char **argv) {
    CLI::App app(
        "occ-cg - Interactions of molecules with neighbours in a crystal");
    std::string cif_filename{""}, charge_string{""}, region_string{""},
        verbosity{"warn"}, solvent{"water"}, wfn_choice{"gas"};

    int threads{1};
    double radius{3.8}, cg_radius{3.8};
    bool write_dump_files{false}, spherical{false};

    CLI::Option *input_option =
        app.add_option("input", cif_filename, "input CIF");
    input_option->required();
    app.add_option("-t,--threads", threads, "number of threads");
    app.add_option("-r,--radius", radius,
                   "maximum radius (Angstroms) for neighbours");
    app.add_option("-c,--cg-radius", cg_radius,
                   "maximum radius (Angstroms) for nearest neighbours in CG "
                   "file (must be <= radius)");
    app.add_option("-s,--solvent", solvent, "solvent name");
    app.add_option("--charges", charge_string, "system net charge");
    app.add_option("-g,--region", region_string,
                   "Restrict to this region (xyz file)");
    app.add_option("-v,--verbosity", verbosity, "logging verbosity");
    app.add_option("-w,--wavefunction-choice", wfn_choice,
                   "Choice of wavefunctions");
    app.add_flag("-d,--dump", write_dump_files, "Write dump files");

    CLI11_PARSE(app, argc, argv);

    occ::log::setup_logging(verbosity);
    occ::timing::StopWatch global_timer;
    global_timer.start();
    std::optional<Molecule> region;

    if (!region_string.empty()) {
        region = occ::io::molecule_from_xyz_file(region_string);
    }
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
        occ::timing::StopWatch sw;
        sw.start();
        std::string basename = fs::path(cif_filename).stem();
        Crystal c_symm = read_crystal(cif_filename);
        sw.stop();
        double tprev = sw.read();
        occ::log::debug("Loaded crystal from {} in {:.6f} seconds",
                        cif_filename, tprev);

        sw.start();
        auto molecules = c_symm.symmetry_unique_molecules();
        sw.stop();

        occ::log::debug(
            "Found {} symmetry unique molecules in {} in {:.6f} seconds",
            molecules.size(), cif_filename, sw.read() - tprev);

        if (!charge_string.empty()) {
            auto tokens = occ::util::tokenize(charge_string, ",");
            if (tokens.size() != molecules.size()) {
                throw fmt::format(
                    "Require {} charges to be specified, found {}",
                    molecules.size(), tokens.size());
            }
            size_t i = 0;
            for (const auto &token : tokens) {
                molecules[i].set_charge(std::stoi(token));
                i++;
            }
        }

        tprev = sw.read();
        sw.start();
        auto wfns = calculate_wavefunctions(basename, molecules);
        sw.stop();

        occ::log::info("Gas phase wavefunctions took {:.6f} seconds",
                       sw.read() - tprev);

        tprev = sw.read();
        sw.start();
        std::vector<Wavefunction> solvated_wavefunctions;
        std::vector<occ::main::SolvatedSurfaceProperties> surfaces;
        std::tie(surfaces, solvated_wavefunctions) =
            occ::main::calculate_solvated_surfaces(basename, molecules, wfns,
                                                   solvent);
        sw.stop();
        occ::log::info("Solution phase wavefunctions took {:.6f} seconds",
                       sw.read() - tprev);

        tprev = sw.read();
        sw.start();
        occ::log::info("Computing monomer energies for gas phase");
        compute_monomer_energies(basename, wfns);
        occ::log::info("Computing monomer energies for solution phase");
        compute_monomer_energies(fmt::format("{}_{}", basename, solvent),
                                 solvated_wavefunctions);
        sw.stop();
        occ::log::info("Computing monomer energies took {:.6f} seconds",
                       sw.read() - tprev);

        const std::string row_fmt_string =
            "{:>7.2f} {:>7.2f} {:>20s} {: 7.2f} "
            "{: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f}";

        const auto &wfns_a = [&]() {
            if (wfn_choice == "gas")
                return wfns;
            else
                return solvated_wavefunctions;
        }();
        const auto &wfns_b = [&]() {
            if (wfn_choice == "solvent")
                return solvated_wavefunctions;
            else
                return wfns;
        }();

        occ::crystal::CrystalDimers crystal_dimers;
        std::vector<occ::interaction::CEEnergyComponents> dimer_energies;

        occ::main::LatticeConvergenceSettings convergence_settings;
        convergence_settings.max_radius = radius;
        occ::log::info("Computing crystal interactions using {} wavefunctions",
                       wfn_choice);
        std::tie(crystal_dimers, dimer_energies) =
            occ::main::converged_lattice_energies(
                c_symm, wfns_a, wfns_b, basename, convergence_settings);

        auto cg_symm_dimers = c_symm.symmetry_unique_dimers(cg_radius);

        if (crystal_dimers.unique_dimers.size() < 1) {
            occ::log::error("No dimers found using neighbour radius {:.3f}",
                            radius);
            exit(0);
        }

        auto dipoles = calculate_net_dipole(wfns_b, crystal_dimers);
        {
            size_t i = 0;
            for (const auto &x : dipoles) {
                occ::log::debug("Net dipole for cluster {}: ({:12.5f}, "
                                "{:12.5f} {:12.5f})",
                                i, x(0), x(1), x(2));
                i++;
            }
        }

        occ::main::SolvationPartitionScheme partition_scheme =
            occ::main::SolvationPartitionScheme::NearestAtom;

        const auto &mol_neighbors = crystal_dimers.molecule_neighbors;
        std::vector<std::vector<occ::main::SolventNeighborContribution>>
            solvation_breakdowns;
        std::vector<std::vector<double>> interaction_energies_vec(
            mol_neighbors.size());
        std::vector<double> solution_terms(mol_neighbors.size(), 0.0);
        double dG_solubility{0.0};
        for (size_t i = 0; i < mol_neighbors.size(); i++) {
            const auto &n = mol_neighbors[i];
            std::string molname = fmt::format("{}_{}_{}", basename, i, solvent);
            std::ofstream gulp_file(fmt::format("{}_{}_gulp.txt", basename, i));
            fmt::print(gulp_file, "{:3s} {:12s} {:12s} {:12s} {:12s} {:12s}\n",
                       "N", "x", "y", "z", "r", "energy");
            auto solv = occ::main::partition_solvent_surface(
                partition_scheme, c_symm, molname, wfns_b, surfaces[i], n,
                cg_symm_dimers.molecule_neighbors[i], solvent);
            solvation_breakdowns.push_back(solv);
            auto crystal_contributions =
                assign_interaction_terms_to_nearest_neighbours(
                    i, crystal_dimers, dimer_energies, cg_radius);
            auto &interactions = interaction_energies_vec[i];
            interactions.reserve(n.size());
            double Gr = molecules[i].rotational_free_energy(298);
            occ::core::MolecularPointGroup pg(molecules[i]);
            occ::log::debug(
                "Molecule {} point group = {}, symmetry number = {}", i,
                pg.point_group_string(), pg.symmetry_number());
            double Gt = molecules[i].translational_free_energy(298);
            double molar_mass = molecules[i].molar_mass();

            occ::log::warn("Neighbors for asymmetric molecule {}", i);

            occ::log::warn("{:>7s} {:>7s} {:>20s} "
                           "{:>7s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}",
                           "Rn", "Rc", "Symop", "E_crys", "ES_AB", "ES_BA",
                           "E_S", "E_nn", "E_int");
            occ::log::warn("============================="
                           "============================="
                           "=============================");

            size_t j = 0;
            occ::interaction::CEEnergyComponents total;

            if (write_dump_files) {
                // write neighbors file for molecule i
                std::string neighbors_filename =
                    fmt::format("{}_{}_neighbors.xyz", basename, i);
                write_xyz_neighbors(neighbors_filename, n);
            }

            size_t num_neighbors = std::accumulate(
                crystal_contributions.begin(), crystal_contributions.end(), 0,
                [](size_t a, const AssignedEnergy &x) {
                    return x.is_nn ? a + 1 : a;
                });

            solution_terms[i] =
                (surfaces[i].dg_gas + surfaces[i].dg_correction) *
                AU_TO_KJ_PER_MOL;

            double total_interaction_energy{0.0};

            for (const auto &dimer : n) {
                if (region) {
                    const auto &b = dimer.b();
                    auto [idx_region_b, idx_mol_b, distance_rb] =
                        region.value().nearest_atom(b);
                    if (distance_rb > 1e-3) {
                        occ::log::debug("Excluding {}", distance_rb);
                        j++;
                        continue;
                    }
                }
                auto s_ab = c_symm.dimer_symmetry_string(dimer);
                size_t idx = crystal_dimers.unique_dimer_idx[i][j];
                double rn = dimer.nearest_distance();
                double rc = dimer.centroid_distance();
                occ::Vec3 v_ab = dimer.v_ab();
                const auto &e =
                    dimer_energies[crystal_dimers.unique_dimer_idx[i][j]];
                if (!e.is_computed) {
                    j++;
                    continue;
                }
                double ecoul = e.coulomb_kjmol(), erep = e.exchange_kjmol(),
                       epol = e.polarization_kjmol(),
                       edisp = e.dispersion_kjmol(), etot = e.total_kjmol();
                total.coulomb += ecoul;
                total.exchange_repulsion += erep;
                total.polarization += epol;
                total.dispersion += edisp;
                total.total += etot;

                fmt::print(
                    gulp_file,
                    "{:3d} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.6f}\n", j,
                    v_ab(0), v_ab(1), v_ab(2), v_ab.norm(), etot);

                double solv_cont = solv[j].total() * AU_TO_KJ_PER_MOL;

                double e_int =
                    solv_cont - etot - crystal_contributions[j].energy;
                if (!crystal_contributions[j].is_nn) {
                    e_int = 0;
                } else {
                    total_interaction_energy += e_int;
                }
                interactions.push_back(e_int);
                auto &sj = solv[j];

                occ::log::warn(row_fmt_string, rn, rc, s_ab, etot,
                               (sj.coulomb.ab + sj.cds.ab) * AU_TO_KJ_PER_MOL,
                               (sj.coulomb.ba + sj.cds.ba) * AU_TO_KJ_PER_MOL,
                               sj.total() * AU_TO_KJ_PER_MOL,
                               crystal_contributions[j].energy, e_int);
                j++;
            }
            constexpr double R = 8.31446261815324;
            constexpr double RT = 298 * R / 1000;
            occ::log::warn("Free energy estimates at T = 298 K, P = 1 atm., "
                           "units: kJ/mol");
            occ::log::warn(
                "-------------------------------------------------------");
            occ::log::warn(
                "lattice energy (crystal)             {: 9.3f}  (E_lat)",
                0.5 * total.total);
            Gr += RT * std::log(pg.symmetry_number());
            occ::log::warn(
                "rotational free energy (molecule)    {: 9.3f}  (E_rot)", Gr);
            occ::log::warn(
                "translational free energy (molecule) {: 9.3f}  (E_trans)", Gt);
            // includes concentration shift
            double dG_solv = surfaces[i].esolv * occ::units::AU_TO_KJ_PER_MOL +
                             1.89 / occ::units::KJ_TO_KCAL;
            occ::log::warn(
                "solvation free energy (molecule)     {: 9.3f}  (E_solv)",
                dG_solv);
            double dH_sub = -0.5 * total.total - 2 * RT;
            occ::log::warn("\u0394H sublimation                       {: 9.3f}",
                           dH_sub);
            double dS_sub = Gr + Gt;
            occ::log::warn("\u0394S sublimation                       {: 9.3f}",
                           dS_sub);
            double dG_sub = dH_sub + dS_sub;
            occ::log::warn("\u0394G sublimation                       {: 9.3f}",
                           dG_sub);
            dG_solubility = dG_solv + dG_sub;
            occ::log::warn("\u0394G solution                          {: 9.3f}",
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

        auto uc_dimers = c_symm.unit_cell_dimers(cg_radius);
        auto &uc_neighbors = uc_dimers.molecule_neighbors;

        // write CG structure file
        {
            std::string cg_structure_filename =
                fmt::format("{}_cg.txt", basename);
            occ::log::info("Writing crystalgrower structure file to '{}'",
                           cg_structure_filename);
            occ::io::crystalgrower::StructureWriter cg_structure_writer(
                cg_structure_filename);
            cg_structure_writer.write(c_symm, uc_dimers);
        }

        std::vector<double> solution_terms_uc(uc_neighbors.size());
        // map interactions surrounding UC molecules to symmetry unique
        // interactions
        for (size_t i = 0; i < uc_neighbors.size(); i++) {
            const auto &m = c_symm.unit_cell_molecules()[i];
            size_t asym_idx = m.asymmetric_molecule_idx();
            solution_terms_uc[i] = solution_terms[asym_idx];
            const auto &m_asym = c_symm.symmetry_unique_molecules()[asym_idx];
            auto &n = uc_neighbors[i];
            occ::log::debug("Molecule {} has {} neighbours within {:.3f}", i,
                            n.size(), cg_radius);
            occ::log::debug("Unit cell index = {}, asymmetric index = {}", i,
                            asym_idx);
            int s_int = m.asymmetric_unit_symop()(0);

            SymmetryOperation symop(s_int);

            const auto &rotation = symop.rotation();
            occ::log::debug(
                "Asymmetric unit symop: {} (has handedness change: {})",
                symop.to_string(), rotation.determinant() < 0);

            occ::log::debug("Neighbors for unit cell molecule {} ({})", i,
                            n.size());

            occ::log::debug("{:<7s} {:>7s} {:>10s} {:>7s} {:>7s}", "N", "b",
                            "Tb", "E_int", "R");

            size_t j = 0;
            const auto &n_asym = mol_neighbors[asym_idx];
            const auto &interaction_energies =
                interaction_energies_vec[asym_idx];

            for (auto &dimer : n) {

                auto shift_b = dimer.b().cell_shift();
                auto idx_b = dimer.b().unit_cell_molecule_idx();

                size_t idx{0};
                bool match_type{false};
                for (idx = 0; idx < n_asym.size(); idx++) {
                    if (dimer.equivalent(n_asym[idx])) {
                        break;
                    }
                    if (dimer.equivalent_under_rotation(n_asym[idx],
                                                        rotation)) {
                        match_type = true;
                        break;
                    }
                }
                if (idx >= n_asym.size()) {
                    throw std::runtime_error(
                        fmt::format("No matching interaction found for uc_mol "
                                    "= {}, dimer = {}\n",
                                    i, j));
                }
                double rn = dimer.nearest_distance();
                double rc = dimer.centroid_distance();

                double e_int =
                    interaction_energies[idx] * occ::units::KJ_TO_KCAL;

                dimer.set_interaction_energy(e_int);
                occ::log::debug(
                    "{:<7d} {:>7d} {:>10s} {:7.2f} {:7.3f} {}", j, idx_b,
                    fmt::format("{},{},{}", shift_b[0], shift_b[1], shift_b[2]),
                    e_int, rc, match_type);
                j++;
            }
        }

        {
            std::string kmcpp_structure_filename =
                fmt::format("{}_kmcpp.json", basename);
            occ::log::info("Writing kmcpp structure file to '{}'",
                           kmcpp_structure_filename);
            occ::io::kmcpp::InputWriter kmcpp_structure_writer(
                kmcpp_structure_filename);
            kmcpp_structure_writer.write(c_symm, uc_dimers, solution_terms_uc);
        }

        // write CG net file
        {
            std::string cg_net_filename = fmt::format("{}_net.txt", basename);
            occ::log::info("Writing crystalgrower net file to '{}'",
                           cg_net_filename);
            occ::io::crystalgrower::NetWriter cg_net_writer(cg_net_filename);
            cg_net_writer.write(c_symm, uc_dimers);
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
