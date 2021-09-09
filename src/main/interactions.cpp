#include <cxxopts.hpp>
#include <filesystem>
#include <fmt/os.h>
#include <occ/core/kabsch.h>
#include <occ/core/logger.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/crystal/crystal.h>
#include <occ/dft/dft.h>
#include <occ/interaction/disp.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/polarization.h>
#include <occ/io/cifparser.h>
#include <occ/io/fchkreader.h>
#include <occ/io/fchkwriter.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>
#include <occ/qm/wavefunction.h>
#include <occ/solvent/solvation_correction.h>

namespace fs = std::filesystem;
using occ::chem::Dimer;
using occ::chem::Element;
using occ::chem::Molecule;
using occ::crystal::Crystal;
using occ::crystal::SymmetryOperation;
using occ::hf::HartreeFock;
using occ::interaction::CEModelInteraction;
using occ::qm::BasisSet;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
using occ::scf::SCF;
using occ::solvent::COSMO;
using occ::units::AU_TO_KCAL_PER_MOL;
using occ::units::AU_TO_KJ_PER_MOL;
using occ::units::BOHR_TO_ANGSTROM;
using occ::util::all_close;

struct SolvatedSurfaceProperties {
    double esolv{0.0};
    double dg_ele{0.0};
    double dg_conc{0.0};
    occ::Mat3N coulomb_pos;
    occ::Mat3N cds_pos;
    occ::Vec e_coulomb;
    occ::Vec e_cds;
    occ::Vec e_ele;
    occ::Vec e_conc;
};

bool dimers_equivalent_in_opposite_frame(const occ::chem::Dimer &d1,
                                         const occ::chem::Dimer &d2) {
    using occ::Mat3N;
    using occ::Vec3;
    using occ::linalg::kabsch_rotation_matrix;
    size_t d1_na = d1.a().size();
    size_t d2_na = d2.a().size();
    size_t d1_nb = d1.b().size();
    size_t d2_nb = d2.b().size();
    if ((d1_na != d2_nb) || (d1_nb != d2_na))
        return false;
    if (d1 != d2)
        return false;

    size_t N = d1_na + d2_na;
    Vec3 Od1 = d1.b().centroid();
    Vec3 Od2 = d2.a().centroid();
    Mat3N posd1(3, N), posd2(3, N);
    // positions d1 (with A <-> B swapped)
    posd1.block(0, 0, 3, d1_nb) = d1.b().positions();
    posd1.block(0, d1_nb, 3, d1_na) = d1.a().positions();
    posd1.colwise() -= Od1;
    // positions d2
    posd2.block(0, 0, 3, d2_na) = d2.a().positions();
    posd2.block(0, d2_na, 3, d1_nb) = d2.b().positions();
    posd2.colwise() -= Od2;

    occ::Mat3 rot = kabsch_rotation_matrix(posd1, posd2);
    Mat3N posd1_rot = rot * posd1;
    return all_close(rot * posd1, posd2, 1e-5, 1e-5);
}

std::string dimer_symop(const occ::chem::Dimer &dimer, const Crystal &crystal) {
    const auto &a = dimer.a();
    const auto &b = dimer.b();
    if (a.asymmetric_molecule_idx() != b.asymmetric_molecule_idx())
        return "-";

    int sa_int = a.asymmetric_unit_symop()(0);
    int sb_int = b.asymmetric_unit_symop()(0);

    SymmetryOperation symop_a(sa_int);
    SymmetryOperation symop_b(sb_int);

    auto symop_ab = symop_b * symop_a.inverted();
    occ::Vec3 c_a =
        symop_ab(crystal.to_fractional(a.positions())).rowwise().mean();
    occ::Vec3 v_ab = crystal.to_fractional(b.centroid()) - c_a;

    symop_ab = symop_ab.translated(v_ab);
    return symop_ab.to_string();
}

Crystal read_crystal(const std::string &filename) {
    occ::io::CifParser parser;
    return parser.parse_crystal(filename).value();
}

std::vector<Wavefunction>
calculate_wavefunctions(const std::string &basename,
                        const std::vector<Molecule> &molecules) {
    const std::string method = "b3lyp";
    const std::string basis_name = "6-31G**";
    std::vector<Wavefunction> wfns;
    size_t index = 0;
    for (const auto &m : molecules) {
        fs::path fchk_path(fmt::format("{}_{}.fchk", basename, index));
        fmt::print("Molecule ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", index,
                   "sym", "x", "y", "z");
        for (const auto &atom : m.atoms()) {
            fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n",
                       Element(atom.atomic_number).symbol(), atom.x, atom.y,
                       atom.z);
        }
        if (fs::exists(fchk_path)) {
            using occ::io::FchkReader;
            FchkReader fchk(fchk_path.string());
            auto wfn = Wavefunction(fchk);
            wfns.push_back(wfn);
        } else {
            BasisSet basis(basis_name, m.atoms());
            basis.set_pure(false);
            fmt::print("Loaded basis set, {} shells, {} basis functions\n",
                       basis.size(), libint2::nbf(basis));
            //            HartreeFock hf(m.atoms(), basis);
            //            SCF<HartreeFock, SpinorbitalKind::Restricted> scf(hf);
            occ::dft::DFT rks(method, basis, m.atoms(),
                              SpinorbitalKind::Restricted);
            SCF<occ::dft::DFT, SpinorbitalKind::Restricted> scf(rks);

            scf.set_charge_multiplicity(0, 1);
            scf.start_incremental_F_threshold = 0.0;
            double e = scf.compute_scf_energy();
            auto wfn = scf.wavefunction();
            occ::io::FchkWriter fchk(fchk_path.string());
            fchk.set_title(fmt::format("{} {}/{} generated by occ-ng",
                                       fchk_path.stem(), method, basis_name));
            fchk.set_method(method);
            fchk.set_basis_name(basis_name);
            wfn.save(fchk);
            fchk.write();
            wfns.push_back(wfn);
        }

        index++;
    }
    return wfns;
}

std::pair<std::vector<SolvatedSurfaceProperties>, std::vector<Wavefunction>>
calculate_solvated_surfaces(const std::string &basename,
                            const std::vector<Wavefunction> &wfns,
                            const std::string &solvent_name) {
    std::vector<SolvatedSurfaceProperties> result;
    using occ::dft::DFT;
    using occ::solvent::SolvationCorrectedProcedure;
    const std::string method = "b3lyp";
    const std::string basis_name = "6-31G**";
    std::vector<Wavefunction> solvated_wfns;
    size_t index = 0;
    for (const auto &wfn : wfns) {
        SolvatedSurfaceProperties props;
        BasisSet basis(basis_name, wfn.atoms);
        double original_energy = wfn.energy.total;
        fmt::print("Total energy (gas) {:.3f}\n", original_energy);
        basis.set_pure(false);
        fmt::print("Loaded basis set, {} shells, {} basis functions\n",
                   basis.size(), libint2::nbf(basis));
        occ::dft::DFT ks(method, basis, wfn.atoms, SpinorbitalKind::Restricted);
        SolvationCorrectedProcedure<DFT> proc_solv(ks, solvent_name);
        SCF<SolvationCorrectedProcedure<DFT>, SpinorbitalKind::Restricted> scf(
            proc_solv);
        scf.start_incremental_F_threshold = 0.0;
        scf.set_charge_multiplicity(0, 1);
        double e = scf.compute_scf_energy();
        solvated_wfns.push_back(scf.wavefunction());
        props.coulomb_pos = proc_solv.surface_positions_coulomb();
        props.cds_pos = proc_solv.surface_positions_cds();
        auto coul_areas = proc_solv.surface_areas_coulomb();
        auto cds_areas = proc_solv.surface_areas_cds();
        props.e_cds = proc_solv.surface_cds_energy_elements();
        auto nuc = proc_solv.surface_nuclear_energy_elements();
        auto elec = proc_solv.surface_electronic_energy_elements(
            SpinorbitalKind::Restricted, scf.D);
        auto pol = proc_solv.surface_polarization_energy_elements();
        props.e_coulomb = nuc + elec + pol;
        occ::log::debug("sum e_nuc {:12.6f}\n", nuc.array().sum());
        occ::log::debug("sum e_ele {:12.6f}\n", elec.array().sum());
        occ::log::debug("sum e_pol {:12.6f}\n", pol.array().sum());
        occ::log::debug("sum e_cds {:12.6f}\n", props.e_cds.array().sum());
        double esolv = nuc.array().sum() + elec.array().sum() +
                       pol.array().sum() + props.e_cds.array().sum();

        double dG_conc = 1.89 / occ::units::KJ_TO_KCAL;
        props.dg_conc = dG_conc / occ::units::AU_TO_KJ_PER_MOL;
        props.dg_ele = e - original_energy - esolv;
        props.esolv = esolv + 1.89 / occ::units::AU_TO_KCAL_PER_MOL;
        occ::log::debug("total e_solv {:12.6f} ({:.3f} kJ/mol)\n", esolv,
                        esolv * occ::units::AU_TO_KJ_PER_MOL);
        props.esolv =
            e - original_energy + 1.89 / occ::units::AU_TO_KCAL_PER_MOL;
        ;
        props.e_ele =
            (props.dg_ele / coul_areas.array().sum()) * coul_areas.array();
        props.e_conc =
            (props.dg_conc / cds_areas.array().sum()) * cds_areas.array();
        result.push_back(props);
    }

    return {result, solvated_wfns};
}

void compute_monomer_energies(std::vector<Wavefunction> &wfns) {
    size_t complete = 0;
    for (auto &wfn : wfns) {
        fmt::print("Calculating unique monomer energies for molecule {}\n",
                   complete, wfns.size());
        std::cout << std::flush;
        HartreeFock hf(wfn.atoms, wfn.basis);
        occ::interaction::compute_ce_model_energies(wfn, hf);
        complete++;
    }
    fmt::print("Finished calculating {} unique monomer energies\n", complete);
}

auto calculate_transform(const Wavefunction &wfn, const Molecule &m,
                         const Crystal &c) {
    int sint = m.asymmetric_unit_symop()(0);
    SymmetryOperation symop(sint);
    occ::Mat3N positions = wfn.positions() * BOHR_TO_ANGSTROM;

    occ::Mat3 rotation =
        c.unit_cell().direct() * symop.rotation() * c.unit_cell().inverse();
    occ::Vec3 translation =
        (m.centroid() - (rotation * positions).rowwise().mean()) /
        BOHR_TO_ANGSTROM;
    return std::make_pair(rotation, translation);
}

void write_xyz_dimer(const std::string &filename, const Dimer &dimer) {
    auto output = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                                 fmt::file::CREATE);
    const auto &pos = dimer.positions();
    const auto &nums = dimer.atomic_numbers();
    output.print("{}\n\n", nums.rows());
    for (size_t i = 0; i < nums.rows(); i++) {
        output.print("{:5s} {:12.5f} {:12.5f} {:12.5f}\n",
                     Element(nums(i)).symbol(), pos(0, i), pos(1, i),
                     pos(2, i));
    }
}

auto calculate_interaction_energy(const Dimer &dimer,
                                  const std::vector<Wavefunction> &wfns_a,
                                  const std::vector<Wavefunction> &wfns_b,
                                  const Crystal &crystal) {
    const std::string model_name = "ce-b3lyp";
    Molecule mol_A = dimer.a();
    Molecule mol_B = dimer.b();
    const auto &wfna = wfns_a[mol_A.asymmetric_molecule_idx()];
    const auto &wfnb = wfns_b[mol_B.asymmetric_molecule_idx()];
    Wavefunction A = wfns_a[mol_A.asymmetric_molecule_idx()];
    Wavefunction B = wfns_b[mol_B.asymmetric_molecule_idx()];
    auto transform_a = calculate_transform(wfna, mol_A, crystal);
    A.apply_transformation(transform_a.first, transform_a.second);

    occ::Mat3N pos_A = mol_A.positions();
    occ::Mat3N pos_A_t = A.positions() * BOHR_TO_ANGSTROM;

    assert(all_close(pos_A, pos_A_t, 1e-5, 1e-5));

    auto transform_b = calculate_transform(wfnb, mol_B, crystal);
    B.apply_transformation(transform_b.first, transform_b.second);

    const auto &pos_B = mol_B.positions();
    const auto pos_B_t = B.positions() * BOHR_TO_ANGSTROM;
    assert(all_close(pos_A, pos_A_t, 1e-5, 1e-5));

    auto model = occ::interaction::ce_model_from_string(model_name);

    CEModelInteraction interaction(model);

    auto interaction_energy = interaction(A, B);
    return interaction_energy;
}

std::pair<occ::IVec, occ::Mat3N>
environment(const std::vector<Dimer> &neighbors) {
    size_t num_atoms = 0;
    for (const auto &n : neighbors) {
        num_atoms += n.b().size();
    }

    occ::IVec mol_idx(num_atoms);
    occ::Mat3N positions(3, num_atoms);
    size_t current_idx = 0;
    size_t i = 0;
    for (const auto &n : neighbors) {
        size_t N = n.b().size();
        mol_idx.block(current_idx, 0, N, 1).array() = i;
        positions.block(0, current_idx, 3, N) =
            n.b().positions() / BOHR_TO_ANGSTROM;
        current_idx += N;
        i++;
    }
    return {mol_idx, positions};
}

struct SolventNeighborContribution {
    double coul_ab{0.0}, cds_ab{0.0}, coul_ba{0.0}, cds_ba{0.0};
    double total_coul() const { return coul_ab + coul_ba; }
    double total_cds() const { return cds_ab + cds_ba; }
    double total() const { return coul_ab + coul_ba + cds_ab + cds_ba; }
};

std::vector<SolventNeighborContribution> compute_solvation_energy_breakdown(
    const Crystal &crystal, const std::string &mol_name,
    const SolvatedSurfaceProperties &surface,
    const std::vector<Dimer> &neighbors, const std::string &solvent) {
    using occ::units::angstroms;
    std::vector<SolventNeighborContribution> energy_contribution(
        neighbors.size());

    occ::Mat3N neigh_pos;
    occ::IVec mol_idx;
    occ::IVec neighbor_idx_coul(surface.coulomb_pos.cols());
    occ::IVec neighbor_idx_cds(surface.cds_pos.cols());
    std::tie(mol_idx, neigh_pos) = environment(neighbors);

    auto cfile =
        fmt::output_file(fmt::format("{}_coulomb.txt", mol_name),
                         fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
    cfile.print("{}\nx y z e neighbor\n", neighbor_idx_coul.rows());
    // coulomb breakdown
    for (size_t i = 0; i < neighbor_idx_coul.rows(); i++) {
        occ::Vec3 x = surface.coulomb_pos.col(i);
        Eigen::Index idx = 0;
        double r =
            (neigh_pos.colwise() - x).colwise().squaredNorm().minCoeff(&idx);
        energy_contribution[mol_idx(idx)].coul_ab +=
            surface.e_coulomb(i) + surface.e_ele(i);
        neighbor_idx_coul(i) = mol_idx(idx);
        cfile.print("{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:5d}\n",
                    angstroms(x(0)), angstroms(x(1)), angstroms(x(2)),
                    surface.e_coulomb(i), mol_idx(idx));
    }

    auto cdsfile =
        fmt::output_file(fmt::format("{}_cds.txt", mol_name),
                         fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);
    cdsfile.print("{}\nx y z e neighbor\n", neighbor_idx_cds.rows());
    // cds breakdowg
    for (size_t i = 0; i < neighbor_idx_cds.rows(); i++) {
        occ::Vec3 x = surface.cds_pos.col(i);
        Eigen::Index idx = 0;
        double r =
            (neigh_pos.colwise() - x).colwise().squaredNorm().minCoeff(&idx);
        energy_contribution[mol_idx(idx)].cds_ab +=
            surface.e_cds(i) + surface.e_conc(i);
        neighbor_idx_cds(i) = mol_idx(idx);
        cdsfile.print("{:12.5f} {:12.5f} {:12.5f} {:12.5f} {:5d}\n",
                      angstroms(x(0)), angstroms(x(1)), angstroms(x(2)),
                      surface.e_cds(i), mol_idx(idx));
    }
    // found A -> B contribution, now find B -> A
    for (int i = 0; i < neighbors.size(); i++) {
        const auto &d1 = neighbors[i];
        for (int j = i + 1; j < neighbors.size(); j++) {
            const auto &d2 = neighbors[j];
            if (dimers_equivalent_in_opposite_frame(d1, d2)) {
                energy_contribution[i].coul_ba = energy_contribution[j].coul_ab;
                energy_contribution[i].cds_ba = energy_contribution[j].cds_ab;
                energy_contribution[j].coul_ba = energy_contribution[i].coul_ab;
                energy_contribution[j].cds_ba = energy_contribution[i].cds_ab;
            }
        }
    }
    return energy_contribution;
}

std::vector<double> assign_interaction_terms_to_nearest_neighbours(
    int i, const std::vector<SolventNeighborContribution> &solv,
    const occ::crystal::CrystalDimers &crystal_dimers,
    const std::vector<CEModelInteraction::EnergyComponents> dimer_energies) {
    std::vector<double> crystal_contributions(solv.size(), 0.0);
    double total_taken{0.0};
    const auto &n = crystal_dimers.molecule_neighbors[i];
    for (size_t k1 = 0; k1 < solv.size(); k1++) {
        if (!(abs(solv[k1].coul_ab) == 0.0 && abs(solv[k1].cds_ab) == 0.0))
            continue;
        auto v = n[k1].v_ab().normalized();
        auto unique_dimer_idx = crystal_dimers.unique_dimer_idx[i][k1];
        total_taken += dimer_energies[unique_dimer_idx].total_kjmol();
        double total_dp = 0.0;
        for (size_t k2 = 0; k2 < solv.size(); k2++) {
            if (abs(solv[k2].coul_ab) == 0.0 && abs(solv[k2].cds_ab) == 0.0)
                continue;
            if (k1 == k2)
                continue;
            auto v2 = n[k2].v_ab().normalized();
            double dp = v.dot(v2);
            if (dp <= 0.0)
                continue;
            total_dp += dp;
        }
        for (size_t k2 = 0; k2 < solv.size(); k2++) {
            if (abs(solv[k2].coul_ab) == 0.0 && abs(solv[k2].cds_ab) == 0.0)
                continue;
            if (k1 == k2)
                continue;
            auto v2 = n[k2].v_ab().normalized();
            double dp = v.dot(v2);
            if (dp <= 0.0)
                continue;
            crystal_contributions[k2] +=
                (dp / total_dp) *
                dimer_energies[unique_dimer_idx].total_kjmol();
        }
    }
    double total_reassigned{0.0};
    for (size_t k1 = 0; k1 < solv.size(); k1++) {
        if (crystal_contributions[k1] == 0.0)
            continue;
        fmt::print("{}: {:.3f}\n", k1, crystal_contributions[k1]);
        total_reassigned += crystal_contributions[k1];
    }
    fmt::print("Total taken from non-nearest neighbors: {:.3f} kJ/mol\n",
               total_taken);
    fmt::print("Total assigned to nearest neighbors: {:.3f} kJ/mol\n",
               total_reassigned);
    return crystal_contributions;
}

int main(int argc, char **argv) {
    cxxopts::Options options(
        "occ-interactions",
        "Interactions of molecules with neighbours in a crystal");
    double radius = 0.0;
    bool dump_visualization_files = false;
    using occ::parallel::nthreads;
    std::string cif_filename{""};
    std::string solvent{"water"};
    std::string wfn_choice{"gas"};
    options.add_options()("h,help", "Print help")(
        "i,input", "Input CIF", cxxopts::value<std::string>(cif_filename))(
        "t,threads", "Number of threads",
        cxxopts::value<int>(nthreads)->default_value("1"))(
        "dump-solvent-file", "Write out solvent file",
        cxxopts::value<bool>(dump_visualization_files))(
        "r,radius", "maximum radius (angstroms) for neighbours",
        cxxopts::value<double>(radius)->default_value("3.8"))(
        "s,solvent", "Solvent name", cxxopts::value<std::string>(solvent))(
        "w,wavefunction-choice", "Choice of wavefunctions",
        cxxopts::value<std::string>(wfn_choice));

    options.parse_positional({"input"});

    occ::log::set_level(occ::log::level::info);
    spdlog::set_level(spdlog::level::info);
    libint2::Shell::do_enforce_unit_normalization(false);
    libint2::initialize();

    occ::timing::StopWatch global_timer;
    global_timer.start();

    try {
        options.parse(argc, argv);
    } catch (const std::runtime_error &err) {
        occ::log::error("error when parsing command line arguments: {}",
                        err.what());
        fmt::print("{}", options.help());
        exit(1);
    }

    fmt::print("Parallelized over {} threads & {} Eigen threads\n", nthreads,
               Eigen::nbThreads());

    const std::string error_format =
        "Exception:\n    {}\nTerminating program.\n";
    try {
        occ::timing::StopWatch sw;
        sw.start();
        std::string basename = fs::path(cif_filename).stem();
        Crystal c = read_crystal(cif_filename);
        sw.stop();
        double tprev = sw.read();
        fmt::print("Loaded crystal from {} in {:.6f} seconds\n", cif_filename,
                   tprev);

        sw.start();
        auto molecules = c.symmetry_unique_molecules();
        sw.stop();

        fmt::print(
            "Found {} symmetry unique molecules in {} in {:.6f} seconds\n",
            molecules.size(), cif_filename, sw.read() - tprev);

        tprev = sw.read();
        sw.start();
        auto wfns = calculate_wavefunctions(basename, molecules);
        sw.stop();

        fmt::print("Gas phase wavefunctions took {:.6f} seconds\n",
                   sw.read() - tprev);

        tprev = sw.read();
        sw.start();
        std::vector<Wavefunction> solvated_wavefunctions;
        std::vector<SolvatedSurfaceProperties> surfaces;
        std::tie(surfaces, solvated_wavefunctions) =
            calculate_solvated_surfaces(basename, wfns, solvent);
        sw.stop();
        fmt::print("Solution phase wavefunctions took {:.6f} seconds\n",
                   sw.read() - tprev);

        tprev = sw.read();
        sw.start();
        compute_monomer_energies(wfns);
        compute_monomer_energies(solvated_wavefunctions);
        sw.stop();
        fmt::print("Computing monomer energies took {:.6f} seconds\n",
                   sw.read() - tprev);

        tprev = sw.read();
        sw.start();
        auto crystal_dimers = c.symmetry_unique_dimers(radius);
        sw.stop();

        const auto &dimers = crystal_dimers.unique_dimers;
        fmt::print("Found {} symmetry unique dimers in {:.6f} seconds\n",
                   dimers.size(), sw.read() - tprev);

        if (dimers.size() < 1) {
            fmt::print("No dimers found using neighbour radius {:.3f}\n",
                       radius);
            exit(0);
        }

        const std::string row_fmt_string =
            "{:>7.2f} {:>7.2f} {:>20s} {: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f} {: "
            "7.2f} {: 7.2f} {: 7.2f} {: 7.2f} {: 7.2f}\n";

        std::vector<CEModelInteraction::EnergyComponents> dimer_energies;

        fmt::print(
            "Calculating unique pair interactions using {} wavefunctions\n",
            wfn_choice);
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

        size_t current_dimer = 0;
        for (const auto &dimer : dimers) {
            auto s_ab = dimer_symop(dimer, c);
            write_xyz_dimer(
                fmt::format("{}_dimer_{}.xyz", basename, dimer_energies.size()),
                dimer);
            tprev = sw.read();
            sw.start();
            fmt::print("{}. ({}<->{}) r = {: 7.3f} symop: {}", current_dimer++,
                       dimer.a().unit_cell_molecule_idx(),
                       dimer.b().unit_cell_molecule_idx(),
                       dimer.nearest_distance(),
                       dimer.center_of_mass_distance(), s_ab);

            std::cout << std::flush;
            dimer_energies.push_back(
                calculate_interaction_energy(dimer, wfns_a, wfns_b, c));
            sw.stop();
            fmt::print("  took {:.3f} seconds\n", sw.read() - tprev);
            std::cout << std::flush;
        }
        fmt::print(
            "Finished calculating {} unique dimer interaction energies\n",
            dimer_energies.size());

        const auto &mol_neighbors = crystal_dimers.molecule_neighbors;
        for (size_t i = 0; i < mol_neighbors.size(); i++) {
            const auto &n = mol_neighbors[i];
            std::optional<std::string> solv_filename{};
            std::string molname = fmt::format("{}_{}_{}", basename, i, solvent);
            if (dump_visualization_files) {
                solv_filename =
                    fmt::format("{}_{}_solvation_vis.xyz", basename, i);
            }
            auto solv = compute_solvation_energy_breakdown(
                c, molname, surfaces[i], n, solvent);
            auto crystal_contributions =
                assign_interaction_terms_to_nearest_neighbours(
                    i, solv, crystal_dimers, dimer_energies);
            double Gr = molecules[i].rotational_free_energy(298);
            double Gt = molecules[i].translational_free_energy(298);

            fmt::print("Neighbors for molecule {}\n", i);

            fmt::print("{:>7s} {:>7s} {:>20s} {:>7s} {:>7s} {:>7s} {:>7s} "
                       "{:>7s} {:>7s} {:>7s} {:>7s} {:>7s}\n",
                       "Rn", "Rc", "Symop", "E_coul", "E_rep", "E_pol",
                       "E_disp", "E_tot", "E_scoul", "E_scds", "E_nn", "E_int");
            fmt::print("======================================================="
                       "========="
                       "============================================\n");

            size_t j = 0;
            CEModelInteraction::EnergyComponents total;

            auto neigh = fmt::output_file(
                fmt::format("{}_{}_neighbors.xyz", basename, i),
                fmt::file::WRONLY | O_TRUNC | fmt::file::CREATE);

            size_t natom = std::accumulate(n.begin(), n.end(), 0,
                                           [](size_t a, const auto &dimer) {
                                               return a + dimer.b().size();
                                           });

            neigh.print("{}\nel x y z idx\n", natom);

            for (const auto &dimer : n) {
                auto pos = dimer.b().positions();
                auto els = dimer.b().elements();
                for (size_t a = 0; a < dimer.b().size(); a++) {
                    neigh.print("{:.3s} {:12.5f} {:12.5f} {:12.5f} {:5d}\n",
                                els[a].symbol(), pos(0, a), pos(1, a),
                                pos(2, a), j);
                }

                auto shift_a = dimer.a().cell_shift();
                auto idx_a = dimer.a().unit_cell_molecule_idx();
                auto shift_b = dimer.b().cell_shift();
                auto idx_b = dimer.b().unit_cell_molecule_idx();
                fmt::print("{}: ({} {} {}) {}: ({} {} {})\n", idx_a, shift_a[0],
                           shift_a[1], shift_a[2], idx_b, shift_b[0],
                           shift_b[1], shift_b[2]);

                auto s_ab = dimer_symop(dimer, c);
                size_t idx = crystal_dimers.unique_dimer_idx[i][j];
                double rn = dimer.nearest_distance();
                double rc = dimer.center_of_mass_distance();
                const auto &e =
                    dimer_energies[crystal_dimers.unique_dimer_idx[i][j]];
                double ecoul = e.coulomb_kjmol(), erep = e.exchange_kjmol(),
                       epol = e.polarization_kjmol(),
                       edisp = e.dispersion_kjmol(), etot = e.total_kjmol();
                total.coulomb += ecoul;
                total.exchange_repulsion += erep;
                total.polarization += epol;
                total.dispersion += edisp;
                total.total += etot;

                double e_int = etot + crystal_contributions[j] -
                               (solv[j].total()) * AU_TO_KJ_PER_MOL;
                fmt::print(
                    row_fmt_string, rn, rc, s_ab, ecoul, erep, epol, edisp,
                    etot, (solv[j].total_coul()) * AU_TO_KJ_PER_MOL,
                    (solv[j].total_cds()) * AU_TO_KJ_PER_MOL,
                    crystal_contributions[j], e_int * occ::units::KJ_TO_KCAL);
                j++;
            }
            fmt::print("\nFree energy estimates at T = 298 K, P = 1 atm., "
                       "units: kJ/mol\n");
            fmt::print(
                "-------------------------------------------------------\n");
            fmt::print(
                "lattice energy (crystal)             {: 9.3f}  (E_lat)\n",
                0.5 * total.total);
            fmt::print(
                "rotational free energy (molecule)    {: 9.3f}  (E_rot)\n", Gr);
            fmt::print(
                "translational free energy (molecule) {: 9.3f}  (E_trans)\n",
                Gt);
            fmt::print(
                "solvation free energy (molecule)     {: 9.3f}  (E_solv)\n",
                surfaces[i].esolv * occ::units::AU_TO_KJ_PER_MOL);
            fmt::print("E_solv - E_lat:                      {: 9.3f}\n",
                       surfaces[i].esolv * occ::units::AU_TO_KJ_PER_MOL -
                           0.5 * total.total);
            double dG_solubility =
                surfaces[i].esolv * occ::units::AU_TO_KJ_PER_MOL -
                0.5 * total.total + Gr + Gt;
            fmt::print("solubility                           {: 9.3f}\n",
                       dG_solubility);
            constexpr double R = 8.31446261815324;
            constexpr double RT = 298 * R / 1000;
            double equilibrium_constant = std::exp(-dG_solubility / RT);
            fmt::print("equilibrium_constant                 {: 9.2e}\n",
                       equilibrium_constant);
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
        fmt::print("Exception:\n- Unknown...\n");
        return 1;
    }

    global_timer.stop();
    fmt::print("Program exiting successfully after {:.6f} seconds\n",
               global_timer.read());

    return 0;
}
