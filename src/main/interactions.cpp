#include <tonto/crystal/crystal.h>
#include <tonto/io/cifparser.h>
#include <tonto/3rdparty/argparse.hpp>
#include <tonto/qm/wavefunction.h>
#include <tonto/core/logger.h>
#include <tonto/qm/hf.h>
#include <tonto/dft/dft.h>
#include <tonto/qm/scf.h>
#include <tonto/io/fchkwriter.h>
#include <tonto/io/fchkreader.h>
#include <tonto/interaction/pairinteraction.h>
#include <tonto/interaction/disp.h>
#include <tonto/interaction/polarization.h>
#include <tonto/core/kabsch.h>
#include <tonto/solvent/cosmo.h>
#include <tonto/solvent/surface.h>
#include <filesystem>
#include <tonto/core/units.h>
#include <fmt/os.h>

namespace fs = std::filesystem;
using tonto::crystal::Crystal;
using tonto::crystal::SymmetryOperation;
using tonto::chem::Molecule;
using tonto::chem::Dimer;
using tonto::qm::Wavefunction;
using tonto::qm::SpinorbitalKind;
using tonto::qm::BasisSet;
using tonto::scf::SCF;
using tonto::units::BOHR_TO_ANGSTROM;
using tonto::units::AU_TO_KJ_PER_MOL;
using tonto::units::AU_TO_KCAL_PER_MOL;
using tonto::interaction::CEModelInteraction;
using tonto::chem::Element;
using tonto::util::all_close;
using tonto::hf::HartreeFock;
using tonto::solvent::COSMO;


tonto::Vec compute_esp(const Wavefunction &wfn, const tonto::Mat3N &points)
{
    tonto::ints::shellpair_list_t shellpair_list;
    tonto::ints::shellpair_data_t shellpair_data;
    std::tie(shellpair_list, shellpair_data) = tonto::ints::compute_shellpairs(wfn.basis);
    return tonto::ints::compute_electric_potential(wfn.D, wfn.basis, shellpair_list, points);
}


SymmetryOperation dimer_symop(const tonto::chem::Dimer &dimer, const Crystal &crystal)
{
    const auto& a = dimer.a();
    const auto& b = dimer.b();

    int sa_int = a.asymmetric_unit_symop()(0);
    int sb_int = b.asymmetric_unit_symop()(0);

    SymmetryOperation symop_a(sa_int);
    SymmetryOperation symop_b(sb_int);

    auto symop_ab = symop_b * symop_a.inverted();
    tonto::Vec3 c_a = symop_ab(crystal.to_fractional(a.positions())).rowwise().mean();
    tonto::Vec3 v_ab = crystal.to_fractional(b.centroid()) - c_a;

    symop_ab = symop_ab.translated(v_ab);
    return symop_ab;
}

Crystal read_crystal(const std::string &filename)
{
    tonto::io::CifParser parser;
    return parser.parse_crystal(filename).value();
}

std::vector<Wavefunction> calculate_wavefunctions(const std::string &basename, const std::vector<Molecule> &molecules)
{
    const std::string method = "b3lyp";
    const std::string basis_name = "6-31G**";
    std::vector<Wavefunction> wfns;
    size_t index = 0;
    for(const auto& m: molecules)
    {
        fs::path fchk_path(fmt::format("{}_{}.fchk", basename, index));
        auto dmat = fmt::output_file(fmt::format("{}_{}.txt", basename, index));
        fmt::print("Molecule ({})\n{:3s} {:^10s} {:^10s} {:^10s}\n", index, "sym", "x", "y", "z");
        for (const auto &atom : m.atoms()) {
            fmt::print("{:^3s} {:10.6f} {:10.6f} {:10.6f}\n", Element(atom.atomic_number).symbol(),
                       atom.x, atom.y, atom.z);
        }
        if(fs::exists(fchk_path)) {
            using tonto::io::FchkReader;
            FchkReader fchk(fchk_path.string());
            auto wfn = Wavefunction(fchk);
            dmat.print("{}", wfn.D);
            wfns.push_back(wfn);
        }
        else {
            BasisSet basis(basis_name, m.atoms());
            basis.set_pure(false);
            fmt::print("Loaded basis set, {} shells, {} basis functions\n", basis.size(), libint2::nbf(basis));
//            HartreeFock hf(m.atoms(), basis);
//            SCF<HartreeFock, SpinorbitalKind::Restricted> scf(hf);
            tonto::dft::DFT rks(method, basis, m.atoms(), SpinorbitalKind::Restricted);
            SCF<tonto::dft::DFT, SpinorbitalKind::Restricted> scf(rks);

            scf.set_charge_multiplicity(0, 1);
            scf.start_incremental_F_threshold = 0.0;
            double e = scf.compute_scf_energy();
            auto wfn = scf.wavefunction();
            dmat.print("{}", wfn.D);
            tonto::io::FchkWriter fchk(fchk_path.string());
            fchk.set_title(fmt::format("{} {}/{} generated by tonto-ng", fchk_path.stem(), method, basis_name));
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

auto compute_solvent_surface(const Wavefunction &wfn)
{
    tonto::Mat3N pos = wfn.positions();
    tonto::IVec nums = wfn.atomic_numbers();
    tonto::Vec radii = tonto::solvent::cosmo::solvation_radii(nums);
    radii.array() /= BOHR_TO_ANGSTROM;

    auto surface = tonto::solvent::surface::solvent_surface(radii, nums, pos);
    //fmt::print("Surface ({} atoms, {} points) calculated in {}\n", radii.rows(), surface.areas.rows(), sw.read(0));
    return surface;
}

std::vector<tonto::solvent::surface::Surface> compute_solvent_surfaces(const std::vector<Wavefunction> &wfns)
{
    std::vector<tonto::solvent::surface::Surface> surfs;
    for(const auto& wfn: wfns)
    {
        surfs.push_back(compute_solvent_surface(wfn));
    }
    return surfs;
}

void compute_monomer_energies(std::vector<Wavefunction> &wfns)
{
    fmt::print("Computing monomer energies\n");
    for(auto& wfn : wfns)
    {
        HartreeFock hf(wfn.atoms, wfn.basis);
        tonto::interaction::compute_ce_model_energies(wfn, hf);
    }
}

auto calculate_transform(const Wavefunction &wfn, const Molecule &m, const Crystal &c)
{
    int sint = m.asymmetric_unit_symop()(0);
    SymmetryOperation symop(sint);
    tonto::Mat3N positions = wfn.positions() * BOHR_TO_ANGSTROM;

    tonto::Mat3 rotation = c.unit_cell().direct() * symop.rotation() * c.unit_cell().inverse();
    tonto::Vec3 translation = (m.centroid() - (rotation * positions).rowwise().mean()) / BOHR_TO_ANGSTROM;
    return std::make_pair(rotation, translation);
}

void write_xyz_dimer(const std::string &filename, const Dimer &dimer)
{
    auto output = fmt::output_file(filename);
    const auto& pos = dimer.positions();
    const auto& nums = dimer.atomic_numbers();
    output.print("{}\n\n", nums.rows());
    for(size_t i = 0; i < nums.rows(); i++)
    {
        output.print("{} {} {} {}\n", Element(nums(i)).symbol(), pos(0, i), pos(1, i), pos(2, i));
    }
}

auto calculate_interaction_energy(const Dimer &dimer, const std::vector<Wavefunction> &wfns, const Crystal &crystal)
{
    const std::string model_name = "ce-b3lyp";
    Molecule mol_A = dimer.a();
    Molecule mol_B = dimer.b();
    const auto& wfna = wfns[mol_A.asymmetric_molecule_idx()];
    const auto& wfnb = wfns[mol_B.asymmetric_molecule_idx()];
    Wavefunction A = wfns[mol_A.asymmetric_molecule_idx()];
    Wavefunction B = wfns[mol_B.asymmetric_molecule_idx()];
    auto transform_a = calculate_transform(wfna, mol_A, crystal);
    A.apply_transformation(transform_a.first, transform_a.second);

    tonto::Mat3N pos_A = mol_A.positions();
    tonto::Mat3N pos_A_t = A.positions() * BOHR_TO_ANGSTROM;

    assert(all_close(pos_A, pos_A_t, 1e-5, 1e-5));

    auto transform_b = calculate_transform(wfnb, mol_B, crystal);
    B.apply_transformation(transform_b.first, transform_b.second);

    const auto &pos_B = mol_B.positions();
    const auto pos_B_t = B.positions() * BOHR_TO_ANGSTROM;
    assert(all_close(pos_A, pos_A_t, 1e-5, 1e-5));

    auto model = tonto::interaction::ce_model_from_string(model_name);

    CEModelInteraction interaction(model);

    auto interaction_energy = interaction(A, B);
    return interaction_energy;
}

std::pair<tonto::IVec, tonto::Mat3N> environment(const std::vector<Dimer> &neighbors)
{
    size_t num_atoms = 0;
    for(const auto &n: neighbors)
    {
        num_atoms += n.b().size();
    }

    tonto::IVec  mol_idx(num_atoms);
    tonto::Mat3N positions(3, num_atoms);
    size_t current_idx = 0;
    size_t i = 0;
    for(const auto &n: neighbors)
    {
        size_t N = n.b().size();
        mol_idx.block(current_idx, 0, N, 1).array() = i;
        positions.block(0, current_idx, 3, N) = n.b().positions();
        current_idx += N;
        i++;
    }
    return {mol_idx, positions};
}

std::vector<double> compute_solvation_energy_breakdown(
        const tonto::solvent::surface::Surface& surface, const Wavefunction &wfn,
        const std::vector<Dimer> &neighbors)
{
    tonto::Vec areas = surface.areas;
    tonto::Mat3N points = surface.vertices;
    tonto::Mat3N pos = wfn.positions();
    tonto::IVec nums = wfn.atomic_numbers();
    tonto::Vec radii = tonto::solvent::cosmo::solvation_radii(nums);
    tonto::Vec charges = compute_esp(wfn, points);
    for(size_t i = 0; i < nums.rows(); i++)
    {
        auto p1 = pos.col(i);
        double q = nums(i);
        auto r = (points.colwise() - p1).colwise().norm();
        charges.array() += q / r.array();
    }
    //fmt::print("ESP (range = {}, {}) calculated in {}\n", charges.minCoeff(), charges.maxCoeff(), sw.read(0)); 


    COSMO cosmo(78.40);
    cosmo.set_max_iterations(100);
    auto result = cosmo(points, areas, charges);
    //fmt::print("Surface area: {} angstrom**2\n", areas.sum() * 0.52917749 * 0.52917749);
    //fmt::print("Total energy: {} kJ/mol\n", AU_TO_KJ_PER_MOL * result.energy);

    std::vector<double> energy_contribution(neighbors.size());

    tonto::IVec mol_idx;
    tonto::Mat3N neigh_pos;
    std::tie(mol_idx, neigh_pos) = environment(neighbors);
    
    tonto::IVec neighbor_idx(surface.vertices.cols());
    for(size_t i = 0; i < neighbor_idx.rows(); i++)
    {
        tonto::Vec3 x = points.col(i) * BOHR_TO_ANGSTROM;
        Eigen::Index idx = 0;
        double r = (neigh_pos.colwise() - x).colwise().squaredNorm().minCoeff(&idx);
        energy_contribution[mol_idx(idx)] += 0.5 * result.converged(i) * result.initial(i);
    }
    return energy_contribution;
}


int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("interactions");
    parser.add_argument("input").help("Input CIF");
    parser.add_argument("-j", "--threads")
            .help("Number of threads")
            .default_value(2)
            .action([](const std::string& value) { return std::stoi(value); });
    parser.add_argument("--radius")
        .help("Radius (angstroms) for neighbours")
        .default_value(3.8)
        .action([](const std::string& value) { return std::stod(value); });
    tonto::log::set_level(tonto::log::level::info);
    spdlog::set_level(spdlog::level::info);
    libint2::Shell::do_enforce_unit_normalization(false);
    libint2::initialize();
    double radius = 0.0;

    try {
        parser.parse_args(argc, argv);
        radius = parser.get<double>("--radius");
    }
    catch (const std::runtime_error& err) {
        tonto::log::error("error when parsing command line arguments: {}", err.what());
        fmt::print("{}", parser);
        exit(1);
    }


    using tonto::parallel::nthreads;
    nthreads = parser.get<int>("--threads");
    omp_set_num_threads(nthreads);
    fmt::print("Parallelized over {} OpenMP threads & {} Eigen threads\n", nthreads, Eigen::nbThreads());


    const std::string error_format = "Exception:\n    {}\nTerminating program.\n";
    try {
        std::string filename = parser.get<std::string>("input");
        std::string basename = fs::path(filename).stem();
        Crystal c = read_crystal(filename);
        fmt::print("Loaded crystal from {}\n", filename);
        auto molecules = c.symmetry_unique_molecules();
        fmt::print("{} molecules\n", molecules.size());
        auto wfns = calculate_wavefunctions(basename, molecules);
        auto surfaces = compute_solvent_surfaces(wfns);
        compute_monomer_energies(wfns);
        auto crystal_dimers = c.symmetry_unique_dimers(radius);
        const auto &dimers = crystal_dimers.unique_dimers;
        const std::string row_fmt_string = "{:>9.3f} {:>24s} {: 9.3f} {: 9.3f} {: 9.3f} {: 9.3f} | {: 9.3f} | {: 9.3f} | {: 9.3f}\n";

        std::vector<CEModelInteraction::EnergyComponents> dimer_energies;

        for(const auto& dimer: dimers)
        {
            auto s_ab = dimer_symop(dimer, c);
            write_xyz_dimer(fmt::format("{}_dimer_{}.xyz", basename, dimer_energies.size()), dimer);
            fmt::print("Calculating dimer energies {}/{}\r", dimer_energies.size(), dimers.size());
            std::cout << std::flush;
            dimer_energies.push_back(calculate_interaction_energy(dimer, wfns, c));
        }
        fmt::print("Complete {}/{}\n", dimer_energies.size(), dimers.size());

        const auto &mol_neighbors = crystal_dimers.molecule_neighbors;
        for(size_t i = 0; i < mol_neighbors.size(); i++)
        {
            const auto& n = mol_neighbors[i];
            auto solv = compute_solvation_energy_breakdown(surfaces[i], wfns[i], n);

            fmt::print("Neighbors for molecule {}\n", i);

            fmt::print("{:>9s} {:>24s} {:>9s} {:>9s} {:>9s} {:>9s} | {:>9s} | {:>9s} | {:>9s}\n",
                       "R", "Symop", "E_coul", "E_rep", "E_pol", "E_disp", "E_tot", "E_solv", "E_int");
            size_t j = 0;
            CEModelInteraction::EnergyComponents total; 

            for(const auto& dimer: n)
            {
                auto s_ab = dimer_symop(dimer, c).to_string();
                size_t idx = crystal_dimers.unique_dimer_idx[i][j]; 
                double r = dimer.center_of_mass_distance();
                const auto& e = dimer_energies[crystal_dimers.unique_dimer_idx[i][j]];
                double ecoul = e.coulomb_kjmol(), erep = e.exchange_kjmol(),
                    epol = e.polarization_kjmol(), edisp = e.dispersion_kjmol(),
                    etot = e.total_kjmol();
                total.coulomb += ecoul;
                total.exchange_repulsion += erep;
                total.polarization += epol;
                total.dispersion += edisp;
                total.total += etot;

                fmt::print(row_fmt_string, r, s_ab, ecoul, erep, epol, edisp, etot, solv[j] * AU_TO_KJ_PER_MOL,
                           etot + solv[j] * AU_TO_KJ_PER_MOL);
                j++;
            }
            fmt::print("Total: {:.3f} kJ/mol\n", total.total);
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
   
    return 0;
}
