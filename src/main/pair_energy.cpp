#include <filesystem>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <nlohmann/json.hpp>
#include <occ/core/logger.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/main/pair_energy.h>
#include <optional>
#include <scn/scn.h>

namespace fs = std::filesystem;

namespace occ::interaction {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CEEnergyComponents, coulomb,
                                   exchange_repulsion, polarization, dispersion,
                                   total)
}

namespace occ::main {

using occ::core::Dimer;
using occ::core::Molecule;
using occ::crystal::Crystal;
using occ::interaction::CEEnergyComponents;
using occ::interaction::CEModelInteraction;
using occ::interaction::CEParameterizedModel;
using occ::qm::Wavefunction;
using occ::units::BOHR_TO_ANGSTROM;

Wavefunction load_wavefunction(const std::string &filename) {
    namespace fs = std::filesystem;
    using occ::util::to_lower;
    std::string ext = fs::path(filename).extension().string();
    to_lower(ext);
    if (ext == ".fchk") {
        using occ::io::FchkReader;
        FchkReader fchk(filename);
        return Wavefunction(fchk);
    }
    if (ext == ".molden" || ext == ".input") {
        using occ::io::MoldenReader;
        MoldenReader molden(filename);
        return Wavefunction(molden);
    }
    throw std::runtime_error(
        "Unknown file extension when reading wavefunction: " + ext);
}

PairEnergy::PairEnergy(const occ::io::OccInput &input) {
    if (input.pair.source_a == "none" || input.pair.source_b == "none") {
        throw "Both monomers in a pair energy need a wavefunction set";
    }

    a.wfn = load_wavefunction(input.pair.source_a);
    b.wfn = load_wavefunction(input.pair.source_b);
    a.rotation = input.pair.rotation_a;
    b.rotation = input.pair.rotation_b;
    a.translation = input.pair.translation_a * occ::units::ANGSTROM_TO_BOHR;
    b.translation = input.pair.translation_b * occ::units::ANGSTROM_TO_BOHR;

    a.wfn.apply_transformation(a.rotation, a.translation);
    b.wfn.apply_transformation(b.rotation, b.translation);
}

void PairEnergy::compute() {

    CEModelInteraction interaction(model);
    auto interaction_energy = interaction(a.wfn, b.wfn);
    occ::timing::stop(occ::timing::category::global);

    fmt::print("Monomer A energies\n");
    a.wfn.energy.print();

    fmt::print("Monomer B energies\n");
    b.wfn.energy.print();

    fmt::print("\nDimer\n");

    fmt::print("Component              Energy (kJ/mol)\n\n");
    fmt::print("Coulomb               {: 12.6f}\n",
               interaction_energy.coulomb_kjmol());
    fmt::print("Exchange-repulsion    {: 12.6f}\n",
               interaction_energy.exchange_kjmol());
    fmt::print("Polarization          {: 12.6f}\n",
               interaction_energy.polarization_kjmol());
    fmt::print("Dispersion            {: 12.6f}\n",
               interaction_energy.dispersion_kjmol());
    fmt::print("__________________________________\n");
    fmt::print("Total 		      {: 12.6f}\n",
               interaction_energy.total_kjmol());
}

auto calculate_transform(const Wavefunction &wfn, const Molecule &m,
                         const Crystal &c) {
    using occ::crystal::SymmetryOperation;
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

bool write_xyz_dimer(const std::string &filename, const Dimer &dimer,
                     std::optional<CEEnergyComponents> energies) {

    using occ::core::Element;
    auto output = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                                 fmt::file::CREATE);
    const auto &pos = dimer.positions();
    const auto &nums = dimer.atomic_numbers();
    output.print("{}\n", nums.rows());
    if (energies) {
        nlohmann::json j = *energies;
        output.print("{}", j.dump());
    }
    output.print("\n");
    for (size_t i = 0; i < nums.rows(); i++) {
        output.print("{:5s} {:12.5f} {:12.5f} {:12.5f}\n",
                     Element(nums(i)).symbol(), pos(0, i), pos(1, i),
                     pos(2, i));
    }
    return true;
}

CEEnergyComponents ce_model_energy(const Dimer &dimer,
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

    assert(occ::util::all_close(pos_A, pos_A_t, 1e-5, 1e-5));

    auto transform_b = calculate_transform(wfnb, mol_B, crystal);
    B.apply_transformation(transform_b.first, transform_b.second);

    const auto &pos_B = mol_B.positions();
    const auto pos_B_t = B.positions() * BOHR_TO_ANGSTROM;
    assert(occ::util::all_close(pos_A, pos_A_t, 1e-5, 1e-5));

    auto model = occ::interaction::ce_model_from_string(model_name);

    CEModelInteraction interaction(model);

    auto interaction_energy = interaction(A, B);
    interaction_energy.is_computed = true;
    return interaction_energy;
}

int compute_ce_model_energies_radius(
    const Crystal &crystal, const std::vector<Dimer> &dimers,
    const std::vector<Wavefunction> &wfns_a,
    const std::vector<Wavefunction> &wfns_b, const std::string &basename,
    double radius, std::vector<CEEnergyComponents> &dimer_energies) {
    using occ::crystal::SymmetryOperation;
    occ::timing::StopWatch sw;
    if (dimer_energies.size() < dimers.size()) {
        dimer_energies.resize(dimers.size());
    }
    size_t current_dimer{0};
    size_t computed_dimers{0};
    for (const auto &dimer : dimers) {
        auto tprev = sw.read();
        sw.start();

        CEEnergyComponents &dimer_energy = dimer_energies[current_dimer];
        std::string dimer_energy_file(
            fmt::format("{}_dimer_{}_energies.xyz", basename, current_dimer));

        if (dimer.nearest_distance() > radius || dimer_energy.is_computed ||
            load_dimer_energy(dimer_energy_file, dimer_energy)) {
            current_dimer++;
            continue;
        }
        computed_dimers++;

        const auto &a = dimer.a();
        const auto &b = dimer.b();
        const auto asym_idx_a = a.asymmetric_molecule_idx();
        const auto asym_idx_b = b.asymmetric_molecule_idx();
        const auto shift_b = dimer.b().cell_shift();
        occ::log::info(
            "{} ({}[{}] - {}[{} + ({},{},{})]), Rc = {: 5.2f}", current_dimer++,
            asym_idx_a,
            SymmetryOperation(a.asymmetric_unit_symop()(0)).to_string(),
            asym_idx_b,
            SymmetryOperation(b.asymmetric_unit_symop()(0)).to_string(),
            shift_b[0], shift_b[1], shift_b[2],
            dimer.center_of_mass_distance());

        std::cout << std::flush;
        dimer_energy = ce_model_energy(dimer, wfns_a, wfns_b, crystal);
        sw.stop();
        occ::log::info("Took {:.3f} seconds", sw.read() - tprev);
        write_xyz_dimer(dimer_energy_file, dimer, dimer_energy);
        std::cout << std::flush;
    }
    occ::log::info("Finished calculating {} unique dimer interaction energies",
                   computed_dimers);
    return computed_dimers;
}

std::vector<CEEnergyComponents>
ce_model_energies(const Crystal &crystal, const std::vector<Dimer> &dimers,
                  const std::vector<Wavefunction> &wfns_a,
                  const std::vector<Wavefunction> &wfns_b,
                  const std::string &basename) {
    std::vector<CEEnergyComponents> result;
    compute_ce_model_energies_radius(crystal, dimers, wfns_a, wfns_b, basename,
                                     std::numeric_limits<double>::max(),
                                     result);
    return result;
}

bool load_dimer_energy(const std::string &filename,
                       CEEnergyComponents &energies) {
    if (!fs::exists(filename))
        return false;
    occ::log::info("Load dimer energies from {}", filename);
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line);
    std::getline(file, line);
    energies = nlohmann::json::parse(line).get<CEEnergyComponents>();
    energies.is_computed = true;
    return true;
}

std::pair<occ::crystal::CrystalDimers, std::vector<CEEnergyComponents>>
converged_lattice_energies(const Crystal &crystal,
                           const std::vector<Wavefunction> &wfns_a,
                           const std::vector<Wavefunction> &wfns_b,
                           const std::string &basename,
                           const LatticeConvergenceSettings conv) {

    crystal::CrystalDimers converged_dimers;
    std::vector<CEEnergyComponents> converged_energies;
    double lattice_energy{0.0}, previous_lattice_energy{0.0};
    double current_radius = std::max(conv.radius_increment, conv.min_radius);
    size_t cycle{1};
    auto all_dimers = crystal.symmetry_unique_dimers(conv.max_radius);
    fmt::print("Found {} symmetry unique dimers within max radius {:.3f}\n",
               all_dimers.unique_dimers.size(), conv.max_radius);
    fmt::print("Lattice convergence settings:\n");
    fmt::print("Start radius       {: 8.4f} \u212b\n", conv.min_radius);
    fmt::print("Max radius         {: 8.4f} \u212b\n", conv.max_radius);
    fmt::print("Radius increment   {: 8.4f} \u212b\n", conv.radius_increment);
    fmt::print("Energy tolerance   {: 8.4f} kJ/mol\n", conv.energy_tolerance);
    do {
        previous_lattice_energy = lattice_energy;
        const auto &dimers = all_dimers.unique_dimers;
        compute_ce_model_energies_radius(crystal, dimers, wfns_a, wfns_b,
                                         basename, current_radius,
                                         converged_energies);
        const auto &mol_neighbors = all_dimers.molecule_neighbors;
        double etot{0.0};
        size_t mol_idx{0};
        for (const auto &n : mol_neighbors) {
            double molecule_total{0.0};
            size_t dimer_idx{0};
            for (const auto &dimer : n) {
                const auto &e =
                    converged_energies[all_dimers.unique_dimer_idx[mol_idx]
                                                                  [dimer_idx]];

                if (e.is_computed) {
                    molecule_total += e.total_kjmol();
                }
                dimer_idx++;
            }
            etot += molecule_total;
            mol_idx++;
        }
        lattice_energy = 0.5 * etot;
        fmt::print("Cycle {} lattice energy: {}\n", cycle, lattice_energy);
        cycle++;
        current_radius += conv.radius_increment;
    } while (std::abs(lattice_energy - previous_lattice_energy) >
             conv.energy_tolerance);
    converged_dimers = all_dimers;
    return std::make_pair(converged_dimers, converged_energies);
}

} // namespace occ::main
