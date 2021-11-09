#include <occ/main/pair_energy.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/core/molecule.h>
#include <optional>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <filesystem>
#include <scn/scn.h>

namespace fs = std::filesystem;

namespace occ::main {

using occ::interaction::CEModelInteraction;
using EnergyComponentsCE = CEModelInteraction::EnergyComponents;
using occ::chem::Dimer;
using occ::units::BOHR_TO_ANGSTROM;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;
using occ::chem::Molecule;
using occ::crystal::Crystal;

EnergyComponentsCE read_energy_components(const std::string &line) {
    CEModelInteraction::EnergyComponents components;
    scn::scan(line, "{{ e_coul: {}, e_rep: {}, e_pol: {}, e_disp: {}, e_tot: {} }}",
              components.coulomb, components.exchange_repulsion,
              components.polarization, components.dispersion, components.total);
    return components;
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
                     std::optional<EnergyComponentsCE> energies) {

    using occ::chem::Element;
    auto output = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                                 fmt::file::CREATE);
    const auto &pos = dimer.positions();
    const auto &nums = dimer.atomic_numbers();
    output.print("{}\n", nums.rows());
    if(energies) {
        auto e = *energies;
        output.print("{{ e_coul: {}, e_rep: {}, e_pol: {}, e_disp: {}, e_tot: {} }}",
                     e.coulomb, e.exchange_repulsion, e.polarization, e.dispersion, e.total);
    }
    output.print("\n");
    for (size_t i = 0; i < nums.rows(); i++) {
        output.print("{:5s} {:12.5f} {:12.5f} {:12.5f}\n",
                     Element(nums(i)).symbol(), pos(0, i), pos(1, i),
                     pos(2, i));
    }
    return true;
}

EnergyComponentsCE ce_model_energy(const Dimer &dimer,
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


std::vector<EnergyComponentsCE> ce_model_energies(
                              const Crystal &crystal, 
                              const std::vector<Dimer> &dimers,
                              const std::vector<Wavefunction> &wfns_a,
                              const std::vector<Wavefunction> &wfns_b,
                              const std::string &basename) {
    using occ::crystal::SymmetryOperation;
    occ::timing::StopWatch sw;
    std::vector<EnergyComponentsCE> dimer_energies;
    dimer_energies.reserve(dimers.size());
    size_t current_dimer{0};
    for (const auto &dimer : dimers) {
        auto tprev = sw.read();
        sw.start();
        EnergyComponentsCE dimer_energy;
        std::string dimer_energy_file(fmt::format("{}_dimer_{}_energies.xyz", basename, current_dimer));

        if(load_dimer_energy(dimer_energy_file, dimer_energy)) {
            dimer_energies.push_back(dimer_energy);
            current_dimer++;
            continue;
        }

        const auto& a = dimer.a();
        const auto& b = dimer.b();
        const auto asym_idx_a = a.asymmetric_molecule_idx();
        const auto asym_idx_b = b.asymmetric_molecule_idx();
        const auto shift_b = dimer.b().cell_shift();
        fmt::print("{} ({}[{}] - {}[{} + ({},{},{})]), Rc = {: 5.2f}",
                   current_dimer++,
                   asym_idx_a,
                   SymmetryOperation(a.asymmetric_unit_symop()(0)).to_string(),
                   asym_idx_b,
                   SymmetryOperation(b.asymmetric_unit_symop()(0)).to_string(),
                   shift_b[0], shift_b[1], shift_b[2],
                   dimer.center_of_mass_distance());

        std::cout << std::flush;
        dimer_energy = ce_model_energy(dimer, wfns_a, wfns_b, crystal);
        dimer_energies.push_back(dimer_energy);
        sw.stop();
        fmt::print("  took {:.3f} seconds\n", sw.read() - tprev);
        write_xyz_dimer(dimer_energy_file, dimer, dimer_energies[current_dimer - 1]);
        std::cout << std::flush;
    }
    fmt::print(
        "Finished calculating {} unique dimer interaction energies\n",
        dimer_energies.size());
    return dimer_energies;
}


bool load_dimer_energy(const std::string &filename, EnergyComponentsCE &energies) {
    if(!fs::exists(filename))
        return false;
    fmt::print("Load dimer energies from {}\n", filename);
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line);
    std::getline(file, line);
    energies = read_energy_components(line);
    return true;
}



}
