#include <filesystem>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <nlohmann/json.hpp>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/interaction/coulomb.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/wolf.h>
#include <occ/main/pair_energy.h>
#include <occ/qm/chelpg.h>
#include <occ/xtb/xtb_wrapper.h>
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

using WavefunctionList = std::vector<Wavefunction>;

class CEPairEnergyFunctor {
  public:
    CEPairEnergyFunctor(const Crystal &crystal,
                        const WavefunctionList &wavefunctions_a,
                        const WavefunctionList &wavefunctions_b = {})
        : m_crystal(crystal), m_wavefunctions_a(wavefunctions_a),
          m_wavefunctions_b(wavefunctions_b) {}

    CEEnergyComponents operator()(const Dimer &dimer) {
        const std::string model_name = "ce-b3lyp";
        Molecule mol_A = dimer.a();
        Molecule mol_B = dimer.b();
        Wavefunction A = m_wavefunctions_a[mol_A.asymmetric_molecule_idx()];
        Wavefunction B =
            (m_wavefunctions_b.size() > 0)
                ? m_wavefunctions_a[mol_B.asymmetric_molecule_idx()]
                : m_wavefunctions_b[mol_B.asymmetric_molecule_idx()];

        auto transform_a = calculate_transform(A, mol_A, m_crystal);
        A.apply_transformation(transform_a.first, transform_a.second);

        occ::Mat3N pos_A = mol_A.positions();
        occ::Mat3N pos_A_t = A.positions() * BOHR_TO_ANGSTROM;

        auto transform_b = calculate_transform(B, mol_B, m_crystal);
        B.apply_transformation(transform_b.first, transform_b.second);

        Mat3N pos_B = mol_B.positions();
        Mat3N pos_B_t = B.positions() * BOHR_TO_ANGSTROM;

        auto model = occ::interaction::ce_model_from_string(model_name);

        CEModelInteraction interaction(model);

        auto interaction_energy = interaction(A, B);
        interaction_energy.is_computed = true;
        fmt::print("Finished model energy\n");
        return interaction_energy;
    }

    inline const auto &partial_charges() {
        if (m_partial_charges.size() > 0)
            return m_partial_charges;

        m_partial_charges = std::vector<Vec>(m_wavefunctions_a.size());
        for (int i = 0; i < m_wavefunctions_a.size(); i++) {
            m_partial_charges[i] =
                occ::qm::chelpg_charges(m_wavefunctions_a[i]);
        }
        return m_partial_charges;
    }

  private:
    Crystal m_crystal;
    std::vector<Wavefunction> m_wavefunctions_a;
    std::vector<Wavefunction> m_wavefunctions_b;
    std::vector<Vec> m_partial_charges;
};

class XTBPairEnergyFunctor {
  public:
    XTBPairEnergyFunctor(const Crystal &crystal) : m_crystal(crystal) {
        for (const auto &mol : crystal.symmetry_unique_molecules()) {
            occ::xtb::XTBCalculator calc(mol);
            m_monomer_energies.push_back(calc.single_point_energy());

            m_partial_charges.push_back(calc.partial_charges());
        }
    }

    CEEnergyComponents operator()(const Dimer &dimer) {
        Molecule mol_A = dimer.a();
        Molecule mol_B = dimer.b();

        occ::xtb::XTBCalculator calc_AB(dimer);
        double e_a = m_monomer_energies[mol_A.asymmetric_molecule_idx()];
        double e_b = m_monomer_energies[mol_B.asymmetric_molecule_idx()];
        double e_ab = calc_AB.single_point_energy();
        CEEnergyComponents result;
        result.total = e_ab - e_a - e_b;
        result.is_computed = true;
        return result;
    }

    inline const auto &partial_charges() { return m_partial_charges; }
    inline const auto &monomer_energies() { return m_monomer_energies; }

  private:
    Crystal m_crystal;
    std::vector<double> m_monomer_energies;
    std::vector<Vec> m_partial_charges;
};

int compute_coulomb_energies_radius(const std::vector<Dimer> &dimers,
                                    const Vec &asym_charges, double radius,
                                    std::vector<double> &charge_energies) {
    occ::timing::StopWatch sw;
    if (charge_energies.size() < dimers.size()) {
        charge_energies.resize(dimers.size());
    }

    size_t current_dimer{0};
    size_t computed_dimers{0};
    for (const auto &dimer : dimers) {
        if (dimer.nearest_distance() > radius) {
            current_dimer++;
            continue;
        }
        auto tprev = sw.read();
        sw.start();
        const auto asym_idx_a = dimer.a().asymmetric_molecule_idx();
        const auto asym_idx_b = dimer.b().asymmetric_molecule_idx();

        charge_energies[current_dimer] =
            occ::interaction::coulomb_interaction_energy_asym_charges(
                dimer, asym_charges);
        current_dimer++;
    }
    occ::log::info("Finished calculating {} unique dimer coulomb energies",
                   computed_dimers);
    return computed_dimers;
}

template <typename EnergyModel>
int compute_dimer_energies_radius(
    EnergyModel &energy_model, const Crystal &crystal,
    const std::vector<Dimer> &dimers, const std::string &basename,
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
        dimer_energy = energy_model(dimer);
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
    CEPairEnergyFunctor f(crystal, wfns_a, wfns_b);
    compute_dimer_energies_radius(f, crystal, dimers, basename,
                                  std::numeric_limits<double>::max(), result);
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

template <typename EnergyModel>
std::pair<occ::crystal::CrystalDimers, std::vector<CEEnergyComponents>>
converged_lattice_energies(EnergyModel &energy_model, const Crystal &crystal,
                           const std::string &basename,
                           const LatticeConvergenceSettings conv) {

    crystal::CrystalDimers converged_dimers;
    std::vector<CEEnergyComponents> converged_energies;
    std::vector<double> charge_energies;
    double lattice_energy{0.0}, previous_lattice_energy{0.0};
    double current_radius = std::max(conv.radius_increment, conv.min_radius);
    size_t cycle{1};

    auto all_dimers = crystal.symmetry_unique_dimers(conv.max_radius);
    auto asym_mols = crystal.symmetry_unique_molecules();

    Vec asym_charges(crystal.asymmetric_unit().size());
    std::vector<double> charge_self_energies(asym_mols.size());
    // vector per unique molecule or wfn.

    const auto &partial_charges = energy_model.partial_charges();
    for (int i = 0; i < partial_charges.size(); i++) {
        const auto &asymmetric_atom_indices =
            asym_mols[i].asymmetric_unit_idx();
        const auto &charge_vector = partial_charges[i];
        occ::log::info("Charges {}\n{}", i, charge_vector);
        for (int j = 0; j < charge_vector.rows(); j++) {
            asym_charges(asymmetric_atom_indices(j)) = charge_vector(j);
        }
    }
    occ::log::info("Found {} symmetry unique dimers within max radius {:.3f}\n",
                   all_dimers.unique_dimers.size(), conv.max_radius);
    occ::log::info("Lattice convergence settings:");
    occ::log::info("Start radius       {: 8.4f} \u212b", conv.min_radius);
    occ::log::info("Max radius         {: 8.4f} \u212b", conv.max_radius);
    occ::log::info("Radius increment   {: 8.4f} \u212b", conv.radius_increment);
    occ::log::info("Energy tolerance   {: 8.4f} kJ/mol", conv.energy_tolerance);

    double wolf_energy = 0.0;
    int asym_idx = 0;
    auto surrounds = crystal.asymmetric_unit_atom_surroundings(conv.max_radius);
    auto wolf_params = occ::interaction::WolfParams{16.0, 0.2};
    Mat3N asym_cart = crystal.to_cartesian(crystal.asymmetric_unit().positions);
    Vec asym_wolf(surrounds.size());
    for (const auto &s : surrounds) {
        double qi = asym_charges(asym_idx);
        Vec3 pi = asym_cart.col(asym_idx);
        Vec qj(s.size());
        for (int j = 0; j < qj.rows(); j++) {
            qj(j) = asym_charges(s.asym_idx(j));
        }
        asym_wolf(asym_idx) = occ::interaction::wolf_coulomb_energy(
                                  qi, pi, qj, s.cart_pos, wolf_params) *
                              units::AU_TO_KJ_PER_MOL;
        asym_idx++;
    }
    for (int i = 0; i < asym_mols.size(); i++) {
        const auto &mol = asym_mols[i];
        charge_self_energies[i] =
            occ::interaction::coulomb_self_energy_asym_charges(mol,
                                                               asym_charges);
        for (int j = 0; j < mol.size(); j++) {
            wolf_energy += asym_wolf(mol.asymmetric_unit_idx()(j));
        }
    }

    occ::log::debug("Wolf energy ({} asymmetric atoms): {}\n", asym_idx,
                    asym_wolf.sum());

    occ::log::debug("Wolf energy ({} asymmetric molecules): {}\n",
                    asym_mols.size(), wolf_energy);

    do {
        previous_lattice_energy = lattice_energy;
        const auto &dimers = all_dimers.unique_dimers;
        compute_dimer_energies_radius(energy_model, crystal, dimers, basename,
                                      current_radius, converged_energies);
        compute_coulomb_energies_radius(dimers, asym_charges, current_radius,
                                        charge_energies);

        const auto &mol_neighbors = all_dimers.molecule_neighbors;
        double etot{0.0};
        size_t mol_idx{0};
        double ecoul_real{0};
        double ecoul_exact_real{0};
        double ecoul_self{0};
        double coulomb_scale_factor = occ::interaction::CE_B3LYP_631Gdp.coulomb;
        for (const auto &n : mol_neighbors) {
            double molecule_total{0.0};
            size_t dimer_idx{0};
            for (const auto &dimer : n) {
                int unique_idx =
                    all_dimers.unique_dimer_idx[mol_idx][dimer_idx];
                const auto &e = converged_energies[unique_idx];
                double e_charge = charge_energies[unique_idx];

                if (e.is_computed) {
                    molecule_total += e.total_kjmol();
                    ecoul_exact_real += 0.5 * e.coulomb_kjmol();
                    ecoul_real += 0.5 * e_charge * occ::units::AU_TO_KJ_PER_MOL;
                }
                dimer_idx++;
            }
            etot += molecule_total;
            ecoul_self +=
                charge_self_energies[mol_idx] * units::AU_TO_KJ_PER_MOL;
            mol_idx++;
        }
        lattice_energy = 0.5 * etot;
        occ::log::info("Cycle {} lattice energy: {}", cycle, lattice_energy);
        occ::log::debug("Charge-charge intramolecular: {}", ecoul_self);
        occ::log::debug("Charge-charge real space: {}", ecoul_real);
        occ::log::debug("Wolf energy: {}", wolf_energy);
        occ::log::debug("Coulomb (exact) real: {}", ecoul_exact_real);
        occ::log::debug("Wolf - intra: {}", wolf_energy - ecoul_self);
        occ::log::debug("Wolf corrected Coulomb total: {}",
                        wolf_energy - ecoul_self - ecoul_real +
                            ecoul_exact_real);
        occ::log::debug("Wolf corrected lattice energy: {}",
                        coulomb_scale_factor *
                                (wolf_energy - ecoul_self - ecoul_real) +
                            0.5 * etot);
        cycle++;
        current_radius += conv.radius_increment;
    } while (std::abs(lattice_energy - previous_lattice_energy) >
             conv.energy_tolerance);
    converged_dimers = all_dimers;
    return std::make_pair(converged_dimers, converged_energies);
}

std::pair<occ::crystal::CrystalDimers, std::vector<CEEnergyComponents>>
converged_lattice_energies(const Crystal &crystal,
                           const std::vector<Wavefunction> &wfns_a,
                           const std::vector<Wavefunction> &wfns_b,
                           const std::string &basename,
                           const LatticeConvergenceSettings conv) {

    CEPairEnergyFunctor energy_model(crystal, wfns_a, wfns_b);
    return converged_lattice_energies(energy_model, crystal, basename, conv);
}

std::pair<occ::crystal::CrystalDimers, std::vector<CEEnergyComponents>>
converged_xtb_lattice_energies(const Crystal &crystal,
                               const std::string &basename,
                               const LatticeConvergenceSettings conv) {

    XTBPairEnergyFunctor energy_model(crystal);
    return converged_lattice_energies(energy_model, crystal, basename, conv);
}

} // namespace occ::main
