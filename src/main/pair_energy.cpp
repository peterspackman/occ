#include <ankerl/unordered_dense.h>
#include <filesystem>
#include <fmt/chrono.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <nlohmann/json.hpp>
#include <occ/core/log.h>
#include <occ/core/molecule.h>
#include <occ/core/timings.h>
#include <occ/core/util.h>
#include <occ/interaction/coulomb.h>
#include <occ/interaction/pair_potential.h>
#include <occ/interaction/pairinteraction.h>
#include <occ/interaction/wolf.h>
#include <occ/io/wavefunction_json.h>
#include <occ/main/pair_energy.h>
#include <occ/qm/chelpg.h>
#include <occ/xtb/xtb_wrapper.h>
#include <optional>
#include <occ/core/progress.h>

namespace fs = std::filesystem;

namespace occ::interaction {

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CEEnergyComponents, coulomb, exchange,
                                   repulsion, polarization, dispersion, total)
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
        occ::log::debug("Loading Gaussian fchk file from {}", filename);
        using occ::io::FchkReader;
        FchkReader fchk(filename);
        return Wavefunction(fchk);
    } else if (ext == ".molden" || ext == ".input") {
        occ::log::debug("Loading molden file from {}", filename);
        using occ::io::MoldenReader;
        MoldenReader molden(filename);
        occ::log::debug("Wavefunction has {} atoms", molden.atoms().size());
        return Wavefunction(molden);
    } else if (ext == ".json") {
        occ::log::debug("Loading OCC JSON wavefunction from {}", filename);
        occ::io::JsonWavefunctionReader reader(filename);
        return reader.wavefunction();
    } else if (ext == ".orca.json") {
        occ::log::debug("Loading Orca JSON file from {}", filename);
        occ::io::OrcaJSONReader json(filename);
        return Wavefunction(json);
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
    occ::log::debug(
        "Transformed atomic positions for monomer A (charge = {}, ecp = {})",
        a.wfn.charge(), a.wfn.basis.ecp_electrons().size() > 0);
    for (const auto &a : a.wfn.atoms) {
        occ::log::debug("{} {:20.12f} {:20.12f} {:20.12f}",
                        occ::core::Element(a.atomic_number).symbol(),
                        a.x / occ::units::ANGSTROM_TO_BOHR,
                        a.y / occ::units::ANGSTROM_TO_BOHR,
                        a.z / occ::units::ANGSTROM_TO_BOHR);
    }
    occ::log::debug(
        "Transformed atomic positions for monomer B (charge = {}, ecp = {})",
        b.wfn.charge(), b.wfn.basis.ecp_electrons().size() > 0);
    for (const auto &a : b.wfn.atoms) {
        occ::log::debug("{} {:20.12f} {:20.12f} {:20.12f}",
                        occ::core::Element(a.atomic_number).symbol(),
                        a.x / occ::units::ANGSTROM_TO_BOHR,
                        a.y / occ::units::ANGSTROM_TO_BOHR,
                        a.z / occ::units::ANGSTROM_TO_BOHR);
    }
    model = occ::interaction::ce_model_from_string(input.pair.model_name);
}

CEEnergyComponents PairEnergy::compute() {

    CEModelInteraction interaction(model);
    energy = interaction(a.wfn, b.wfn);

    occ::log::info("Monomer A energies\n{}", a.wfn.energy.to_string());
    occ::log::info("Monomer B energies\n{}", b.wfn.energy.to_string());

    occ::log::info("Dimer");

    occ::log::info("Component              Energy (kJ/mol)\n");
    occ::log::info("Coulomb               {: 12.6f}",
                   energy.coulomb_kjmol());
    occ::log::info("Exchange              {: 12.6f}",
                   energy.exchange_kjmol());
    occ::log::info("Repulsion             {: 12.6f}",
                   energy.repulsion_kjmol());
    occ::log::info("Polarization          {: 12.6f}",
                   energy.polarization_kjmol());
    occ::log::info("Dispersion            {: 12.6f}",
                   energy.dispersion_kjmol());
    occ::log::info("__________________________________");
    occ::log::info("Total 		      {: 12.6f}",
                   energy.total_kjmol());
    return energy;
}

auto calculate_transform(const Wavefunction &wfn, const Molecule &m,
                         const Crystal &c) {
    using occ::crystal::SymmetryOperation;

    Mat3N pos_m = m.positions();
    Mat3 rotation;
    Vec3 translation;

    ankerl::unordered_dense::set<int> symops_tested;

    for (int i = 0; i < m.size(); i++) {
        int sint = m.asymmetric_unit_symop()(i);
        if (symops_tested.contains(sint))
            continue;
        symops_tested.insert(sint);
        SymmetryOperation symop(sint);
        occ::Mat3N positions = wfn.positions() * BOHR_TO_ANGSTROM;

        rotation =
            c.unit_cell().direct() * symop.rotation() * c.unit_cell().inverse();
        translation = (m.centroid() - (rotation * positions).rowwise().mean()) /
                      BOHR_TO_ANGSTROM;
        Wavefunction tmp = wfn;
        tmp.apply_transformation(rotation, translation);

        occ::Mat3N tmp_t = tmp.positions() * BOHR_TO_ANGSTROM;
        double rmsd = (tmp_t - pos_m).norm();
        occ::log::debug("Test transform (symop={}) RMSD = {}\n",
                        symop.to_string(), rmsd);
        if (rmsd < 1e-3) {
            occ::log::debug(
                "Symop found: RMSD = {}\nrotation\n{}\ntranslation\n{}\n", rmsd,
                rotation, translation);
            return std::make_pair(rotation, translation);
        }
    }
    throw std::runtime_error(
        "Unable to determine symmetry operation to transform wavefunction");
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

    void set_model_name(const std::string &model_name) {
        m_model_name = model_name;
    }

    CEEnergyComponents operator()(const Dimer &dimer) {
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
        occ::log::debug("Mol A transformed wavefunction positions RMSD = {}",
                        (pos_A_t - pos_A).norm());

        auto transform_b = calculate_transform(B, mol_B, m_crystal);
        B.apply_transformation(transform_b.first, transform_b.second);

        Mat3N pos_B = mol_B.positions();
        Mat3N pos_B_t = B.positions() * BOHR_TO_ANGSTROM;
        occ::log::debug("Mol B transformed wavefunction positions RMSD = {}",
                        (pos_B_t - pos_B).norm());

        auto model = occ::interaction::ce_model_from_string(m_model_name);

        CEModelInteraction interaction(model);

        auto interaction_energy = interaction(A, B);
        interaction_energy.is_computed = true;

        occ::log::debug("Finished model energy");
        return interaction_energy;
    }

    Mat3N electric_field(const Dimer &dimer) {
        Molecule mol_A = dimer.a();
        Molecule mol_B = dimer.b();
        Wavefunction A = m_wavefunctions_a[mol_A.asymmetric_molecule_idx()];
        Wavefunction B =
            (m_wavefunctions_b.size() > 0)
                ? m_wavefunctions_a[mol_B.asymmetric_molecule_idx()]
                : m_wavefunctions_b[mol_B.asymmetric_molecule_idx()];

        auto transform_b = calculate_transform(B, mol_B, m_crystal);
        B.apply_transformation(transform_b.first, transform_b.second);

        occ::Mat3N pos_A = mol_A.positions();
        Mat3N pos_A_t = A.positions() * BOHR_TO_ANGSTROM;

        occ::Mat3N pos_A_bohr =
            mol_A.positions() * occ::units::ANGSTROM_TO_BOHR;

        Mat3N pos_B = mol_B.positions();
        Mat3N pos_B_t = B.positions() * BOHR_TO_ANGSTROM;

        return B.electric_field(pos_A_bohr);
    }

    inline const auto &wavefunctions() const { return m_wavefunctions_a; }

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

    inline double coulomb_scale_factor() const {
        auto model = occ::interaction::ce_model_from_string(m_model_name);
        return model.coulomb;
    }

  private:
    Crystal m_crystal;
    std::string m_model_name{"ce-b3lyp"};
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
    inline double coulomb_scale_factor() const { return 1.0; }

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
        sw.start();

        charge_energies[current_dimer] =
            occ::interaction::coulomb_interaction_energy_asym_charges(
                dimer, asym_charges);
        current_dimer++;
    }
    occ::log::debug("Finished calculating {} unique dimer coulomb energies",
                    computed_dimers);
    return computed_dimers;
}

Mat3N compute_point_charge_efield(const Dimer &dimer, const Vec &asym_charges) {
    return occ::interaction::coulomb_efield_asym_charges(dimer, asym_charges)
        .first;
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
    size_t dimers_to_compute{0};
    for (size_t i = 0; i < dimers.size(); i++) {
        const auto &dimer = dimers[i];
        if (dimer.nearest_distance() > radius || dimer_energies[i].is_computed) continue;
        dimers_to_compute++;
    }

    PairEnergyStore store{PairEnergyStore::Kind::Xyz, fmt::format("{}_dimers", basename)};
    occ::core::ProgressTracker progress(dimers_to_compute);

    for (const auto &dimer : dimers) {
        auto tprev = sw.read();
        sw.start();

        const auto &a = dimer.a();
        const auto &b = dimer.b();
        const auto asym_idx_a = a.asymmetric_molecule_idx();
        const auto asym_idx_b = b.asymmetric_molecule_idx();
        const auto shift_b = dimer.b().cell_shift();
        std::string dimer_name = dimer.name();

        CEEnergyComponents &dimer_energy = dimer_energies[current_dimer];
        std::string dimer_energy_file(
            fmt::format("{}_dimer_{}_energies.xyz", basename, current_dimer));

        if (dimer.nearest_distance() > radius || dimer_energy.is_computed) {
          current_dimer++;
          continue;
        }
        if(store.load(current_dimer, dimer, dimer_energy)) {
          progress.update(computed_dimers, dimers_to_compute, 
                              fmt::format("Load [{}|{}]: {}", asym_idx_a, asym_idx_b, dimer_name));
          computed_dimers++;
          current_dimer++;
          continue;
        }

        progress.update(computed_dimers, dimers_to_compute, 
                        fmt::format("E[{}|{}]: {}", asym_idx_a, asym_idx_b, dimer_name));
        occ::log::debug(
            "{} ({}[{}] - {}[{} + ({},{},{})]), Rc = {: 5.2f}", current_dimer,
            asym_idx_a,
            SymmetryOperation(a.asymmetric_unit_symop()(0)).to_string(),
            asym_idx_b,
            SymmetryOperation(b.asymmetric_unit_symop()(0)).to_string(),
            shift_b[0], shift_b[1], shift_b[2],
            dimer.center_of_mass_distance());

        std::cout << std::flush;
        dimer_energy = energy_model(dimer);
        sw.stop();
        occ::log::debug("Took {:.3f} seconds", sw.read() - tprev);
        store.save(current_dimer, dimer, dimer_energy);
        computed_dimers++;
        current_dimer++;
    }
    progress.clear();
    occ::log::info("Finished calculating {} unique dimer interaction energies in {:%H:%M:%S}",
                   computed_dimers, std::chrono::round<std::chrono::seconds>(progress.time_taken()));
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
    occ::log::debug("Load dimer energies from {}", filename);
    std::ifstream file(filename);
    std::string line;
    std::getline(file, line);
    std::getline(file, line);
    energies = nlohmann::json::parse(line).get<CEEnergyComponents>();
    energies.is_computed = true;
    return true;
}

struct WolfSumAccelerator {
    occ::interaction::WolfParams parameters{16.0, 0.2};
    Vec asym_charges;
    std::vector<double> charge_self_energies;
    std::vector<Mat3N> electric_field_values;
    double energy{0.0};

    template <typename EnergyModel>
    void setup(const Crystal &crystal, EnergyModel &energy_model) {
        auto asym_mols = crystal.symmetry_unique_molecules();
        asym_charges = Vec(crystal.asymmetric_unit().size());
        charge_self_energies = std::vector<double>(asym_mols.size());
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

        int asym_idx = 0;
        auto surrounds =
            crystal.asymmetric_unit_atom_surroundings(parameters.cutoff);
        Mat3N asym_cart =
            crystal.to_cartesian(crystal.asymmetric_unit().positions);
        Vec asym_wolf(surrounds.size());
        for (const auto &s : surrounds) {
            double qi = asym_charges(asym_idx);
            Vec3 pi = asym_cart.col(asym_idx);
            Vec qj(s.size());
            for (int j = 0; j < qj.rows(); j++) {
                qj(j) = asym_charges(s.asym_idx(j));
            }
            asym_wolf(asym_idx) = occ::interaction::wolf_coulomb_energy(
                                      qi, pi, qj, s.cart_pos, parameters) *
                                  units::AU_TO_KJ_PER_MOL;
            asym_idx++;
        }
        for (int i = 0; i < asym_mols.size(); i++) {
            const auto &mol = asym_mols[i];
            electric_field_values.push_back(Mat3N::Zero(3, mol.size()));
            charge_self_energies[i] =
                occ::interaction::coulomb_self_energy_asym_charges(
                    mol, asym_charges);
            for (int j = 0; j < mol.size(); j++) {
                energy += asym_wolf(mol.asymmetric_unit_idx()(j));
            }
        }

        occ::log::debug("Wolf energy ({} asymmetric atoms): {}\n", asym_idx,
                        asym_wolf.sum());

        occ::log::debug("Wolf energy ({} asymmetric molecules): {}\n",
                        asym_mols.size(), energy);
    }

    void initialize(const Crystal &crystal) {}
};

template <typename EnergyModel>
LatticeEnergyResult
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
    const auto &asym_mols = crystal.symmetry_unique_molecules();

    occ::log::info("Found {} symmetry unique dimers within max radius {:.3f}\n",
                   all_dimers.unique_dimers.size(), conv.max_radius);
    occ::log::info("Lattice convergence settings:");
    occ::log::info("Start radius       {: 8.4f} angstrom", conv.min_radius);
    occ::log::info("Max radius         {: 8.4f} angstrom", conv.max_radius);
    occ::log::info("Radius increment   {: 8.4f} angstrom",
                   conv.radius_increment);
    occ::log::info("Energy tolerance   {: 8.4f} kJ/mol", conv.energy_tolerance);

    WolfSumAccelerator wolf;
    if (conv.wolf_sum) {
        wolf.parameters.cutoff = conv.max_radius;
        wolf.setup(crystal, energy_model);
    }

    do {
        previous_lattice_energy = lattice_energy;
        const auto &dimers = all_dimers.unique_dimers;
        compute_dimer_energies_radius(energy_model, crystal, dimers, basename,
                                      current_radius, converged_energies);
        if (conv.wolf_sum) {
            compute_coulomb_energies_radius(dimers, wolf.asym_charges,
                                            current_radius, charge_energies);
        }

        const auto &mol_neighbors = all_dimers.molecule_neighbors;
        CEEnergyComponents total;
        size_t mol_idx{0};
        double epol_total{0};
        double ecoul_real{0};
        double ecoul_exact_real{0};
        double ecoul_self{0};
        double coulomb_scale_factor = energy_model.coulomb_scale_factor();
        for (const auto &n : mol_neighbors) {
            CEEnergyComponents molecule_total;
            size_t dimer_idx{0};

            Mat3N efield = Mat3N::Zero(3, asym_mols[mol_idx].size());

            if (conv.wolf_sum && conv.crystal_field_polarization) {
                wolf.electric_field_values[mol_idx].setZero();
                occ::log::debug("Field total =\n{}\n",
                                wolf.electric_field_values[mol_idx]);
            }

            for (const auto &[dimer, unique_idx] : n) {
                const auto &e = converged_energies[unique_idx];

                if (e.is_computed) {
                    molecule_total += e;
                    epol_total += e.polarization_kjmol();
                    if (conv.wolf_sum) {
                        double e_charge = charge_energies[unique_idx];
                        ecoul_exact_real += 0.5 * e.coulomb_kjmol();
                        ecoul_real +=
                            0.5 * e_charge * occ::units::AU_TO_KJ_PER_MOL;
                        if (conv.crystal_field_polarization) {
                            wolf.electric_field_values[mol_idx] +=
                                compute_point_charge_efield(dimer,
                                                            wolf.asym_charges);
                        }
                    }
                    if (conv.crystal_field_polarization) {
                        if constexpr (std::is_same<
                                          EnergyModel,
                                          CEPairEnergyFunctor>::value) {
                            efield += energy_model.electric_field(dimer);
                        }
                    }
                }
                dimer_idx++;
            }
            total += molecule_total;
            if (conv.wolf_sum) {
                ecoul_self += wolf.charge_self_energies[mol_idx] *
                              units::AU_TO_KJ_PER_MOL;

                if (conv.crystal_field_polarization) {
                    auto &electric_field = wolf.electric_field_values[mol_idx];
                    occ::log::info("Field total =\n{}", electric_field);
                    occ::log::info("Field total =\n{}", efield);
                    if constexpr (std::is_same<EnergyModel,
                                               CEPairEnergyFunctor>::value) {
                        const auto &wfn_a =
                            energy_model.wavefunctions()[mol_idx];
                        double e_pol_chg =
                            occ::interaction::polarization_energy(
                                wfn_a.xdm_polarizabilities, electric_field);
                        occ::log::debug("Crystal polarizability (chg): {}",
                                        e_pol_chg *
                                            occ::units::AU_TO_KJ_PER_MOL);
                    }
                }
            }
            if (conv.crystal_field_polarization) {
                if constexpr (std::is_same<EnergyModel,
                                           CEPairEnergyFunctor>::value) {
                    const auto &wfn_a = energy_model.wavefunctions()[mol_idx];
                    double e_pol_qm = occ::interaction::polarization_energy(
                        wfn_a.xdm_polarizabilities, efield);
                    occ::log::debug("Crystal polarizability (qm): {}",
                                    e_pol_qm * occ::units::AU_TO_KJ_PER_MOL);
                }
            }
            mol_idx++;
        }
        lattice_energy = 0.5 * total.total_kjmol();
        if (conv.wolf_sum) {
            occ::log::debug("Charge-charge intramolecular: {}", ecoul_self);
            occ::log::debug("Charge-charge real space: {}", ecoul_real);
            occ::log::debug("Wolf energy: {}", wolf.energy);
            occ::log::debug("Coulomb (exact) real: {}", ecoul_exact_real);
            occ::log::debug("Wolf - intra: {}", wolf.energy - ecoul_self);
            occ::log::debug("Wolf corrected Coulomb total: {}",
                            wolf.energy - ecoul_self - ecoul_real +
                                ecoul_exact_real);
            lattice_energy =
                coulomb_scale_factor * (wolf.energy - ecoul_self - ecoul_real) +
                0.5 * total.total_kjmol();
            occ::log::info("Wolf corrected lattice energy: {}", lattice_energy);
        }
        occ::log::info("Cycle {} lattice energy: {}", cycle, lattice_energy);
        occ::log::debug("Total polarization term: {:.3f}",
                        0.5 * total.polarization_kjmol());
        occ::log::debug("Total coulomb term: {:.3f}",
                        0.5 * total.coulomb_kjmol());
        occ::log::debug("Total dispersion term: {:.3f}",
                        0.5 * total.dispersion_kjmol());
        occ::log::debug("Total repulsion term: {:.3f}",
                        0.5 * total.repulsion_kjmol());
        occ::log::debug("Total exchange term: {:.3f}",
                        0.5 * total.exchange_kjmol());
        occ::log::debug("Wolf correction: {:.3f}",
                        (wolf.energy - ecoul_self - ecoul_real));
        cycle++;
        current_radius += conv.radius_increment;
    } while (std::abs(lattice_energy - previous_lattice_energy) >
             conv.energy_tolerance);
    converged_dimers = all_dimers;
    return {lattice_energy, converged_dimers, converged_energies};
}

LatticeEnergyResult converged_lattice_energies(
    const Crystal &crystal, const std::vector<Wavefunction> &wfns_a,
    const std::vector<Wavefunction> &wfns_b, const std::string &basename,
    const LatticeConvergenceSettings conv) {

    CEPairEnergyFunctor energy_model(crystal, wfns_a, wfns_b);
    energy_model.set_model_name(conv.model_name);
    return converged_lattice_energies(energy_model, crystal, basename, conv);
}

LatticeEnergyResult
converged_xtb_lattice_energies(const Crystal &crystal,
                               const std::string &basename,
                               const LatticeConvergenceSettings conv) {

    XTBPairEnergyFunctor energy_model(crystal);
    return converged_lattice_energies(energy_model, crystal, basename, conv);
}

std::string PairEnergyStore::dimer_filename(int id, const Dimer &d) {
  return fmt::format("dimer_{}.xyz", id);
}

bool PairEnergyStore::save(int id, const Dimer &d, const CEEnergyComponents &e) {
  fs::path parent(name);
  if(!fs::exists(parent)) {
    fs::create_directories(parent);
  }
  return write_xyz_dimer((parent / fs::path(dimer_filename(id, d))).string(), d, e);
}

bool PairEnergyStore::load(int id, const Dimer &d, CEEnergyComponents &e) {
  fs::path parent(name);
  return load_dimer_energy((parent / fs::path(dimer_filename(id, d))).string(), e);
}

} // namespace occ::main
