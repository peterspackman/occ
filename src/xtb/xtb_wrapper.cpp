#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/3rdparty/subprocess.hpp>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/units.h>
#include <occ/io/eigen_json.h>
#include <occ/io/engrad.h>
#include <occ/xtb/xtb_wrapper.h>

using occ::core::Dimer;
using occ::core::Molecule;
using occ::crystal::Crystal;
namespace fs = std::filesystem;

namespace occ::xtb {

inline bool try_remove_file(const std::string &filename) {
    fs::path file_path(filename);
    return fs::remove(file_path);
}

struct XTBJsonOutput {
    double energy{0.0};
    double homo_lumo_gap{0.0};
    double electronic_energy{0.0};
    Vec3 dipole{0.0, 0.0, 0.0};
    Vec partial_charges;
    Mat atomic_dipoles;
    Mat atomic_quadrupoles;
    int num_molecular_orbitals{0};
    int num_electrons{0};
    int num_unpaired_electrons{0};
    Vec orbital_energies;
    Vec fractional_occupation;
    std::string program_call;
    std::string method;
    std::string xtb_version;
};

void from_json(const nlohmann::json &J, XTBJsonOutput &out) {
    auto maybe_get = [&](const char *name, auto &dest) {
        if (J.contains(name)) {
            J.at(name).get_to(dest);
        }
    };

    maybe_get("total energy", out.energy);
    maybe_get("HOMO-LUMO gap/eV", out.homo_lumo_gap);
    maybe_get("electronic energy", out.electronic_energy);

    maybe_get("dipole", out.dipole);
    maybe_get("partial charges", out.partial_charges);
    maybe_get("atomic dipole moments", out.atomic_dipoles);
    maybe_get("atomic quadrupole moments", out.atomic_quadrupoles);

    maybe_get("number of molecular orbitals", out.num_molecular_orbitals);
    maybe_get("number of electrons", out.num_electrons);
    maybe_get("number of unpaired electrons", out.num_unpaired_electrons);

    maybe_get("orbital energies/eV", out.orbital_energies);
    maybe_get("fractional occupation", out.fractional_occupation);

    maybe_get("program call", out.program_call);
    maybe_get("method", out.method);
    maybe_get("xtb version", out.xtb_version);
}

XTBCalculator::XTBCalculator(const Molecule &mol)
    : m_positions_bohr(mol.positions() * occ::units::ANGSTROM_TO_BOHR),
      m_atomic_numbers(mol.atomic_numbers()), m_charge(mol.charge()),
      m_num_unpaired_electrons(mol.multiplicity() - 1) {
    initialize_structure();
}

XTBCalculator::XTBCalculator(const Molecule &mol, Method method)
    : m_positions_bohr(mol.positions() * occ::units::ANGSTROM_TO_BOHR),
      m_atomic_numbers(mol.atomic_numbers()), m_method(method),
      m_charge(mol.charge()), m_num_unpaired_electrons(mol.multiplicity() - 1) {
    initialize_structure();
}

XTBCalculator::XTBCalculator(const Dimer &mol)
    : m_positions_bohr(mol.positions() * occ::units::ANGSTROM_TO_BOHR),
      m_atomic_numbers(mol.atomic_numbers()), m_charge(mol.charge()),
      m_num_unpaired_electrons(mol.multiplicity() - 1) {
    initialize_structure();
}

XTBCalculator::XTBCalculator(const Dimer &mol, Method method)
    : m_positions_bohr(mol.positions() * occ::units::ANGSTROM_TO_BOHR),
      m_atomic_numbers(mol.atomic_numbers()), m_method(method),
      m_charge(mol.charge()), m_num_unpaired_electrons(mol.multiplicity() - 1) {
    initialize_structure();
}

XTBCalculator::XTBCalculator(const Crystal &crystal)
    : m_positions_bohr(crystal.unit_cell_atoms().cart_pos *
                       occ::units::ANGSTROM_TO_BOHR),
      m_atomic_numbers(crystal.unit_cell_atoms().atomic_numbers), m_charge(0),
      m_num_unpaired_electrons(0), m_periodic{true, true, true},
      m_lattice_vectors(crystal.unit_cell().direct() *
                        occ::units::ANGSTROM_TO_BOHR) {
    initialize_structure();
}

XTBCalculator::XTBCalculator(const Crystal &crystal, Method method)
    : m_positions_bohr(crystal.unit_cell_atoms().cart_pos *
                       occ::units::ANGSTROM_TO_BOHR),
      m_atomic_numbers(crystal.unit_cell_atoms().atomic_numbers),
      m_method(method), m_charge(0),
      m_num_unpaired_electrons(0), m_periodic{true, true, true},
      m_lattice_vectors(crystal.unit_cell().direct() *
                        occ::units::ANGSTROM_TO_BOHR) {
    initialize_structure();
}

void XTBCalculator::initialize_structure() {
    int natoms = m_atomic_numbers.rows();
    m_gradients = Mat3N::Zero(3, natoms);
    m_virial = Mat3::Zero();
}

void XTBCalculator::update_structure(const Mat3N &new_positions) {
    m_positions_bohr = new_positions;
}

void XTBCalculator::update_structure(const Mat3N &new_positions,
                                     const Mat3 &lattice) {
    m_positions_bohr = new_positions;
    m_lattice_vectors = lattice;
}

void XTBCalculator::set_charge(double charge) { m_charge = charge; }

void XTBCalculator::set_num_unpaired_electrons(int n) {
    m_num_unpaired_electrons = n;
}

void XTBCalculator::set_accuracy(double accuracy) { m_accuracy = accuracy; }

void XTBCalculator::set_max_iterations(int iterations) {
    m_max_iterations = iterations;
}

void XTBCalculator::set_temperature(double temp) { m_temperature = temp; }

void XTBCalculator::set_mixer_damping(double damping_factor) {
    m_damping_factor = damping_factor;
}

void XTBCalculator::set_solvent(const std::string &solvent) {
    m_solvent = solvent;
}

void XTBCalculator::set_solvation_model(const std::string &model) {
    m_solvation_model = model;
}

double XTBCalculator::single_point_energy() {
    using subprocess::CalledProcessError;
    using subprocess::CompletedProcess;
    using subprocess::PipeOption;
    using subprocess::RunBuilder;
    subprocess::EnvGuard env_guard;

    const char *xtbinput_filename = "xtbinput.coord";
    const char *xtbengrad_filename = "xtbinput.engrad";
    write_input_file(xtbinput_filename);

    std::vector<std::string> command_line{
        m_xtb_executable_path, xtbinput_filename, "--json", "--grad"};
    if (!m_solvent.empty()) {
        command_line.push_back(fmt::format("--{}", m_solvation_model));
        command_line.push_back(m_solvent);
    }

    subprocess::cenv["OMP_NUM_THREADS"] =
        fmt::format("{}", occ::parallel::get_num_threads());
    auto process = subprocess::run(
        command_line,
        RunBuilder().cerr(PipeOption::pipe).cout(PipeOption::pipe));
    if (process.returncode != 0) {
        occ::log::critical("Error encountered when running xtb, return code = "
                           "{}, stdout contents:\n{}",
                           process.returncode, process.cout);
        occ::log::critical("stderr:\n{}", process.cerr);
        throw std::runtime_error("Failure when running xtb");
    }

    try_remove_file(xtbinput_filename);
    try_remove_file("wbo");
    try_remove_file("xtbrestart");
    try_remove_file("charges");
    try_remove_file("energy");
    try_remove_file("gradient");
    try_remove_file("xtbtopo.mol");
    read_json_contents("xtbout.json");
    try_remove_file("xtbout.json");
    read_engrad_contents(xtbengrad_filename);
    try_remove_file(xtbengrad_filename);
    return m_energy;
}

Vec XTBCalculator::charges() const {
    Vec chg(num_atoms());
    return chg;
}

Mat XTBCalculator::bond_orders() const {
    int natoms = num_atoms();
    Mat bo(natoms, natoms);
    return bo;
}

Crystal XTBCalculator::to_crystal() const {
    occ::crystal::UnitCell uc(m_lattice_vectors * occ::units::BOHR_TO_ANGSTROM);
    occ::crystal::SpaceGroup sg(1);
    occ::crystal::AsymmetricUnit asym(
        uc.to_fractional(m_positions_bohr * occ::units::BOHR_TO_ANGSTROM),
        m_atomic_numbers);
    return Crystal(asym, sg, uc);
}

Molecule XTBCalculator::to_molecule() const {
    return Molecule(m_atomic_numbers,
                    m_positions_bohr / occ::units::BOHR_TO_ANGSTROM);
}

int XTBCalculator::gfn_method() const {
    switch (m_method) {
    case Method::GFN1:
        return 1;
    default:
        return 2;
    }
}

void XTBCalculator::write_input_file(const std::string &dest) {
    std::ofstream of(dest);
    fmt::print(of, "$chrg {}\n", m_charge);
    fmt::print(of, "$spin {}\n", m_num_unpaired_electrons);
    fmt::print(of, "$gfn\n  method={}\n", gfn_method());
    fmt::print(of, "$coord\n");
    for (int i = 0; i < num_atoms(); i++) {
        fmt::print(of, "{:20.12f} {:20.12f} {:20.12f} {}\n",
                   m_positions_bohr(0, i), m_positions_bohr(1, i),
                   m_positions_bohr(2, i),
                   core::Element(m_atomic_numbers(i)).symbol());
    }
    fmt::print(of, "$end");
}

void XTBCalculator::read_json_contents(const std::string &json_filename) {
    std::ifstream json_output_file(json_filename);
    XTBJsonOutput output =
        nlohmann::json::parse(json_output_file).get<XTBJsonOutput>();
    m_energy = output.energy;
    m_partial_charges = output.partial_charges;
}

void XTBCalculator::read_engrad_contents(const std::string &engrad_filename) {
    io::EngradReader engrad(engrad_filename);
    m_gradients = engrad.gradient();
    m_positions_bohr = engrad.positions();
}

} // namespace occ::xtb
