#include <cstdint>
#include <cstdlib>
#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/xtb/xtb_wrapper.h>

using occ::core::Molecule;

namespace occ::xtb {

void check_error(tblite_error err) {
    if (tblite_check_error(err)) {
        char message[512];
        tblite_get_error(err, message, nullptr);
        occ::log::critical("Fatal error in tblite: {}", message);
        throw std::runtime_error("Unrecoverable error using tblite");
    }
}

std::string tblite_version() {
    auto v = tblite_get_version();
    return fmt::format("v{}", v);
}

XTBCalculator::XTBCalculator(const Molecule &mol)
    : m_positions_bohr(mol.positions() * occ::units::ANGSTROM_TO_BOHR),
      m_atomic_numbers(mol.atomic_numbers()), m_charge(mol.charge()),
      m_num_unpaired_electrons(mol.multiplicity() - 1) {
    initialize_context();
    initialize_structure();
    initialize_method();
}

XTBCalculator::XTBCalculator(const Molecule &mol, Method method)
    : m_positions_bohr(mol.positions() * occ::units::ANGSTROM_TO_BOHR),
      m_atomic_numbers(mol.atomic_numbers()), m_method(method),
      m_charge(mol.charge()), m_num_unpaired_electrons(mol.multiplicity() - 1) {
    initialize_context();
    initialize_structure();
    initialize_method();
}

void XTBCalculator::initialize_context() {
    m_tb_error = tblite_new_error();
    m_tb_ctx = tblite_new_context();
    m_tb_result = tblite_new_result();
}

void XTBCalculator::initialize_structure() {
    if (m_tb_structure) {
        tblite_delete_structure(&m_tb_structure);
    }
    int natoms = m_atomic_numbers.rows();
    m_gradients = Mat3N::Zero(3, natoms);
    m_tb_structure = tblite_new_structure(
        m_tb_error, natoms, m_atomic_numbers.data(), m_positions_bohr.data(),
        &m_charge, &m_num_unpaired_electrons, m_lattice_vectors.data(),
        m_periodic.data());
    check_error(m_tb_error);
}

void XTBCalculator::initialize_method() {
    if (m_tb_calc) {
        tblite_delete_calculator(&m_tb_calc);
    }
    switch (m_method) {
    case Method::GFN2:
        m_tb_calc = tblite_new_gfn2_calculator(m_tb_ctx, m_tb_structure);
        break;
    case Method::GFN1:
        m_tb_calc = tblite_new_gfn1_calculator(m_tb_ctx, m_tb_structure);
        break;
    }
}

void XTBCalculator::update_structure(const Mat3N &new_positions) {
    m_positions_bohr = new_positions;
    tblite_update_structure_geometry(m_tb_error, m_tb_structure,
                                     m_positions_bohr.data(),
                                     m_lattice_vectors.data());
    check_error(m_tb_error);
}

void XTBCalculator::update_structure(const Mat3N &new_positions,
                                     const Mat3 &lattice) {
    m_positions_bohr = new_positions;
    m_lattice_vectors = lattice;
    tblite_update_structure_geometry(m_tb_error, m_tb_structure,
                                     m_positions_bohr.data(),
                                     m_lattice_vectors.data());
    check_error(m_tb_error);
}

void XTBCalculator::set_charge(double charge) {
    m_charge = charge;
    tblite_update_structure_charge(m_tb_error, m_tb_structure, &m_charge);
    check_error(m_tb_error);
}

void XTBCalculator::set_num_unpaired_electrons(int n) {
    m_num_unpaired_electrons = n;
    tblite_update_structure_uhf(m_tb_error, m_tb_structure,
                                &m_num_unpaired_electrons);
    check_error(m_tb_error);
}

void XTBCalculator::set_accuracy(double accuracy) {
    if (!m_tb_calc || !m_tb_ctx)
        return;
    tblite_set_calculator_accuracy(m_tb_ctx, m_tb_calc, accuracy);
}

void XTBCalculator::set_max_iterations(int iterations) {
    if (!m_tb_calc || !m_tb_ctx)
        return;
    tblite_set_calculator_max_iter(m_tb_ctx, m_tb_calc, iterations);
}

void XTBCalculator::set_temperature(double temp) {
    if (!m_tb_calc || !m_tb_ctx)
        return;
    tblite_set_calculator_temperature(m_tb_ctx, m_tb_calc, temp);
}

void XTBCalculator::set_mixer_damping(double damping_factor) {
    if (!m_tb_calc || !m_tb_ctx)
        return;
    tblite_set_calculator_mixer_damping(m_tb_ctx, m_tb_calc, damping_factor);
}

double XTBCalculator::single_point_energy() {
    double energy;
    tblite_get_singlepoint(m_tb_ctx, m_tb_structure, m_tb_calc, m_tb_result);
    tblite_get_result_energy(m_tb_error, m_tb_result, &energy);
    check_error(m_tb_error);
    tblite_get_result_gradient(m_tb_error, m_tb_result, m_gradients.data());
    check_error(m_tb_error);
    return energy;
}

Vec XTBCalculator::charges() const {
    int natoms;
    tblite_get_result_number_of_atoms(m_tb_error, m_tb_result, &natoms);
    check_error(m_tb_error);
    Vec chg(natoms);
    tblite_get_result_charges(m_tb_error, m_tb_result, chg.data());
    check_error(m_tb_error);
    return chg;
}

Mat XTBCalculator::bond_orders() const {
    int natoms;
    tblite_get_result_number_of_atoms(m_tb_error, m_tb_result, &natoms);
    check_error(m_tb_error);
    Mat bo(natoms, natoms);
    tblite_get_result_bond_orders(m_tb_error, m_tb_result, bo.data());
    check_error(m_tb_error);
    return bo;
}

XTBCalculator::~XTBCalculator() {
    if (m_tb_error) {
        tblite_delete_error(&m_tb_error);
    }
    if (m_tb_ctx) {
        tblite_delete_context(&m_tb_ctx);
    }
    if (m_tb_result) {
        tblite_delete_result(&m_tb_result);
    }
    if (m_tb_structure) {
        tblite_delete_structure(&m_tb_structure);
    }
}

} // namespace occ::xtb
