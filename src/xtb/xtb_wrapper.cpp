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
      m_atomic_numbers(mol.atomic_numbers()) {
    initialize_context();
    initialize_structure();
    initialize_method();
}

XTBCalculator::XTBCalculator(const Molecule &mol, Method method)
    : m_positions_bohr(mol.positions() * occ::units::ANGSTROM_TO_BOHR),
      m_atomic_numbers(mol.atomic_numbers()), m_method(method) {
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
    int natoms = m_atomic_numbers.rows();
    m_gradients = Mat3N::Zero(3, natoms);
    m_tb_structure = tblite_new_structure(
        m_tb_error, natoms, m_atomic_numbers.data(), m_positions_bohr.data(),
        nullptr, nullptr, nullptr, nullptr);
    check_error(m_tb_error);
}

void XTBCalculator::initialize_method() {
    switch (m_method) {
    case Method::GFN2:
        m_tb_calc = tblite_new_gfn2_calculator(m_tb_ctx, m_tb_structure);
        break;
    case Method::GFN1:
        m_tb_calc = tblite_new_gfn1_calculator(m_tb_ctx, m_tb_structure);
        break;
    }
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
