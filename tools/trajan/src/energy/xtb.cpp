#include "xtb.h"
#include <occ/core/units.h>
#include <stdexcept>
#include <trajan/energy/xtb.h>

namespace trajan::energy {

void XTBModel::initialise() {
  m_env = xtb_newEnvironment();
  m_calc = xtb_newCalculator();
  m_res = xtb_newResults();
  xtb_setVerbosity(m_env, XTB_VERBOSITY_FULL);
  this->check_errors();
}

XTBModel::XTBModel()
    : m_env(nullptr), m_calc(nullptr), m_res(nullptr), m_mol(nullptr) {
  this->initialise();
}

void XTBModel::free_all() {
  if (m_res != nullptr)
    xtb_delResults(&m_res);
  if (m_calc != nullptr)
    xtb_delCalculator(&m_calc);
  if (m_mol != nullptr)
    xtb_delMolecule(&m_mol);
  if (m_env != nullptr)
    xtb_delEnvironment(&m_env);
}

XTBModel::~XTBModel() { this->free_all(); }

void XTBModel::check_errors() {
  if (!xtb_checkEnvironment(m_env)) {
    return;
  }
  std::vector<char> buffer(1000);
  int maxLength = buffer.size();
  xtb_getError(m_env, buffer.data(), &maxLength);
  throw std::runtime_error(std::string(buffer.data()));
}

void XTBModel::set_verbosity(int v) {
  if (m_env == nullptr) {
    trajan::log::warn("No environment to modify verbosity");
    return;
  }
  xtb_setVerbosity(m_env, v);
}

constexpr double FORCE_SCALE =
    occ::units::AU_TO_KJ_PER_MOL /
    occ::units::BOHR_TO_ANGSTROM; // Convert Hartree/bohr to kJ/mol/Ang
constexpr double HESS_SCALE =
    occ::units::AU_TO_KJ_PER_MOL / occ::units::BOHR_TO_ANGSTROM /
    occ::units::BOHR_TO_ANGSTROM; // Convert Hartree/bohr^2 to kJ/mol/Ang^2

void XTBModel::initialise_molecule(const core::Frame &frame, Type model) {
  int num_particles = frame.num_atoms();
  if (num_particles >= 1000) {
    trajan::log::warn(
        "Large systems (>1000) maybe be vulnerable to segmentation faults.");
    trajan::log::warn("Try increasing OMP_STACKSIZE if this occurs.");
  }
  occ::Vec positions = frame.cart_pos_flat(occ::units::ANGSTROM_TO_BOHR);

  double box_vectors[9];
  if (frame.has_unit_cell()) {
    auto box = frame.unit_cell().value().direct();
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        // convert column major to row major
        // TODO: test pbc
        box_vectors[3 * i + j] = occ::units::ANGSTROM_TO_BOHR * box(j, i);
      }
    }
  }
  if (!m_has_initialised_molecule) {
    bool periodic[3];
    if (frame.has_unit_cell()) {
      periodic[0] = periodic[1] = periodic[2] = true;
    } else {
      periodic[0] = periodic[1] = periodic[2] = false;
    }
    auto at = frame.atomic_numbers();
    m_mol = xtb_newMolecule(
        m_env, &num_particles, frame.atomic_numbers().data(), positions.data(),
        &frame.get_charge(), &frame.get_multiplicity(), box_vectors, periodic);

    this->check_errors();
    switch (model) {
    case GFN0xTB:
      xtb_loadGFN0xTB(m_env, m_mol, m_calc, NULL);
      break;
    case GFN1xTB:
      xtb_loadGFN1xTB(m_env, m_mol, m_calc, NULL);
      break;
    case GFN2xTB:
      xtb_loadGFN2xTB(m_env, m_mol, m_calc, NULL);
      break;
    case GFNFF:
      xtb_loadGFNFF(m_env, m_mol, m_calc, NULL);
      break;
    }
    this->check_errors();
    m_has_initialised_molecule = true;
  } else {
    xtb_updateMolecule(m_env, m_mol, positions.data(), box_vectors);
  }
  this->check_errors();
}

SinglePoint XTBModel::single_point(const core::Frame &frame, Type model) {
  if (m_previous_calc != Calc::Singlepoint) {
    this->free_all();
    this->initialise();
    m_previous_calc = Calc::Singlepoint;
  }

  this->initialise_molecule(frame, model);

  SinglePoint sp;
  xtb_singlepoint(m_env, m_mol, m_calc, m_res);
  this->check_errors();

  double energy;
  xtb_getEnergy(m_env, m_res, &energy);
  this->check_errors();
  sp.energy = energy * occ::units::AU_TO_KJ_PER_MOL;

  sp.grads.resize(3, frame.num_atoms());
  xtb_getGradient(m_env, m_res, sp.grads.data());
  sp.grads *= FORCE_SCALE;
  this->check_errors();

  // NOTE: only some models have virial
  // xtb_getVirial(m_env, m_res, sp.virial.data());
  // sp.virial *= occ::units::AU_TO_KJ_PER_MOL;
  // this->check_errors();

  return sp;
};

occ::Mat XTBModel::hessian(const core::Frame &frame, Type model) {
  int num_particles = frame.num_atoms();
  occ::Mat hessian(num_particles * 3, num_particles * 3);
  if (m_previous_calc != Calc::Hessian) {
    this->free_all();
    this->initialise();
    m_previous_calc = Calc::Hessian;
  }

  this->initialise_molecule(frame, model);

  xtb_hessian(m_env, m_mol, m_calc, m_res, hessian.data(), NULL, NULL, NULL,
              NULL);
  hessian *= HESS_SCALE;
  this->check_errors();
  return hessian;
};

} // namespace trajan::energy
