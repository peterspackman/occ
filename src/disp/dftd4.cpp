#include "dftd_damping.h"

#include <occ/disp/dftd4.h>
#include <occ/core/log.h>

namespace occ::disp {

D4Dispersion::D4Dispersion(const occ::core::Molecule &mol) {
  set_atoms(mol.atoms());
  set_charge(mol.charge());
}

D4Dispersion::D4Dispersion(const std::vector<occ::core::Atom> &atoms) {
  set_atoms(atoms);
}

void D4Dispersion::set_atoms(const std::vector<occ::core::Atom> &atoms) {
  m_tmol = {};
  m_tmol.GetMemory(atoms.size());
  for (int i = 0; i < atoms.size(); i++) {
    const auto &atom = atoms[i];
    m_tmol.CC(i, 0) = atom.x;
    m_tmol.CC(i, 1) = atom.y;
    m_tmol.CC(i, 2) = atom.z;
    m_tmol.ATNO(i) = atom.atomic_number;
  }
}

bool D4Dispersion::set_functional(const std::string &functional) {
  bool lmbd{true};
  int res = dftd4::d4par(functional, m_parameters, lmbd);

  occ::log::debug("D4 parameters for functional '{}': s6={:.6f}, s8={:.6f}, s10={:.6f}, s9={:.6f}, a1={:.6f}, a2={:.6f}, alp={}",
                  functional, m_parameters.s6, m_parameters.s8, m_parameters.s10,
                  m_parameters.s9, m_parameters.a1, m_parameters.a2, m_parameters.alp);

  return res == EXIT_SUCCESS;
}

double D4Dispersion::energy() const {
  occ::log::debug("D4 energy calculation: {} atoms, charge={}", m_tmol.NAtoms, m_charge);

  double energy{0.0};
  int info = dftd4::get_dispersion(m_tmol, m_charge, m_d4, m_parameters,
                                   m_cutoff, energy, nullptr);
  if (info != EXIT_SUCCESS) {
    throw std::runtime_error("Error running dftd4");
  }

  occ::log::debug("D4 dispersion energy: {:.12f} Ha", energy);

  return energy;
}

std::pair<double, Mat3N> D4Dispersion::energy_and_gradient() const {
  occ::log::debug("D4 gradient calculation: {} atoms, charge={}", m_tmol.NAtoms, m_charge);

  double energy{0.0};
  Mat3N gradient = Mat3N::Zero(3, m_tmol.NAtoms);
  int info = dftd4::get_dispersion(m_tmol, m_charge, m_d4, m_parameters,
                                   m_cutoff, energy, gradient.data());
  if (info != EXIT_SUCCESS) {
    throw std::runtime_error("Error running dftd4");
  }

  occ::log::debug("D4 dispersion energy: {:.12f} Ha", energy);
  occ::log::debug("D4 gradient norm: {:.12f}", gradient.norm());

  return {energy, gradient};
}

} // namespace occ::disp
