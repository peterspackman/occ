#include "dftd_damping.h"
#include "dftd_parameters.h"
#include <occ/disp/dftd4.h>

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
  return res == EXIT_SUCCESS;
}

double D4Dispersion::energy() const {
  double energy{0.0};
  int info = dftd4::get_dispersion(m_tmol, m_charge, m_d4, m_parameters,
                                   m_cutoff, energy, nullptr);
  if (info != EXIT_SUCCESS) {
    throw std::runtime_error("Error running dftd4");
  }
  return energy;
}

std::pair<double, Mat3N> D4Dispersion::energy_and_gradient() const {
  double energy{0.0};
  Mat3N gradient = Mat3N::Zero(3, m_tmol.NAtoms);
  int info = dftd4::get_dispersion(m_tmol, m_charge, m_d4, m_parameters,
                                   m_cutoff, energy, gradient.data());
  if (info != EXIT_SUCCESS) {
    throw std::runtime_error("Error running dftd4");
  }
  return {energy, gradient};
}

} // namespace occ::disp
