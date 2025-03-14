#pragma once
#include "dftd_dispersion.h"
#include <occ/core/molecule.h>

namespace occ::disp {

class D4Dispersion {
public:
  D4Dispersion(const occ::core::Molecule &);
  D4Dispersion(const std::vector<occ::core::Atom> &atoms);

  bool set_functional(const std::string &functional);
  inline void set_parameters(const dftd4::dparam &parameters) {
    m_parameters = parameters;
  }

  double energy() const;
  std::pair<double, Mat3N> energy_and_gradient() const;

  inline void set_charge(int charge) { m_charge = charge; }

  void set_atoms(const std::vector<occ::core::Atom> &atoms);

private:
  int m_charge{0};

  // parameters for pbe
  dftd4::dparam m_parameters{
      1.0,         // s6
      0.95948085,  // s8
      0.0,         // s10
      1.0,         // s9
      0.38574991,  // a1
      4.80688534,  // a2
      16,          // alp
  };

  dftd4::TMolecule m_tmol;
  dftd4::TCutoff m_cutoff;
  dftd4::TD4Model m_d4;
};

} // namespace occ::disp
