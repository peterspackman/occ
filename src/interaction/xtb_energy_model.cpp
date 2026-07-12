#include <occ/interaction/xtb_energy_model.h>
#include <occ/xtb/xtb_calculator.h>

namespace occ::interaction {

// Uses the in-process XtbCalculator; the external-process wrapper exchanges
// fixed-name files in the cwd, which race under parallel dimer evaluation.
XTBEnergyModel::XTBEnergyModel(const crystal::Crystal &crystal)
    : m_crystal(crystal) {

  for (const auto &mol : crystal.symmetry_unique_molecules()) {
    occ::xtb::XtbCalculator calc(mol);
    m_monomer_energies.push_back(calc.single_point_energy());
    m_partial_charges.push_back(calc.charges());
  }
}

CEEnergyComponents XTBEnergyModel::compute_energy(const core::Dimer &dimer) {
  core::Molecule mol_A = dimer.a();
  core::Molecule mol_B = dimer.b();

  occ::xtb::XtbCalculator calc_AB(dimer);
  double e_a = m_monomer_energies[mol_A.asymmetric_molecule_idx()];
  double e_b = m_monomer_energies[mol_B.asymmetric_molecule_idx()];
  double e_ab = calc_AB.single_point_energy();

  CEEnergyComponents result;
  result.total = e_ab - e_a - e_b;
  result.is_computed = true;
  return result;
}

Mat3N XTBEnergyModel::compute_electric_field(const core::Dimer &) {
  // XTB doesn't provide electric field calculations
  return Mat3N::Zero(3, 1);
}

const std::vector<Vec> &XTBEnergyModel::partial_charges() const {
  return m_partial_charges;
}

} // namespace occ::interaction
