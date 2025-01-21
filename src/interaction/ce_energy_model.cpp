#include <occ/interaction/ce_energy_model.h>
#include <occ/interaction/wavefunction_transform.h>
#include <occ/qm/chelpg.h>

namespace occ::interaction {

CEEnergyModel::CEEnergyModel(const crystal::Crystal &crystal,
                             const std::vector<Wavefunction> &wfns_a,
                             const std::vector<Wavefunction> &wfns_b)
    : m_crystal(crystal), m_wavefunctions_a(wfns_a),
      m_wavefunctions_b(wfns_b.empty() ? wfns_a : wfns_b) {}

Wavefunction
CEEnergyModel::prepare_wavefunction(const core::Molecule &mol,
                                    const Wavefunction &wfn) const {

  auto transform = transform::WavefunctionTransformer::calculate_transform(
      wfn, mol, m_crystal);

  Wavefunction result = wfn;
  result.apply_transformation(transform.rotation, transform.translation);

  Mat3N pos_mol = mol.positions();
  Mat3N pos_transformed = result.positions() * units::BOHR_TO_ANGSTROM;

  occ::log::debug("Transformed wavefunction positions RMSD = {}",
                  (pos_transformed - pos_mol).norm());

  return result;
}

CEEnergyComponents CEEnergyModel::compute_energy(const core::Dimer &dimer) {
  core::Molecule mol_A = dimer.a();
  core::Molecule mol_B = dimer.b();

  Wavefunction A = m_wavefunctions_a[mol_A.asymmetric_molecule_idx()];
  Wavefunction B = m_wavefunctions_b[mol_B.asymmetric_molecule_idx()];

  A = prepare_wavefunction(mol_A, A);
  B = prepare_wavefunction(mol_B, B);

  auto model = ce_model_from_string(m_model_name);
  CEModelInteraction interaction(model);

  auto result = interaction(A, B);
  result.is_computed = true;

  occ::log::debug("Finished model energy calculation");
  return result;
}

Mat3N CEEnergyModel::compute_electric_field(const core::Dimer &dimer) {
  core::Molecule mol_A = dimer.a();
  core::Molecule mol_B = dimer.b();

  Wavefunction B = m_wavefunctions_b[mol_B.asymmetric_molecule_idx()];
  B = prepare_wavefunction(mol_B, B);

  Mat3N pos_A_bohr = mol_A.positions() * units::ANGSTROM_TO_BOHR;
  return B.electric_field(pos_A_bohr);
}

const std::vector<Vec> &CEEnergyModel::partial_charges() const {
  if (m_partial_charges.empty()) {
    m_partial_charges.reserve(m_wavefunctions_a.size());
    for (const auto &wfn : m_wavefunctions_a) {
      m_partial_charges.push_back(occ::qm::chelpg_charges(wfn));
    }
  }
  return m_partial_charges;
}

double CEEnergyModel::coulomb_scale_factor() const {
  auto model = ce_model_from_string(m_model_name);
  return model.coulomb;
}

} // namespace occ::interaction
