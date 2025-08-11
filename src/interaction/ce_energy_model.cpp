#include <occ/interaction/ce_energy_model.h>
#include <occ/interaction/wavefunction_transform.h>
#include <occ/interaction/polarization.h>
#include <occ/qm/chelpg.h>

namespace occ::interaction {

CEEnergyModel::CEEnergyModel(const crystal::Crystal &crystal,
                             const std::vector<Wavefunction> &wfns_a,
                             const std::vector<Wavefunction> &wfns_b)
    : m_crystal(crystal), m_model_params(ce_model_from_string(m_model_name)),
      m_wavefunctions_a(wfns_a),
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

  occ::log::trace("DIMER_DEBUG: Computing energy for dimer A(asym={},unit={}) <-> B(asym={},unit={})", 
                  mol_A.asymmetric_molecule_idx(), mol_A.unit_cell_molecule_idx(),
                  mol_B.asymmetric_molecule_idx(), mol_B.unit_cell_molecule_idx());

  Wavefunction A = m_wavefunctions_a[mol_A.asymmetric_molecule_idx()];
  Wavefunction B = m_wavefunctions_b[mol_B.asymmetric_molecule_idx()];

  A = prepare_wavefunction(mol_A, A);
  B = prepare_wavefunction(mol_B, B);

  CEModelInteraction interaction(m_model_params);

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
  return m_model_params.coulomb;
}

void CEEnergyModel::compute_total_energy(CEEnergyComponents &components) const {
  components.total = m_model_params.scaled_total(
    components.coulomb, components.exchange, components.repulsion, 
    components.polarization, components.dispersion);
}

Mat3N CEEnergyModel::compute_total_electric_field_from_neighbors(
    const core::Molecule &target_molecule,
    const std::vector<core::Dimer> &neighbor_dimers) {
  
  Mat3N total_field = Mat3N::Zero(3, target_molecule.size());
  
  for (const auto &dimer : neighbor_dimers) {
    // Find which molecule is the neighbor (not the target)
    const core::Molecule *neighbor_mol = nullptr;
    
    if (dimer.a().unit_cell_molecule_idx() == target_molecule.unit_cell_molecule_idx()) {
      neighbor_mol = &dimer.b();
    } else if (dimer.b().unit_cell_molecule_idx() == target_molecule.unit_cell_molecule_idx()) {
      neighbor_mol = &dimer.a();
    } else {
      continue; // Skip dimers that don't involve the target molecule
    }
    
    // Always create dimer with target as a() and neighbor as b()
    core::Dimer field_dimer(target_molecule, *neighbor_mol);
    Mat3N neighbor_field = compute_electric_field(field_dimer);
    total_field += neighbor_field;
  }
  
  return total_field;
}

double CEEnergyModel::compute_crystal_field_polarization_energy(
    const core::Molecule &molecule,
    const Mat3N &crystal_field) const {
  
  Wavefunction wfn = m_wavefunctions_a[molecule.asymmetric_molecule_idx()];
  wfn = prepare_wavefunction(molecule, wfn);
  
  double e_pol = 0.0;
  if (wfn.have_xdm_parameters) {
    e_pol = polarization_energy(wfn.xdm_polarizabilities, crystal_field);
  } else {
    bool is_charged = (wfn.atoms.size() == 1) && (wfn.charge() != 0);
    e_pol = ce_model_polarization_energy(wfn.atomic_numbers(), crystal_field, is_charged);
  }
  
  // Debug logging for crystal field comparison
  auto field_norms = crystal_field.colwise().norm();
  occ::log::trace("CRYSTAL_POL_DEBUG: Mol asym_idx={}, atoms={}, max_field={:.6f} au, avg_field={:.6f} au, e_pol={:.6f} au", 
                  molecule.asymmetric_molecule_idx(), wfn.atomic_numbers().size(), 
                  field_norms.maxCoeff(), field_norms.mean(), e_pol);
  
  return e_pol;
}

Vec CEEnergyModel::get_polarizabilities(const core::Molecule &molecule) const {
  Wavefunction wfn = m_wavefunctions_a[molecule.asymmetric_molecule_idx()];
  wfn = prepare_wavefunction(molecule, wfn);
  
  if (wfn.have_xdm_parameters) {
    return wfn.xdm_polarizabilities;
  } else {
    Vec polarizabilities(wfn.atomic_numbers().size());
    bool is_charged = (wfn.atoms.size() == 1) && (wfn.charge() != 0);
    
    const auto &pol_table = is_charged ? 
      occ::interaction::Charged_atomic_polarizibility : 
      occ::interaction::Thakkar_atomic_polarizability;
    
    for (int i = 0; i < wfn.atomic_numbers().size(); ++i) {
      polarizabilities(i) = pol_table[wfn.atomic_numbers()(i) - 1];
    }
    return polarizabilities;
  }
}


} // namespace occ::interaction
