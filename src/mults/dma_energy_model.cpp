#include <algorithm>
#include <occ/core/units.h>
#include <occ/dma/dma.h>
#include <occ/interaction/wavefunction_transform.h>
#include <occ/mults/dma_energy_model.h>
#include <occ/mults/rotation.h>

namespace occ::mults {

using occ::interaction::CEEnergyComponents;

DMAMonomer DMAMonomer::from_wavefunction(const occ::qm::Wavefunction &wfn,
                                         const occ::dma::DMASettings &settings) {
  occ::dma::DMACalculator calc(wfn);
  calc.update_settings(settings);
  // Keep in sync with the H defaults in src/driver/dma_driver.cpp.
  calc.set_radius_for_element(1, 0.35);
  calc.set_limit_for_element(1, 1);

  auto result = calc.compute_multipoles();
  const auto &sites = calc.sites();

  DMAMonomer mon;
  mon.wfn = wfn;
  mon.reference.multipoles = std::move(result.multipoles);
  mon.reference.positions = sites.positions; // Bohr
  mon.reference.atomic_numbers.resize(sites.size());
  for (int s = 0; s < sites.size(); ++s)
    mon.reference.atomic_numbers[s] =
        sites.atoms[sites.atom_indices(s)].atomic_number;

  // NEIGHCRYS atom types for the typed exp-6 sets (FIT / W99); typing is
  // rotation-invariant so it carries to every oriented crystal image.
  std::vector<Vec3> pos_ang(sites.size());
  for (int s = 0; s < sites.size(); ++s)
    pos_ang[s] = mon.reference.positions.col(s) * occ::units::BOHR_TO_ANGSTROM;
  mon.reference.type_codes = ForceFieldParams::classify_atom_types(
      mon.reference.atomic_numbers, pos_ang);
  return mon;
}

namespace {

// Orient a reference-frame multipole by the cartesian symmetry operation R.
// Wigner-D rotation only handles proper rotations, so an improper op
// (det = -1) is decomposed as the proper rotation -R followed by inversion,
// which flips every odd-rank multipole by (-1)^l.
occ::dma::Mult orient_multipole(const occ::dma::Mult &m, const Mat3 &R) {
  if (R.determinant() > 0.0)
    return rotated_multipole(m, R);
  occ::dma::Mult out = rotated_multipole(m, Mat3(-R)); // -R is a proper rotation
  for (int l = 1; l <= out.max_rank; l += 2) {
    const int start = l * l, n = 2 * l + 1; // rank-l block in (l,m) layout
    out.q.segment(start, n) *= -1.0;
  }
  return out;
}

} // namespace

DMAExp6EnergyModel::DMAExp6EnergyModel(const occ::crystal::Crystal &crystal,
                                      std::vector<DMAMonomer> monomers,
                                      ForceFieldParams ff,
                                      MultipoleInteractions::Config elec_config)
    : m_crystal(crystal), m_monomers(std::move(monomers)), m_ff(std::move(ff)),
      m_elec_config(elec_config) {
  // Ensure the electrostatic evaluator covers the DMA rank present in the
  // data; never lowers a caller-supplied rank (truncate via
  // max_interaction_rank instead).
  int data_rank = 0;
  for (const auto &mon : m_monomers)
    for (const auto &mp : mon.reference.multipoles)
      data_rank = std::max(data_rank, mp.max_rank);
  m_elec_config.max_rank = std::max(m_elec_config.max_rank, data_rank);
}

const MoleculeMultipoles &
DMAExp6EnergyModel::oriented(const occ::core::Molecule &mol) const {
  const auto &shift = mol.cell_shift();
  const std::array<int, 4> key{mol.unit_cell_molecule_idx(), shift(0), shift(1),
                               shift(2)};
  {
    std::lock_guard<std::mutex> lock(m_oriented_mutex);
    auto it = m_oriented_cache.find(key);
    if (it != m_oriented_cache.end())
      return it->second;
  }

  // Compute outside the lock; concurrent misses on the same key are merely
  // redundant, not incorrect.
  const auto &mon = m_monomers.at(mol.asymmetric_molecule_idx());
  auto t = occ::interaction::transform::WavefunctionTransformer::
      calculate_transform(mon.wfn, mol, m_crystal);

  MoleculeMultipoles out;
  out.positions =
      (t.rotation * mon.reference.positions).colwise() + t.translation; // Bohr
  out.atomic_numbers = mon.reference.atomic_numbers;
  out.type_codes = mon.reference.type_codes; // typing is rotation-invariant
  out.multipoles.reserve(mon.reference.multipoles.size());
  for (const auto &mp : mon.reference.multipoles)
    out.multipoles.push_back(orient_multipole(mp, t.rotation));

  std::lock_guard<std::mutex> lock(m_oriented_mutex);
  return m_oriented_cache.emplace(key, std::move(out)).first->second;
}

CEEnergyComponents
DMAExp6EnergyModel::compute_energy(const occ::core::Dimer &dimer) {
  const auto &a = oriented(dimer.a());
  const auto &b = oriented(dimer.b());
  const auto e = dimer_interaction_energy(a, b, m_ff, m_elec_config);

  const double inv = 1.0 / occ::units::AU_TO_KJ_PER_MOL; // kJ/mol -> Hartree
  CEEnergyComponents c;
  c.coulomb = e.electrostatic * inv;
  c.repulsion = e.repulsion * inv;
  c.dispersion = e.dispersion * inv;
  c.exchange = 0.0;
  c.polarization = 0.0;
  c.total = c.coulomb + c.repulsion + c.dispersion;
  c.is_computed = true;
  return c;
}

void DMAExp6EnergyModel::compute_total_energy(CEEnergyComponents &c) const {
  c.total = c.coulomb + c.exchange + c.repulsion + c.polarization + c.dispersion;
}

Mat3N DMAExp6EnergyModel::compute_electric_field(const occ::core::Dimer &dimer) {
  // No induction in this model; crystal-field polarization is disabled.
  return Mat3N::Zero(3, dimer.a().size());
}

const std::vector<Vec> &DMAExp6EnergyModel::partial_charges() const {
  if (m_partial_charges.empty()) {
    m_partial_charges.reserve(m_monomers.size());
    for (const auto &mon : m_monomers) {
      const auto &mps = mon.reference.multipoles;
      Vec q(static_cast<Eigen::Index>(mps.size()));
      for (size_t s = 0; s < mps.size(); ++s)
        q(static_cast<Eigen::Index>(s)) =
            mps[s].q.size() > 0 ? mps[s].q(0) : 0.0;
      m_partial_charges.push_back(std::move(q));
    }
  }
  return m_partial_charges;
}

} // namespace occ::mults
