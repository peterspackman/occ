#pragma once
#include <array>
#include <map>
#include <mutex>
#include <occ/crystal/crystal.h>
#include <occ/dma/dma.h>
#include <occ/interaction/energy_model_base.h>
#include <occ/mults/dimer_interaction.h>
#include <occ/qm/wavefunction.h>
#include <vector>

namespace occ::mults {

/// Reference (monomer-frame) DMA data for one symmetry-unique molecule. The
/// wavefunction is retained only to recover the symmetry operation mapping the
/// reference onto each crystal image.
struct DMAMonomer {
  occ::qm::Wavefunction wfn;
  MoleculeMultipoles reference; ///< reference-frame DMA sites (Bohr)

  /// Run DMA on a converged monomer wavefunction and package the
  /// reference-frame sites.
  static DMAMonomer from_wavefunction(const occ::qm::Wavefunction &wfn,
                                      const occ::dma::DMASettings &settings = {});
};

/// EnergyModelBase using distributed multipoles + exp-6 (Williams), with no
/// induction. compute_energy() orients each molecule's reference multipoles
/// onto its crystal image (handling improper symmetry operations) and
/// evaluates dimer_interaction_energy().
class DMAExp6EnergyModel : public occ::interaction::EnergyModelBase {
public:
  DMAExp6EnergyModel(const occ::crystal::Crystal &crystal,
                     std::vector<DMAMonomer> monomers, ForceFieldParams ff,
                     MultipoleInteractions::Config elec_config = {});

  occ::interaction::CEEnergyComponents
  compute_energy(const occ::core::Dimer &dimer) override;
  Mat3N compute_electric_field(const occ::core::Dimer &dimer) override;
  const std::vector<Vec> &partial_charges() const override;
  double coulomb_scale_factor() const override { return 1.0; }
  void
  compute_total_energy(occ::interaction::CEEnergyComponents &c) const override;

private:
  /// Reference multipoles + positions oriented onto \p mol's crystal image,
  /// memoized per image (the WavefunctionTransform is the hot path). Returns a
  /// reference into the cache; std::map nodes are address-stable, so it stays
  /// valid under concurrent inserts.
  const MoleculeMultipoles &oriented(const occ::core::Molecule &mol) const;

  occ::crystal::Crystal m_crystal;
  std::vector<DMAMonomer> m_monomers;
  ForceFieldParams m_ff;
  MultipoleInteractions::Config m_elec_config;
  mutable std::vector<Vec> m_partial_charges;

  // Keyed by crystal image {uc_molecule_idx, cell_shift xyz}; compute_energy()
  // runs under parallel_for, so guard with the mutex.
  mutable std::map<std::array<int, 4>, MoleculeMultipoles> m_oriented_cache;
  mutable std::mutex m_oriented_mutex;
};

} // namespace occ::mults
