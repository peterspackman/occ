#include <occ/core/log.h>
#include <occ/qm/post_hf_method.h>

namespace occ::qm {

PostHFMethod::PostHFMethod(const AOBasis &basis, const MolecularOrbitals &mo,
                           double scf_energy)
    : m_mo(mo), m_scf_energy(scf_energy) {

  // Create integral engine for this basis
  m_ao_engine = std::make_unique<IntegralEngine>(basis);

  // Create MO integral engine
  m_mo_engine = std::make_unique<MOIntegralEngine>(*m_ao_engine, m_mo);

  occ::log::debug("PostHFMethod initialized with SCF energy: {:.10f}",
                  m_scf_energy);
  occ::log::debug("Molecular orbitals: {} occupied, {} virtual", n_occupied(),
                  n_virtual());
}

} // namespace occ::qm