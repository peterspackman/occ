#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/dma/dma.h>
#include <occ/gto/density.h>
#include <occ/dma/dmaql0.h>
#include <occ/dma/dmaqlm.h>

namespace occ::dma {

DMACalculator::DMACalculator(const qm::Wavefunction &wfn) : m_atoms(wfn.atoms),
            m_site_positions(wfn.positions()), 
            m_atom_indices(IVec::LinSpaced(wfn.atoms.size(), 0, wfn.atoms.size())),
            m_basis(wfn.basis), m_mo(wfn.mo)
{

  log::info("Site positions\n{}\n", format_matrix(m_site_positions));
  log::info("Site to atom map\n{}\n", format_matrix(m_atom_indices, "{}"));

}


DMAResult DMACalculator::compute_multipoles() {

  DMAResult result;
  result.max_rank = m_settings.max_rank;

  return result;

}

} // namespace occ::dma
