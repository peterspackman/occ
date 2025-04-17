#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/dma/dma.h>
#include <occ/dma/dmaql0.h>
#include <occ/dma/dmaqlm.h>
#include <occ/gto/density.h>
#include <occ/io/conversion.h>

namespace occ::dma {

DMACalculator::DMACalculator(const qm::Wavefunction &wfn)
    : m_basis(wfn.basis), m_mo(wfn.mo) {

  m_basis.set_pure(false);
  m_mo.to_cartesian(wfn.basis, m_basis);

  m_sites.atoms = wfn.atoms;
  m_sites.positions = wfn.positions();
  const auto N = m_sites.size();

  m_sites.atom_indices = IVec::LinSpaced(N, 0, N);
  m_sites.radii = Vec::Ones(N) * 0.65 * occ::units::ANGSTROM_TO_BOHR;
  m_sites.limits = IVec::Ones(N) * 4;

  log::info("Site positions\n{}\n", format_matrix(m_sites.positions));
  log::info("Site to atom map\n{}\n",
            format_matrix(m_sites.atom_indices, "{}"));
}

void DMACalculator::set_radius_for_element(int atomic_number,
                                           double radius_angs) {
  const double radius = radius_angs * occ::units::ANGSTROM_TO_BOHR;

  for (int site = 0; site < m_sites.size(); site++) {
    int atom_index = m_sites.atom_indices(site);
    if (atom_index < 0)
      continue;
    if (m_sites.atoms[atom_index].atomic_number == atomic_number) {
      m_sites.radii(site) = radius;
    }
  }
}

void DMACalculator::set_limit_for_element(int atomic_number, int limit) {

  for (int site = 0; site < m_sites.size(); site++) {
    int atom_index = m_sites.atom_indices(site);
    if (atom_index < 0)
      continue;
    if (m_sites.atoms[atom_index].atomic_number == atomic_number) {
      m_sites.limits(site) = limit;
    }
  }
}

void DMACalculator::update_settings(const DMASettings &settings) {
  m_settings = settings;
  m_sites.limits = m_sites.limits.cwiseMin(settings.max_rank);
}

DMAResult DMACalculator::compute_multipoles() {

  DMAResult result;
  result.max_rank = m_settings.max_rank;
  result.multipoles = dmaqlm(m_basis, m_mo, m_sites, m_settings);

  return result;
}

} // namespace occ::dma
