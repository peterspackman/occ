#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/dma/dma.h>
#include <occ/dma/multipole_calculator.h>
#include <occ/dma/multipole_shifter.h>
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
  for (int i = 0; i < m_sites.size(); i++) {
    m_sites.name.push_back(
        core::Element(m_sites.atoms[m_sites.atom_indices(i)].atomic_number)
            .symbol());
  }

  log::debug("Site positions\n{}\n", format_matrix(m_sites.positions));
  log::debug("Site to atom map\n{}\n",
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

  log::debug("Site limits: \n{}\n", format_matrix(m_sites.limits, "{}"));
  MultipoleCalculator calculator(m_basis, m_mo, m_sites, m_settings);
  result.multipoles = calculator.calculate();

  return result;
}

Mult DMACalculator::compute_total_multipoles(const DMAResult &result) const {
  DMASites total_sites;
  total_sites.positions = Mat3N::Zero(3, 1);
  total_sites.radii = Vec::Ones(1);
  total_sites.limits = IVec::Constant(1, result.max_rank);

  Vec3 origin(0, 0, 0);
  std::vector<Mult> total{Mult(result.max_rank)};
  std::vector<Mult> mults_result = result.multipoles;
  for (int i = 0; i < m_sites.size(); i++) {
    auto &m = mults_result[i];
    MultipoleShifter shifter(m_sites.positions.col(i), m, total_sites, total, result.max_rank);
    shifter.shift_multipoles(m, 0, m.max_rank, total[0], result.max_rank,
                             m_sites.positions.col(i) - origin);
  }
  return total[0];
}

} // namespace occ::dma
