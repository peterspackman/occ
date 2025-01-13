#include <occ/cg/solvation_contribution.h>
#include <stdexcept>

namespace occ::cg {

double ContributionPair::total() const { return forward + reverse; }

void ContributionPair::exchange_with(ContributionPair &other) {
  other.reverse = forward;
  reverse = other.forward;
}

void SolvationContribution::add_coulomb(double value, bool is_forward) {
  if (is_forward)
    m_coulomb.forward += value;
  else
    m_coulomb.reverse += value;
}

void SolvationContribution::add_cds(double value, bool is_forward) {
  if (is_forward)
    m_cds.forward += value;
  else
    m_cds.reverse += value;
}

void SolvationContribution::add_coulomb_area(double value, bool is_forward) {
  if (is_forward)
    m_coulomb_area.forward += value;
  else
    m_coulomb_area.reverse += value;
}

void SolvationContribution::add_cds_area(double value, bool is_forward) {
  if (is_forward)
    m_cds_area.forward += value;
  else
    m_cds_area.reverse += value;
}

double SolvationContribution::total_energy() const {
  double total = m_coulomb.total() + m_cds.total();
  if (!m_antisymmetrize)
    return total;

  // Account for asymmetric contributions
  double asymmetric_difference =
      m_coulomb.forward + m_cds.forward - m_coulomb.reverse - m_cds.reverse;
  return total + 0.5 * asymmetric_difference;
}

void SolvationContribution::exchange_with(SolvationContribution &other) {
  if (m_exchanged || other.m_exchanged) {
    throw std::runtime_error(
        "Attempting to exchange already processed contributions");
  }
  m_coulomb.exchange_with(other.m_coulomb);
  m_cds.exchange_with(other.m_cds);
  m_coulomb_area.exchange_with(other.m_coulomb_area);
  m_cds_area.exchange_with(other.m_cds_area);
  m_exchanged = true;
  other.m_exchanged = true;
}

} // namespace occ::cg
