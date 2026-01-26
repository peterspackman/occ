#pragma once
#include <memory>
#include <occ/qm/integral_engine.h>
#include <occ/qm/mo.h>
#include <occ/qm/mo_integral_engine.h>
#include <occ/gto/shell.h>

namespace occ::qm {

class PostHFMethod {
public:
  PostHFMethod(const AOBasis &basis, const MolecularOrbitals &mo,
               double scf_energy);
  virtual ~PostHFMethod() = default;

  virtual double compute_correlation_energy() = 0;

  const MolecularOrbitals &molecular_orbitals() const { return m_mo; }
  const IntegralEngine &ao_engine() const { return *m_ao_engine; }
  const MOIntegralEngine &mo_engine() const { return *m_mo_engine; }

  double scf_energy() const { return m_scf_energy; }
  double correlation_energy() const { return m_correlation_energy; }
  double total_energy() const { return m_scf_energy + m_correlation_energy; }

protected:
  const MolecularOrbitals &m_mo;
  std::unique_ptr<IntegralEngine> m_ao_engine;
  std::unique_ptr<MOIntegralEngine> m_mo_engine;

  double m_scf_energy;
  double m_correlation_energy = 0.0;

  size_t n_occupied() const { return m_mo_engine->n_occupied(); }
  size_t n_virtual() const { return m_mo_engine->n_virtual(); }
  size_t n_ao() const { return m_mo_engine->n_ao(); }
};

} // namespace occ::qm