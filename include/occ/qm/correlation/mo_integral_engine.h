#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/mo.h>

namespace occ::qm {

/// Lightweight holder of molecular-orbital information for post-HF methods,
/// plus a brute-force single MO electron-repulsion integral evaluator kept as a
/// reference/testing utility. The performance-critical AO->MO transforms live
/// in the individual correlation methods (e.g. AO-direct conventional MP2 and
/// the density-fitted DFIntegrals B-tensor).
class MOIntegralEngine {
public:
  explicit MOIntegralEngine(const IntegralEngine &ao_engine,
                            const MolecularOrbitals &mo);

  /// Brute-force chemist-notation (ij|kl) in the MO basis. O(nshell^4) per
  /// call; intended for tests and small reference checks only.
  double compute_mo_eri(size_t i, size_t j, size_t k, size_t l) const;

  size_t n_occupied() const { return m_n_occ; }
  size_t n_virtual() const { return m_n_virt; }
  size_t n_ao() const { return m_mo.n_ao; }

  const MolecularOrbitals &molecular_orbitals() const { return m_mo; }
  const IntegralEngine &ao_engine() const { return m_ao_engine; }

private:
  void setup_mo_coefficients();

  const IntegralEngine &m_ao_engine;
  const MolecularOrbitals &m_mo;
  size_t m_n_occ{0};
  size_t m_n_virt{0};
};

} // namespace occ::qm
