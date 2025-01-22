#pragma once
#include <Eigen/IterativeLinearSolvers>
#include <occ/qm/integral_engine.h>

namespace occ::qm {

class IntegralEngineDF {
public:
  enum Policy { Choose, Direct, Stored };
  using ShellPairList = std::vector<std::vector<size_t>>;
  using ShellList = std::vector<Shell>;
  using AtomList = std::vector<occ::core::Atom>;
  using ShellKind = Shell::Kind;
  using Op = cint::Operator;
  using Buffer = std::vector<double>;
  using IntegralResult = IntegralEngine::IntegralResult<3>;

  IntegralEngineDF(const AtomList &atoms, const ShellList &ao,
                   const ShellList &df);

  Mat exchange(const MolecularOrbitals &mo);
  Mat coulomb(const MolecularOrbitals &mo);
  JKPair coulomb_and_exchange(const MolecularOrbitals &mo);
  Mat fock_operator(const MolecularOrbitals &mo);

  inline void set_integral_policy(Policy p) { m_policy = p; }
  inline Policy integral_policy() const { return m_policy; }

  void set_range_separated_omega(double omega);
  void set_precision(double precision);
  inline double precision() const { return m_precision; };

private:
  inline size_t num_rows() const {
    const auto &aobasis = m_ao_engine.aobasis();
    const auto &shellpairs = m_ao_engine.shellpairs();
    size_t n = 0;
    for (size_t s1 = 0; s1 < aobasis.size(); s1++) {
      size_t s1_size = aobasis[s1].size();
      size_t pairs_size = 0;
      for (const auto &s2 : shellpairs.at(s1)) {
        pairs_size += aobasis[s2].size();
      }
      n += s1_size * pairs_size;
    }
    return n;
  }

  size_t integral_storage_max_size() const {
    return m_ao_engine.auxbasis().nbf() * num_rows();
  }
  void compute_stored_integrals();

  inline bool use_stored_integrals() const {
    if (m_policy == Policy::Choose) {
      return (m_integral_store_memory_limit > integral_storage_max_size());
    }
    return (m_policy == Policy::Stored);
  }

  double m_precision{1e-12};

  mutable IntegralEngine m_ao_engine;  // engine with ao basis & aux basis
  mutable IntegralEngine m_aux_engine; // engine with just aux basis
  Eigen::LLT<Mat> V_LLt;
  Mat m_integral_store;
  Policy m_policy{Policy::Choose};
  size_t m_integral_store_memory_limit{512 * 1024 * 1024}; // 512 MiB
};

} // namespace occ::qm
