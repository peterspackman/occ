#pragma once
#include <Eigen/Cholesky>
#include <Eigen/IterativeLinearSolvers>
#include <occ/qm/coulomb_metric.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/split_ri_j.h>

namespace occ::qm {

/// Method for computing Coulomb matrix in density fitting
enum class CoulombMethod {
    Traditional,  ///< libcint 3-center integrals (default)
    SplitRIJ      ///< MMD kernels using Hermite basis (no atomics)
};

using gto::Shell;
using gto::AOBasis;

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

  /// Set the method used for Coulomb matrix computation
  void set_coulomb_method(CoulombMethod method);
  /// Get the current Coulomb method
  inline CoulombMethod coulomb_method() const { return m_coulomb_method; }

  void set_range_separated_omega(double omega);
  inline double range_separated_omega() const { return m_omega; }
  void set_precision(double precision);
  inline double precision() const { return m_precision; };

  // Access to engines and stored integrals
  inline const IntegralEngine &ao_engine() const { return m_ao_engine; }
  inline const IntegralEngine &aux_engine() const { return m_aux_engine; }
  inline const Mat &integral_store() const { return m_integral_store; }
  /// Factorization of the Coulomb metric V=(P|Q) for the active omega.
  inline const CoulombMetric &coulomb_metric() const { return V_LLt; }
  void compute_stored_integrals();

  /// Integral-direct density-fitting B tensor for the given MO coefficient
  /// blocks: B(i*nR + a, P) = Σ_μν C_left(μ,i) C_right(ν,a) (μν|P), streamed
  /// without materializing the dense (μν|P) store. Returns (nL*nR x naux).
  Mat build_b_direct(Eigen::Ref<const Mat> C_left,
                     Eigen::Ref<const Mat> C_right) const;

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

  /// Bytes `compute_stored_integrals` would allocate: the full nbf x nbf
  /// square for every auxiliary function.
  ///
  /// This has to mirror that allocation exactly. It previously counted
  /// screened shell pairs and returned a number of elements, which was then
  /// compared against a limit in bytes -- so the guard let through roughly
  /// sixteen times what it advertised, and the store blew up instead.
  size_t integral_storage_bytes() const {
    const size_t nbf = m_ao_engine.nbf();
    const size_t ndf = m_aux_engine.nbf();
    return nbf * nbf * ndf * sizeof(double);
  }

  inline bool use_stored_integrals() const {
    // The store holds one operator's three-centre integrals and does not
    // record which, and `compute_stored_integrals` fills it only once -- so
    // reusing it under a different operator would silently hand back full
    // Coulomb integrals for the attenuated one. It can therefore only ever
    // serve `1/r`, whatever the policy says. This is a property of the store,
    // not a policy decision, which is why it is stated here rather than by
    // overwriting the caller's policy when omega changes.
    if (m_omega != 0.0)
      return false;
    if (m_policy == Policy::Choose) {
      return (integral_storage_bytes() < m_integral_store_memory_limit);
    }
    return (m_policy == Policy::Stored);
  }

  double m_precision{1e-12};

  mutable IntegralEngine m_ao_engine;  // engine with ao basis & aux basis
  mutable IntegralEngine m_aux_engine; // engine with just aux basis
  CoulombMetric V_LLt;        ///< metric factorization for the active omega
  CoulombMetric m_V_LLt_full; ///< cached omega = 0 factorization
  CoulombMetric m_V_LLt_lr;   ///< cached factorization for m_lr_omega
  double m_omega{0.0};          ///< omega the active factorization belongs to
  double m_lr_omega{0.0};       ///< omega of the cached long-range metric
  Mat m_integral_store;
  /// Direct by default: recomputing the 3-centre integrals each iteration
  /// costs time, whereas storing them costs an allocation that scales as
  /// nbf^2 x ndf and cannot be recovered from when it fails -- which is what
  /// browsers, with a 1 GB heap, kept hitting. Ask for `Stored` explicitly
  /// when the memory is known to be there.
  Policy m_policy{Policy::Direct};
  /// Ceiling on the stored 3-centre integrals, only consulted by
  /// `Policy::Choose`. The browser gets a much smaller share: its whole heap
  /// is capped at 1 GB and the wavefunction, grids and Fock matrices have to
  /// live there too, so half a gigabyte of integrals is not affordable even
  /// when the allocation nominally succeeds.
#ifdef __EMSCRIPTEN__
  size_t m_integral_store_memory_limit{128 * 1024 * 1024}; // 128 MiB
#else
  size_t m_integral_store_memory_limit{512 * 1024 * 1024}; // 512 MiB
#endif

  // Coulomb method selection
  CoulombMethod m_coulomb_method{CoulombMethod::Traditional};
  mutable std::unique_ptr<SplitRIJ> m_split_rij;  // Lazy-initialized
};

} // namespace occ::qm
