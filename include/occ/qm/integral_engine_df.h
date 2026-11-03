#pragma once
#include <Eigen/Cholesky>
#include <Eigen/IterativeLinearSolvers>
#include <occ/qm/coulomb_metric.h>
#include <occ/qm/df_packed_store.h>
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
  /// Nuclear gradient (3 x natom) of the RI-J Coulomb energy, derivative
  /// integrals only (the orbital response enters through the energy-weighted
  /// density as for any SCF gradient). Restricted and unrestricted.
  Mat3N coulomb_gradient(const MolecularOrbitals &mo);
  /// Nuclear gradient (3 x natom) of the RI-K exchange energy, derivative
  /// integrals only, from the occupied coefficients (mo.Cocc).
  Mat3N exchange_gradient(const MolecularOrbitals &mo);
  /// RI-K second derivatives of tr(D K) at fixed density (unrestricted:
  /// 1/2 sum_s tr(P^s K[P^s])), the skeleton derivatives of K (unrestricted:
  /// K[P^s] per spin, alpha rows over beta rows), and K[D1] for densities
  /// D1 = Cx C^T + C Cx^T given by their factors Cx (C the occupied
  /// coefficients).
  Mat exchange_hessian(const MolecularOrbitals &mo);
  std::vector<Mat> exchange_derivative_fock(const MolecularOrbitals &mo);
  /// exchange_derivative_fock contracted with the occupied coefficients
  /// (nbf x o each; unrestricted, alpha and beta side by side).
  std::vector<Mat> exchange_derivative_fock_occ(const MolecularOrbitals &mo);
  std::vector<Mat> exchange_response(const Mat &C, const std::vector<Mat> &Cx);
  /// exchange_response contracted with C, K[D1] C (nbf x o each): one
  /// integral pass for every density, nothing per density stored.
  std::vector<Mat> exchange_response_occ(const Mat &C,
                                         const std::vector<Mat> &Cx);
  /// RI-J second derivatives at fixed density (3N x 3N), the skeleton Fock
  /// derivatives d(J)/dX (3N matrices) and the J response to each density
  /// in D1 (the RI-J build on D1), all in the convention of coulomb().
  Mat coulomb_hessian(const MolecularOrbitals &mo);
  std::vector<Mat> coulomb_derivative_fock(const MolecularOrbitals &mo);
  std::vector<Mat> coulomb_response(const std::vector<Mat> &D1);
  JKPair coulomb_and_exchange(const MolecularOrbitals &mo);
  Mat fock_operator(const MolecularOrbitals &mo);

  /// Changing the policy drops a store built under the previous one.
  inline void set_integral_policy(Policy p) {
    if (p != m_policy)
      m_packed.clear();
    m_policy = p;
  }
  inline Policy integral_policy() const { return m_policy; }

  /// Bytes the packed 3-centre store may take under `Policy::Choose`; what
  /// does not fit is recomputed every build.
  inline void set_integral_store_memory_limit(size_t bytes) {
    if (bytes != m_integral_store_memory_limit)
      m_packed.clear();
    m_integral_store_memory_limit = bytes;
  }
  inline size_t integral_store_memory_limit() const {
    return m_integral_store_memory_limit;
  }
  /// The packed store the Coulomb and exchange builds use (empty until the
  /// first build under a storing policy).
  inline const PackedDFStore &packed_store() const { return m_packed; }

  /// Bytes a packed store of every screened shell pair would take.
  size_t packed_storage_bytes() const {
    return PackedDFStore::bytes_for(
        m_ao_engine.aobasis(), m_ao_engine.shellpairs(), m_aux_engine.nbf());
  }

  /// Makes the packed store complete if it is not already and the whole of it
  /// fits `max_bytes`, whatever the policy: for callers (the correlated
  /// methods) that make their own memory decision. False when it stays
  /// incomplete, including under a range-separated operator.
  bool ensure_complete_packed_store(size_t max_bytes);

  /// Set the method used for Coulomb matrix computation
  void set_coulomb_method(CoulombMethod method);
  /// Get the current Coulomb method
  inline CoulombMethod coulomb_method() const { return m_coulomb_method; }

  void set_range_separated_omega(double omega);
  inline double range_separated_omega() const { return m_omega; }
  void set_precision(double precision);
  inline double precision() const { return m_precision; };

  inline const IntegralEngine &ao_engine() const { return m_ao_engine; }
  inline const IntegralEngine &aux_engine() const { return m_aux_engine; }
  /// Factorization of the Coulomb metric V=(P|Q) for the active omega.
  inline const CoulombMetric &coulomb_metric() const { return V_LLt; }

  /// Integral-direct density-fitting B tensor for the given MO coefficient
  /// blocks: B(i*nR + a, P) = Σ_μν C_left(μ,i) C_right(ν,a) (μν|P), streamed
  /// without storing (μν|P). Returns (nL*nR x naux).
  Mat build_b_direct(Eigen::Ref<const Mat> C_left,
                     Eigen::Ref<const Mat> C_right) const;

private:
  /// Builds the packed store on first use: every pair under `Stored`, what
  /// fits the memory limit under `Choose`. False when nothing is stored.
  bool ensure_packed_store();
  /// For builds that include exchange: only a complete store helps them, so
  /// under `Choose` a store that would not fit whole is not built for them.
  bool ensure_complete_store_for_exchange();
  void build_packed_store(size_t max_bytes);
  Mat packed_coulomb(const MolecularOrbitals &mo);
  Mat packed_exchange(const MolecularOrbitals &mo);

  inline bool use_stored_integrals() const {
    // The store holds one operator's three-centre integrals and does not
    // record which, and it is filled only once -- so reusing it under a
    // different operator would silently hand back full Coulomb integrals for
    // the attenuated one. It can therefore only ever serve `1/r`, whatever
    // the policy says. This is a property of the store, not a policy
    // decision, which is why it is stated here rather than by overwriting the
    // caller's policy when omega changes.
    if (m_omega != 0.0)
      return false;
    return m_policy != Policy::Direct;
  }

  double m_precision{1e-12};

  mutable IntegralEngine m_ao_engine;  // engine with ao basis & aux basis
  mutable IntegralEngine m_aux_engine; // engine with just aux basis
  CoulombMetric V_LLt;        ///< metric factorization for the active omega
  CoulombMetric m_V_LLt_full; ///< cached omega = 0 factorization
  Vec m_V_diag_sqrt;          ///< sqrt(V_PP), omega = 0 (split RI-J screening)
  CoulombMetric m_V_LLt_lr;   ///< cached factorization for m_lr_omega
  double m_omega{0.0};          ///< omega the active factorization belongs to
  double m_lr_omega{0.0};       ///< omega of the cached long-range metric
  PackedDFStore m_packed;
  /// Natively, store the 3-centre integrals when they fit the limit below
  /// (otherwise they are recomputed every SCF iteration). In the browser,
  /// direct: a failed allocation of the store cannot be recovered from, and
  /// the heap is capped at 1 GB. Ask for `Stored` or `Direct` explicitly to
  /// override either.
#ifdef __EMSCRIPTEN__
  Policy m_policy{Policy::Direct};
#else
  Policy m_policy{Policy::Choose};
#endif
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
