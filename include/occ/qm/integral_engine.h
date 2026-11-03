#pragma once
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/ints/ecp.h>
#include <occ/ints/esp.h>
#include <occ/qm/cint_interface.h>
#include <occ/qm/expectation.h>
#include <occ/qm/mo.h>
#include <occ/qm/shellblock_norm.h>
#include <unsupported/Eigen/CXX11/Tensor>


namespace occ::qm {

using gto::Shell;
using gto::AOBasis;

struct JKPair {
  Mat J, K;
};

struct JKTriple {
  MatTriple J, K;
};

class IntegralEngine {
public:
  template <size_t num_centers> struct IntegralResult {
    int thread{0};
    std::array<int, num_centers> shell{0};
    std::array<int, num_centers> bf{0};
    std::array<int, num_centers> dims{0};
    const double *buffer{nullptr};
  };

  using ShellList = std::vector<Shell>;
  using AtomList = std::vector<occ::core::Atom>;
  using ShellPairList = std::vector<std::vector<size_t>>;
  using IntEnv = cint::IntegralEnvironment;
  using ShellKind = Shell::Kind;
  using Op = cint::Operator;

  // Tag to skip the O(nsh²) shellpair-overlap-norm probe in the constructor.
  // Use this for short-lived engines whose integrals are evaluated without
  // the shellpair list (e.g. the per-T merged-basis engines in the periodic
  // GFN2 path, where we extract a rectangular block rather than filtering by
  // significant shellpairs).
  struct NoShellPairs {};

  IntegralEngine(const AtomList &at, const ShellList &sh)
      : m_aobasis(at, sh), m_env(at, sh) {

    if (is_spherical()) {
      compute_shellpairs<ShellKind::Spherical>();
    } else {
      compute_shellpairs<ShellKind::Cartesian>();
    }
  }

  IntegralEngine(const AtomList &at, const ShellList &sh, NoShellPairs)
      : m_aobasis(at, sh), m_env(at, sh) {}

  IntegralEngine(const AOBasis &basis)
      : m_aobasis(basis), m_env(basis.atoms(), basis.shells()) {

    if (is_spherical()) {
      compute_shellpairs<ShellKind::Spherical>();
    } else {
      compute_shellpairs<ShellKind::Cartesian>();
    }

    if (m_aobasis.have_ecps()) {
      set_effective_core_potentials(m_aobasis.ecp_shells(),
                                    m_aobasis.ecp_electrons());
    }
  }

  IntegralEngine(const AOBasis &basis, NoShellPairs)
      : m_aobasis(basis), m_env(basis.atoms(), basis.shells()) {
    if (m_aobasis.have_ecps()) {
      set_effective_core_potentials(m_aobasis.ecp_shells(),
                                    m_aobasis.ecp_electrons());
    }
  }

  inline auto nbf() const noexcept { return m_aobasis.nbf(); }
  inline auto nbf_aux() const noexcept { return m_auxbasis.nbf(); }
  inline auto nsh() const noexcept { return m_aobasis.nsh(); }
  inline auto nsh_aux() const noexcept { return m_auxbasis.nsh(); }
  inline const AOBasis &aobasis() const { return m_aobasis; }
  inline const AOBasis &auxbasis() const { return m_auxbasis; }
  inline const auto &env() const { return m_env; }
  inline auto &env() { return m_env; }

  inline const auto &first_bf() const noexcept { return m_aobasis.first_bf(); }
  inline const auto &first_bf_aux() const noexcept {
    return m_auxbasis.first_bf();
  }
  inline const auto &shellpairs() const noexcept { return m_shellpairs; }
  inline const auto &shells() const noexcept { return m_aobasis.shells(); }

  inline void set_auxiliary_basis(const ShellList &bs, bool dummy = false) {
    if (!dummy) {
      clear_auxiliary_basis();
      m_auxbasis = AOBasis(m_aobasis.atoms(), bs);
      ShellList combined = m_aobasis.shells();
      combined.insert(combined.end(), m_auxbasis.shells().begin(),
                      m_auxbasis.shells().end());
      m_env = IntEnv(m_aobasis.atoms(), combined);
      apply_ecp_effective_charges();
    } else {
      AtomList dummy_atoms;
      dummy_atoms.reserve(bs.size());
      for (const auto &shell : bs) {
        dummy_atoms.push_back(
            {0, shell.origin(0), shell.origin(1), shell.origin(2)});
      }
      set_dummy_basis(dummy_atoms, bs);
    }
  }

  inline void set_dummy_basis(const AtomList &dummy_atoms,
                              const ShellList &bs) {
    clear_auxiliary_basis();
    m_auxbasis = AOBasis(dummy_atoms, bs);
    AtomList combined_sites = m_aobasis.atoms();
    combined_sites.insert(combined_sites.end(), dummy_atoms.begin(),
                          dummy_atoms.end());
    ShellList combined = m_aobasis.shells();
    combined.insert(combined.end(), m_auxbasis.shells().begin(),
                    m_auxbasis.shells().end());
    m_env = IntEnv(combined_sites, combined);
    apply_ecp_effective_charges();
  }

  inline void clear_auxiliary_basis() {
    m_auxbasis = AOBasis();
    m_env = IntEnv(m_aobasis.atoms(), m_aobasis.shells());
    apply_ecp_effective_charges();
  }

  inline bool have_auxiliary_basis() const noexcept {
    return m_auxbasis.nsh() > 0;
  }

  inline bool have_effective_core_potentials() const noexcept {
    return m_have_ecp;
  }

  Mat one_electron_operator(Op op, bool use_shellpair_list = true) const;

  MatTriple one_electron_operator_grad(Op op,
                                       bool use_shellpair_list = true) const;

  /// Multipole AO matrices with the gradient operator on the ket. For
  /// `op == Op::dipole` libcint computes <φ_μ | r_α | ∇_β φ_ν> (9
  /// components: α outer, β inner). For `op == Op::quadrupole` it computes
  /// <φ_μ | r_α r_β | ∇_γ φ_ν> (27 components: row-major (α,β) outer, γ
  /// inner). Returns one MatTriple per multipole component (3 for dipole,
  /// 9 for quadrupole), with the inner MatTriple holding the 3 spatial
  /// derivative components.
  ///
  /// Use the chain rule
  ///   ∂M_α(μ, ν, O=0)/∂R_(atom of ν)_β = - <φ_μ | M_α | ∂_β φ_ν>
  /// to convert these "ket-side" integrals to per-atom Cartesian gradients.
  std::vector<MatTriple>
  multipole_operator_grad(Op op, bool use_shellpair_list = true) const;

  MatSix one_electron_operator_hess(Op op,
                                    bool use_shellpair_list = true) const;

  Mat effective_core_potential(bool use_shellpair_list = true) const;

  /// Nuclear gradient of Tr(D V_ecp), per atom (3 x natoms). Zero when there
  /// are no ECPs.
  Mat3N effective_core_potential_gradient(const Mat &D,
                                          bool use_shellpair_list = true) const;

  /// Skeleton ECP Hessian sum P d2 V_ecp / dR dR (3N x 3N), P the total
  /// density. Zero when there are no ECPs.
  Mat effective_core_potential_hessian(const Mat &P) const;

  /// Add dV_ecp/dR for every nuclear coordinate into H1 (3N matrices).
  void add_effective_core_potential_derivatives(std::vector<Mat> &H1) const;

  /// Nuclear charge each atom's operators carry: Z minus its ECP electrons.
  double effective_nuclear_charge(size_t atom) const {
    const auto &e = m_aobasis.ecp_electrons();
    return m_aobasis.atoms()[atom].atomic_number -
           (atom < e.size() ? e[atom] : 0);
  }

  Mat fock_operator(SpinorbitalKind, const MolecularOrbitals &mo,
                    const Mat &Schwarz = Mat()) const;
  Mat fock_operator_mixed_basis(const Mat &D, const AOBasis &D_bs,
                                bool is_shell_diagonal);

  MatTriple fock_operator_grad(SpinorbitalKind, const MolecularOrbitals &mo,
                               const Mat &Schwarz = Mat()) const;

  MatSix fock_operator_hess(SpinorbitalKind, const MolecularOrbitals &mo,
                            const Mat &Schwarz = Mat()) const;

  Mat coulomb(SpinorbitalKind, const MolecularOrbitals &mo,
              const Mat &Schwarz = Mat()) const;

  MatTriple coulomb_grad(SpinorbitalKind, const MolecularOrbitals &mo,
                         const Mat &Schwarz = Mat()) const;

  MatSix coulomb_hess(SpinorbitalKind, const MolecularOrbitals &mo,
                      const Mat &Schwarz = Mat()) const;

  MatSix exchange_hess(SpinorbitalKind, const MolecularOrbitals &mo,
                       const Mat &Schwarz = Mat()) const;

  JKTriple coulomb_exchange_grad(SpinorbitalKind, const MolecularOrbitals &mo,
                                 const Mat &Schwarz = Mat()) const;

  /// Skeleton one-electron Hessian (3N x 3N): sum P h^{xy} - sum Q S^{xy}
  /// for total density P and total energy-weighted density Q, including the
  /// nuclear-attraction operator-centre terms. No ECPs.
  Mat one_electron_hessian(const Mat &P, const Mat &Q) const;

  /// First-derivative AO matrices for every nuclear coordinate (index
  /// 3 * atom + xyz): S^{(x)} and h^{(x)} = T^{(x)} + V^{(x)}, the latter
  /// including the Hellmann-Feynman operator derivative. Full derivatives
  /// (bra and ket), symmetric.
  void one_electron_derivative_matrices(std::vector<Mat> &S1,
                                        std::vector<Mat> &H1) const;

  /// Skeleton two-electron Hessian (3N x 3N) of jf * E_J + kf * E_K at fixed
  /// density, contracted over unique quartets with second-derivative
  /// integrals; blocks of the omitted centre by translational invariance.
  Mat jk_hessian(const MolecularOrbitals &mo, double jf, double kf,
                 const Mat &Schwarz = Mat()) const;

  /// Skeleton two-electron Fock derivatives for every nuclear coordinate:
  /// jf J^{(x)}[P_total] - kf K^{(x)}[P_spin], derivative integrals with the
  /// density held fixed. Unrestricted: one per spin, alpha rows over beta
  /// rows.
  std::vector<Mat> jk_derivative_fock(const MolecularOrbitals &mo, double jf,
                                      double kf,
                                      const Mat &Schwarz = Mat()) const;

  /// Restricted two-electron response for a batch of symmetric per-spin
  /// densities: jf * 2 J[D1] - kf * K[D1] for each, the last
  /// `exchange_only` of them without the Coulomb term. One integral pass per
  /// chunk of densities, digested as small dense products over the whole
  /// chunk (density index fastest), so the cost per extra density is close to
  /// the arithmetic, not memory traffic. Used by CPHF.
  std::vector<Mat> coulomb_exchange_response(const std::vector<Mat> &D,
                                             double jf, double kf,
                                             const Mat &Schwarz = Mat(),
                                             size_t exchange_only = 0) const;

  /// Nuclear gradient (3 x natom) of the two-electron energy
  ///   jf * E_J[D] + kf * E_K[D]
  /// where E_K is the full exact-exchange energy (negative), so a global
  /// hybrid passes kf = its exact-exchange fraction and Coulomb-only passes
  /// kf = 0. Contracted inside the integral loop over unique shell quartets;
  /// nothing nbf x nbf is formed. Restricted and unrestricted use the
  /// contracted kernel, general spinorbitals the matrix path.
  Mat3N jk_gradient(const MolecularOrbitals &mo, double jf, double kf,
                    const Mat &Schwarz = Mat()) const;
  JKPair coulomb_and_exchange(SpinorbitalKind, const MolecularOrbitals &mo,
                              const Mat &Schwarz = Mat()) const;

  std::vector<Mat> coulomb_list(SpinorbitalKind,
                                const std::vector<MolecularOrbitals> &mos,
                                const Mat &Schwarz = Mat()) const;

  std::vector<JKPair>
  coulomb_and_exchange_list(SpinorbitalKind,
                            const std::vector<MolecularOrbitals> &mos,
                            const Mat &Schwarz = Mat()) const;

  Eigen::Tensor<double, 4>
  four_center_integrals_tensor(const Mat &Schwarz = Mat()) const;

  /// Unique AO 2-electron integrals as a packed symmetric matrix A(P,Q) =
  /// (μν|λσ), where P = canonical pair index of (μ≤ν) and Q of (λ≤σ). Built in a
  /// single AO pass (8-fold-symmetric storage: ~npair²/2 unique vs nao⁴), the
  /// GEMM-friendly form for in-core four-index transforms.
  Mat four_center_integrals_packed(const Mat &Schwarz = Mat()) const;

  /// AO-direct first-quarter MO transform. Returns the half-transformed
  /// integrals H(i, ν, ρ, σ) = Σ_μ C(μ,i) (μν|ρσ) for the coefficient block
  /// C (nbf x nblk), streaming shell quartets with an inlined contraction so
  /// the full N⁴ tensor is never materialized. Used by AO-direct correlation
  /// methods; the inner loop is templated (not a std::function) for speed.
  Eigen::Tensor<double, 4>
  ao_direct_half_transform(const Mat &C, const Mat &Schwarz = Mat()) const;

  // Helper function to access integrals with 8-fold symmetry from canonical
  // storage
  static double
  get_integral_8fold_symmetry(const Eigen::Tensor<double, 4> &tensor, size_t i,
                              size_t j, size_t k, size_t l, size_t n_ao);

  Mat point_charge_potential(const std::vector<occ::core::PointCharge> &charges,
                             double alpha = 1e16);

  Mat wolf_point_charge_potential(
      const std::vector<occ::core::PointCharge> &charges,
      const std::vector<double> &partial_charges, double alpha,
      double cutoff_radius);
  Vec electric_potential(const MolecularOrbitals &mo, const Mat3N &points);

  /// Compute electric potential at grid points using fast MMD implementation
  /// @param mo  Molecular orbitals (density matrix)
  /// @param points  Grid points [3 x npts]
  /// @return  Electric potential at each point [npts]
  Vec electric_potential_mmd(const MolecularOrbitals &mo, const Mat3N &points) const;

  // Point-charge integrals L^k = <mu| 1/|r - s_k| |nu> at `points` (3 x M),
  // exact point charges, contracted in the shell-pair loop. Where charges
  // move, `owner(k)` is the atom point k rides on (< 0: fixed in space).

  /// sum_{mu nu} D_{mu nu} L^k_{mu nu} for each symmetric D: M x D.size().
  Mat point_charge_potentials(const std::vector<Mat> &D,
                              const Mat3N &points) const;

  /// sum_k Q(k, v) L^k for each column v of Q (M x nv): nv matrices.
  std::vector<Mat> point_charge_matrices(const Mat &Q,
                                         const Mat3N &points) const;

  /// d/dx sum_{mu nu} D_{mu nu} L^k_{mu nu} for every nuclear coordinate x
  /// (row 3 * atom + xyz) and point k (column): 3N x M, D symmetric.
  Mat point_charge_potential_derivatives(const Mat &D, const Mat3N &points,
                                         const IVec &owner) const;

  /// sum_k q_k sum_{mu nu} D_{mu nu} d2 L^k_{mu nu} / dx dy: 3N x 3N.
  Mat point_charge_hessian(const Mat &D, const Vec &q, const Mat3N &points,
                           const IVec &owner) const;

  /// d/dx sum_k q_k L^k for every nuclear coordinate x: 3N symmetric
  /// matrices, basis-function and charge-position derivatives together.
  std::vector<Mat> point_charge_derivative_matrices(const Vec &q,
                                                    const Mat3N &points,
                                                    const IVec &owner) const;

  template <ShellKind kind>
  inline void compute_shellpairs(double threshold = 1e-12) {
    occ::log::debug("computing shellpairs (threshold = {}, kind = {}",
                    threshold,
                    kind == ShellKind::Cartesian ? "cartesian" : "pure");
    constexpr auto op = Op::overlap;
    const auto nsh = m_aobasis.size();
    size_t num_significant_shellpairs{0}, num_total_shellpairs{0};
    m_shellpairs.resize(nsh);
    auto buffer = std::make_unique<double[]>(m_env.buffer_size_1e(op));
    for (int p = 0; p < nsh; p++) {
      auto &plist = m_shellpairs[p];
      for (int q = 0; q <= p; q++) {
        num_total_shellpairs++;
        if (m_aobasis.shells_share_origin(p, q)) {
          num_significant_shellpairs++;
          plist.push_back(q);
          continue;
        }
        std::array<int, 2> idxs{p, q};
        std::array<int, 2> dims = m_env.two_center_helper<op, kind>(
            idxs, nullptr, buffer.get(), nullptr);
        Eigen::Map<const occ::Mat> tmp(buffer.get(), dims[0], dims[1]);
        if (tmp.norm() >= threshold) {
          num_significant_shellpairs++;
          plist.push_back(q);
        }
      }
    }
    occ::log::debug("significant shellpairs = {} ({} total)",
                    num_significant_shellpairs, num_total_shellpairs);
  }

  Mat rinv_operator_atom_center(size_t atom_index,
                                bool use_shellpair_list = true) const;
  MatTriple rinv_operator_grad_atom(size_t atom_index,
                                    bool use_shellpair_list = true) const;

  MatSix rinv_operator_hess_atom(size_t atom_index,
                                 bool use_shellpair_list = true) const;

  Vec multipole(int order, const MolecularOrbitals &mo,
                const Vec3 &origin = {0, 0, 0}) const;

  /// AO matrices <mu| r_a - O_a |nu> for a = x, y, z (symmetric, nbf x nbf).
  /// The electric-field perturbation operator: an applied field F adds
  /// F . r to the one-electron Hamiltonian of an electron.
  MatTriple dipole_matrices(const Vec3 &origin = {0, 0, 0}) const;

  Mat schwarz() const;

  inline bool is_spherical() const {
    return m_aobasis.kind() == Shell::Kind::Spherical;
  }

  void set_effective_core_potentials(const ShellList &ecp_shells,
                                     const std::vector<int> &ecp_electrons);

  inline double range_separated_omega() const {
    return m_env.range_separated_omega();
  }

  inline void set_range_separated_omega(double omega) {
    m_env.set_range_separated_omega(omega);
  }

  inline void set_precision(double precision) { m_precision = precision; }
  inline double precision() const { return m_precision; }

private:
  /// Initialize ESP evaluator with shell pairs (lazy initialization)
  void ensure_esp_initialized() const;

  double m_precision{1e-12};
  AOBasis m_aobasis, m_auxbasis;
  ShellPairList m_shellpairs;
  // TODO remove mutable
  mutable IntEnv m_env;

  // Lazy-initialized ESP evaluator for fast MMD-based potential evaluation
  mutable std::unique_ptr<occ::ints::ESPEvaluator<double>> m_esp_evaluator;
  mutable bool m_esp_initialized{false};

  /// Re-apply the ECP-reduced nuclear charges to the integral environment.
  ///
  /// IntEnv is constructed with bare atomic numbers, and
  /// set_effective_core_potentials patches them to Z - n_ecp. Any code that
  /// rebuilds m_env (auxiliary-basis setup/teardown, dummy bases) silently
  /// throws that patch away, leaving libcint charging ECP atoms at their bare
  /// Z. The cached core Hamiltonian survives -- it is built before the guess
  /// -- but anything recomputed afterwards, the nuclear-attraction gradient
  /// especially, gets the wrong operator. No-op without ECPs.
  inline void apply_ecp_effective_charges() {
    if (!m_have_ecp)
      return;
    const auto &ats = m_aobasis.atoms();
    const auto &ecp_electrons = m_aobasis.ecp_electrons();
    // set_dummy_basis appends its sites *after* the real atoms, so real-atom
    // indices are unchanged in all three rebuild sites.
    for (size_t i = 0; i < ecp_electrons.size() && i < ats.size(); i++)
      m_env.set_atom_charge(static_cast<int>(i),
                            ats[i].atomic_number - ecp_electrons[i]);
  }

  bool m_have_ecp{false};
  std::shared_ptr<const occ::ints::EcpEvaluator> m_ecp_native;
};

/// Applies a range-separation parameter for the duration of a scope, and puts
/// back whatever was there before however the scope is left.
///
/// The engines carry the operator as mutable state, so a build that throws
/// part way through would otherwise leave omega set, and every later Fock
/// build would silently use the attenuated operator instead of the full
/// Coulomb one. Works on anything with the `range_separated_omega()` /
/// `set_range_separated_omega()` pair: `IntegralEngine`, `IntegralEngineDF`,
/// `HartreeFock`, `cint::IntegralEnvironment`.
template <typename T> class ScopedRangeSeparation {
public:
  ScopedRangeSeparation(T &target, double omega)
      : m_target(target), m_previous(target.range_separated_omega()) {
    m_target.set_range_separated_omega(omega);
  }
  ~ScopedRangeSeparation() { m_target.set_range_separated_omega(m_previous); }
  ScopedRangeSeparation(const ScopedRangeSeparation &) = delete;
  ScopedRangeSeparation &operator=(const ScopedRangeSeparation &) = delete;

private:
  T &m_target;
  double m_previous;
};
template <typename T>
ScopedRangeSeparation(T &, double) -> ScopedRangeSeparation<T>;

} // namespace occ::qm
