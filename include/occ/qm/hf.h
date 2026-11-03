#pragma once
#include <optional>
#include <array>
#include <occ/core/energy_components.h>
#include <occ/core/multipole.h>
#include <occ/core/point_charge.h>
#include <occ/numint/grid_settings.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/mo.h>
#include <occ/qm/scf_method.h>
#include <occ/qm/seminumerical_exchange.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

using occ::gto::AOBasis;
using occ::qm::MolecularOrbitals;
using PointChargeList = std::vector<occ::core::PointCharge>;
/// ORCA's COSX grid schedule: iterate on a coarse grid, switch to a finer one
/// once the SCF is close (DIIS error below refine_threshold, or converged on
/// the coarse grid), and evaluate the final exchange energy once on the final
/// grid. Gradients then use the final grid.
///
/// The default is for energies. Its final energy is the GRIDX3 functional at
/// orbitals converged on GRIDX2, which is not stationary on GRIDX3, so the
/// GRIDX3 gradient misses an orbital-response term. `for_gradients()` refines
/// on the final grid instead, so the SCF is converged where the gradient is
/// evaluated, as for a fixed GRIDX3 SCF.
struct COSXGridSchedule {
  occ::numint::COSXGridLevel iterate{occ::numint::COSXGridLevel::Grid1};
  occ::numint::COSXGridLevel refine{occ::numint::COSXGridLevel::Grid2};
  occ::numint::COSXGridLevel final{occ::numint::COSXGridLevel::Grid3};
  double refine_threshold{1e-3};
  /// On the refine grid, iterate with the iterate grid's exchange plus a
  /// correction K_refine(D) - K_iterate(D) frozen at the last exact build,
  /// refreshed whenever the SCF converges on it. The correction depends on D
  /// only through the grid error, so it changes little between iterations:
  /// the SCF converges to the refine grid's own solution with a few refine
  /// grid builds instead of one per iteration.
  bool frozen_correction{true};

  static COSXGridSchedule for_gradients() {
    return {occ::numint::COSXGridLevel::Grid1,
            occ::numint::COSXGridLevel::Grid3,
            occ::numint::COSXGridLevel::Grid3, 1e-3, true};
  }
};

class HartreeFock : public SCFMethodBase {
public:
  HartreeFock(const AOBasis &basis);
  inline const auto &aobasis() const { return m_engine.aobasis(); }
  inline auto nbf() const { return m_engine.nbf(); }

  bool usual_scf_energy() const { return true; }
  void update_scf_energy(occ::core::EnergyComponents &) const { return; }

  FockBuildProperties fock_build_properties() const {
    // COSX exchange is linear in the density on a fixed grid and its
    // charge x density screen is absolute, so a small difference density
    // drops most shell pairs. Grid changes and frozen-correction refreshes
    // force a full build (see the SCF hooks below).
    return {.linear_in_density = true,
            .constant_core_hamiltonian = true,
            // The conventional 4-centre build and the COSX exchange screen on
            // the density; RI-J/RI-K contract through the auxiliary basis.
            .density_screened = !m_df_engine || m_cosx_engine != nullptr};
  }

  inline bool have_effective_core_potentials() const {
    return m_engine.have_effective_core_potentials();
  }

  void set_density_fitting_basis(const std::string &, double auto_aux_threshold = 1e-4);
  void set_density_fitting_policy(IntegralEngineDF::Policy policy);
  void set_density_fitting_memory_limit(size_t bytes);
  void set_coulomb_method(CoulombMethod method);
  /// COSX exchange on one fixed grid.
  void set_cosx_exchange(occ::numint::COSXGridLevel level = occ::numint::COSXGridLevel::Grid1);
  void set_cosx_settings(const occ::qm::cosx::Settings &settings);

  /// The factor the caller applies to this exchange (a global hybrid's exact
  /// exchange fraction). The COSX shell-pair screen is loosened by it, so the
  /// absolute screening error in the Fock matrix stays that of HF.
  void set_cosx_exchange_scale(double scale);

  using COSXGridSchedule = occ::qm::COSXGridSchedule;
  void set_cosx_grid_schedule(const COSXGridSchedule &schedule = {});

  /// SCF hook: called after each iteration's DIIS step. Returns true when
  /// the procedure switched to a finer COSX grid, in which case the SCF must
  /// rebuild the Fock matrix in full and keep iterating.
  bool advance_scf_grid(double diis_error, bool converged);

  /// SCF hooks for the frozen grid correction: whether the last Fock build
  /// used it (so convergence there is not convergence on the refine grid),
  /// and a request that the next build be exact and refresh it.
  bool scf_fock_approximate() const { return m_cosx_last_approx; }
  void refresh_scf_fock_correction() { ++m_cosx_generation; }

  /// SCF hook: whether the next Fock build is handed a difference density.
  /// An incremental build leaves out the frozen grid correction (it is
  /// already in the accumulated Fock matrix) and must not refresh it.
  void set_scf_fock_incremental(bool incremental);

  /// SCF hook: the DIIS error below which difference-density builds stop.
  /// The COSX charge x density screen is absolute, so each difference build
  /// drops a different set of shell pairs and the accumulated exchange
  /// carries their error; below this the iterations are full builds, so the
  /// SCF converges onto the full build's own fixed point.
  double incremental_fock_floor() const {
    return m_cosx_engine ? 100.0 * m_cosx_engine->settings().f_threshold
                         : 0.0;
  }

  /// SCF hook: called once at convergence. With a COSX schedule, moves to the
  /// final grid and returns the change in the exact-exchange energy,
  /// -sum_t c_t [tr(D K_t,final) - tr(D K_t,current)] for the operators
  /// (c_t, omega_t) given (default: full HF exchange).
  double final_energy_correction(const MolecularOrbitals &mo);
  /// Exact-exchange operators (c_t, omega_t) in the energy; HF is {(1, 0)}.
  using ExchangeOperators = std::vector<std::pair<double, double>>;
  double finalize_cosx_grid(const MolecularOrbitals &mo,
                            const ExchangeOperators &operators);
  /// COSX exchange energy at mo on the current grid minus that on `other`,
  /// non-self-consistently, leaving the current grid in place. Used to put
  /// energies from SCFs on different grids on one surface.
  double cosx_energy_shift_from(occ::numint::COSXGridLevel other,
                                const MolecularOrbitals &mo,
                                const ExchangeOperators &operators = {
                                    {1.0, 0.0}});

  inline bool using_cosx() const { return m_cosx_engine != nullptr; }
  inline bool using_density_fitting() const { return m_df_engine != nullptr; }
  /// The exact-integral engine (one-electron and point-charge integrals).
  inline const IntegralEngine &integral_engine() const { return m_engine; }
  
  /**
   * @brief Create a new HartreeFock instance with the same settings but different basis
   * @param new_basis The new basis set to use
   * @return New HartreeFock instance
   */
  HartreeFock with_new_basis(const AOBasis &new_basis) const;

  /// Give `other` (typically the same system at another geometry) this
  /// object's two-electron treatment: density fitting (basis rebuilt for
  /// other's atoms), Coulomb method and storage policy, and COSX (fixed grid
  /// or schedule, and settings). Finite-difference derivatives need this to
  /// stay on one energy surface.
  void copy_two_electron_setup_to(HartreeFock &other) const;

  /// Factor each response density as Cx C^T + C Cx^T (no virtual-virtual
  /// block); throws otherwise.
  std::vector<Mat> factor_response_densities(const std::vector<Mat> &D1,
                                             const Mat &C) const;
  /// RI-J engine for approximate responses: the SCF's own when it fits J,
  /// else one built on def2-universal-jkfit (null if unavailable).
  IntegralEngineDF *approximate_coulomb_engine() const;
  /// The density-fitting engine for exchange with operator omega (the
  /// attenuated twin for omega != 0).
  IntegralEngineDF *exchange_df_engine(double omega) const;

  inline void set_precision(double precision) {
    m_engine.set_precision(precision);

    if (m_df_engine != nullptr) {
      m_df_engine->set_precision(precision);
    }
    // The attenuated-operator twins take their precision at construction, so
    // they would otherwise keep whatever was current when they were first
    // built and quietly disagree with the engines beside them.
    if (m_lr_engine != nullptr) {
      m_lr_engine->set_precision(precision);
    }
    if (m_lr_df_engine != nullptr) {
      m_lr_df_engine->set_precision(precision);
    }
    // Note: COSX doesn't have precision setting
  }

  inline double integral_precision() const {
    return m_engine.precision();
  }

  double nuclear_point_charge_interaction_energy(const PointChargeList &) const;
  double wolf_point_charge_interaction_energy(
      const PointChargeList &, const std::vector<double> &partial_charges,
      double alpha, double rc) const;

  Mat compute_fock(const MolecularOrbitals &mo,
                   const Mat &Schwarz = Mat()) const;

  inline Mat3N additional_atomic_gradients(const MolecularOrbitals &mo) const {
    return Mat3N::Zero(3, m_atoms.size());
  }

  MatTriple compute_fock_gradient(const MolecularOrbitals &mo,
                                  const Mat &Schwarz = Mat()) const;

  Mat compute_fock_mixed_basis(const MolecularOrbitals &mo_minbs,
                               const gto::AOBasis &bs, bool is_shell_diagonal);

  /// Fock matrix from a density already expressed in this method's own basis.
  ///
  /// The conventional four-centre build, which reads `mo.D` and screens on it
  /// as well as on Schwarz. Density fitting and COSX cannot serve here: both
  /// build exchange from the occupied coefficients, and a guess has a density
  /// before it has any orbitals.
  ///
  /// Distinct from `compute_fock_mixed_basis`, which exists to project a
  /// density out of a *different* basis. Handing that function this method's
  /// own basis makes it duplicate every shell as auxiliary and run the full
  /// four-index loop unscreened -- on a 566-function system that cost 256 s
  /// against 59 s for the whole rest of an SCF iteration.
  Mat compute_fock_from_density(const MolecularOrbitals &mo,
                                const Mat &Schwarz = Mat()) const {
    return m_engine.fock_operator(mo.kind, mo, Schwarz);
  }
  JKPair compute_JK(const MolecularOrbitals &mo,
                    const Mat &Schwarz = Mat()) const;
  JKTriple compute_JK_gradient(const MolecularOrbitals &mo,
                               const Mat &Schwarz = Mat()) const;

  std::vector<JKPair> compute_JK_list(const std::vector<MolecularOrbitals> &mo,
                                      const Mat &Schwarz = Mat()) const;

  Mat compute_J(const MolecularOrbitals &mo, const Mat &Schwarz = Mat()) const;

  // Exchange-only (K) build. Avoids the wasted Coulomb build when only K is
  // needed, e.g. the long-range term of a range-separated hybrid.
  Mat compute_K(const MolecularOrbitals &mo, const Mat &Schwarz = Mat()) const;

  // Coulomb plus the HF exact-exchange contribution of a (range-separated)
  // hybrid. Returns J (full 1/r Coulomb) and
  //   K = (alpha + beta) * K[1/r] - beta * K[erf(omega*r)/r].
  // Chooses the cheapest build for the active engine: a single fused COSX grid
  // sweep where possible, otherwise only the nonzero-weight operators (so a
  // pure long-range-corrected functional never builds the full-range K). Not
  // const: it briefly toggles the range-separation omega on the engines.
  JKPair coulomb_and_range_separated_exchange(const MolecularOrbitals &mo,
                                              double omega, double alpha,
                                              double beta,
                                              const Mat &Schwarz = Mat());

  MatTriple compute_J_gradient(const MolecularOrbitals &mo,
                               const Mat &Schwarz = Mat()) const;

  /// Nuclear gradient of jf * E_J + kf * E_K (E_K the full exact-exchange
  /// energy), contracted in the integral loop: 3 x natom, no derivative
  /// matrices. Always the conventional four-centre engine.
  Mat3N compute_jk_gradient(const MolecularOrbitals &mo, double jf, double kf,
                            const Mat &Schwarz = Mat()) const;

  /// As compute_jk_gradient, exchange only, with the attenuated operator
  /// erf(omega r)/r: the gradient of kf * E_K[erf].
  Mat3N compute_k_gradient_long_range(const MolecularOrbitals &mo,
                                      double omega, double kf,
                                      const Mat &Schwarz = Mat()) const;

  // --- Linear response ---------------------------------------------------
  //
  // Two forms. The general one, compute_response_fock / compute_response_jk,
  // takes AO first-order densities D1 and returns full matrices (any
  // spinorbital kind; used for checks and properties). The CPHF one,
  // response_occ, takes the perturbed occupied coefficients Cx of a
  // restricted or unrestricted reference (per spin D1 = Cx Co^T + Co Cx^T,
  // which is what a CPHF rotation or an occupied-occupied perturbation
  // gives; layout of cphf::occupied_orbitals) and returns G[D1] Co, all CPHF
  // and the Hessian use. Both follow the SCF's operators.

  /// G[D1] = dF/dD . D1 for each first-order density (same layout as mo.D).
  std::vector<Mat> compute_response_fock(const MolecularOrbitals &mo,
                                         const std::vector<Mat> &D1,
                                         const Mat &Schwarz = Mat()) const;

  /// jf * 2 J[D1] - kf * K[D1] for each restricted D1; omega != 0 selects
  /// the attenuated exchange (then jf must be 0). `ref` (the reference
  /// orbitals) is needed for density-fitted exchange.
  std::vector<Mat> compute_response_jk(const std::vector<Mat> &D1, double jf,
                                       double kf, double omega = 0.0,
                                       const Mat &Schwarz = Mat(),
                                       const MolecularOrbitals *ref = nullptr)
      const;

  /// Operators for a CPHF response: the SCF's own (Exact), or a cheaper
  /// stand-in to precondition with (Approximate: RI-J and four-centre
  /// exchange; a DFT procedure also coarsens its XC grid).
  enum class ResponseMode { Exact, Approximate };

  /// (jf 2 J[D1] - sum_t c_t K_{omega_t}[D1]) Co for D1 = Cx Co^T + Co Cx^T
  /// (restricted), or per spin (jf J[D1^a + D1^b] - sum_t c_t K[D1^s]) Co^s
  /// (unrestricted). Four-centre Coulomb and full-range exchange share one
  /// integral pass when both are four-centre.
  std::vector<Mat> response_occ(const MolecularOrbitals &mo,
                                const std::vector<Mat> &Cx, double jf,
                                const ExchangeOperators &K, const Mat &Schwarz,
                                ResponseMode mode = ResponseMode::Exact) const;

  /// The CPHF response of HF: G[D1] Co with D1 = Cx Co^T + Co Cx^T.
  std::vector<Mat> compute_response_fock_occ(const MolecularOrbitals &mo,
                                             const std::vector<Mat> &Cx,
                                             const Mat &Schwarz = Mat()) const {
    return response_occ(mo, Cx, 1.0, {{1.0, 0.0}}, Schwarz);
  }
  std::vector<Mat>
  compute_response_fock_occ_approx(const MolecularOrbitals &mo,
                                   const std::vector<Mat> &Cx,
                                   const Mat &Schwarz = Mat()) const {
    return response_occ(mo, Cx, 1.0, {{1.0, 0.0}}, Schwarz,
                        ResponseMode::Approximate);
  }
  /// Whether the approximate response (density-fitted Coulomb, four-centre
  /// exchange) is cheaper than the exact one: only when the exact exchange
  /// is COSX, since four-centre exchange costs as much as RI-K.
  bool has_approximate_response() const {
    return static_cast<bool>(m_cosx_engine);
  }

  // --- Second-derivative building blocks (analytic Hessians) -------------

  /// Empty if the analytic Hessian supports this calculation, otherwise why
  /// not (the caller falls back to finite differences).
  std::string analytic_hessian_unsupported(SpinorbitalKind kind) const {
    if (kind == SpinorbitalKind::General)
      return "general spinorbitals are not supported";
    if (kind != SpinorbitalKind::Restricted && m_cosx_engine)
      return "unrestricted COSX second derivatives are not implemented";
    return {};
  }

  /// Skeleton Hessian (fixed density) of jf E_J + sum_t c_t E_K[omega_t],
  /// with the SCF's operators (four-centre, RI-J, RI-K; four-centre Coulomb
  /// and full-range exchange in one pass).
  Mat compute_jk_hessian(const MolecularOrbitals &mo, double jf,
                         const ExchangeOperators &K,
                         const Mat &Schwarz = Mat()) const;

  /// Skeleton Fock derivatives of jf 2J - sum_t c_t K[omega_t] for every
  /// nuclear coordinate, as full matrices (unrestricted: per spin, alpha rows
  /// over beta rows) or contracted with C_o (cphf::contract_occupied).
  std::vector<Mat> compute_jk_derivative_fock(const MolecularOrbitals &mo,
                                              double jf,
                                              const ExchangeOperators &K,
                                              const Mat &Schwarz = Mat()) const;
  std::vector<Mat>
  compute_jk_derivative_fock_occ(const MolecularOrbitals &mo, double jf,
                                 const ExchangeOperators &K,
                                 const Mat &Schwarz = Mat()) const;

  /// d/dR (1/2 dE_K/dD - F_K) C_o at fixed density, F_K = -sum_t c_t
  /// K_{omega_t} the SCF's exchange: the COSX Fock is not exactly the
  /// energy's density derivative, and the analytic gradient's density
  /// response sees the latter. Empty when they coincide (not COSX).
  std::vector<Mat>
  compute_exchange_nonvariational_fock_occ(const MolecularOrbitals &mo,
                                           const ExchangeOperators &K) const;

  /// sum P h^{xy} - sum Q S^{xy} (3N x 3N), total densities.
  Mat compute_one_electron_hessian(const Mat &P, const Mat &Q) const {
    return m_engine.one_electron_hessian(P, Q);
  }

  /// S^{(x)} and h^{(x)} for every nuclear coordinate (3N each).
  void compute_one_electron_derivative_matrices(std::vector<Mat> &S1,
                                                std::vector<Mat> &H1) const {
    m_engine.one_electron_derivative_matrices(S1, H1);
  }

  /// Skeleton two-electron Hessian (fixed density), Coulomb + full exchange.
  Mat two_electron_hessian(const MolecularOrbitals &mo,
                           const Mat &Schwarz = Mat()) const {
    return compute_jk_hessian(mo, 1.0, {{1.0, 0.0}}, Schwarz);
  }

  /// Skeleton two-electron Fock derivatives (3N matrices).
  std::vector<Mat> two_electron_derivative_fock(const MolecularOrbitals &mo,
                                                const Mat &Schwarz = Mat()) const {
    return compute_jk_derivative_fock(mo, 1.0, {{1.0, 0.0}}, Schwarz);
  }
  /// The same contracted with the occupied coefficients (nbf x nocc each).
  std::vector<Mat>
  two_electron_derivative_fock_occ(const MolecularOrbitals &mo,
                                   const Mat &Schwarz = Mat()) const {
    return compute_jk_derivative_fock_occ(mo, 1.0, {{1.0, 0.0}}, Schwarz);
  }
  std::vector<Mat>
  nonvariational_fock_derivative_occ(const MolecularOrbitals &mo) const {
    return compute_exchange_nonvariational_fock_occ(mo, {{1.0, 0.0}});
  }

  /// Gradient of kf * E_K for the exchange operator in use: COSX when it is
  /// enabled (consistent with the COSX energy), else exact four-centre.
  /// `omega` != 0 selects erf(omega r)/r.
  Mat3N compute_exchange_gradient(const MolecularOrbitals &mo, double kf,
                                  double omega = 0.0,
                                  const Mat &Schwarz = Mat()) const;

  /// Two-electron part of the HF nuclear gradient (Coulomb + full exchange),
  /// with the same exchange approximation as the SCF.
  Mat3N two_electron_gradient(const MolecularOrbitals &mo,
                              const Mat &Schwarz = Mat()) const {
    if (!m_cosx_engine && !m_df_engine)
      return compute_jk_gradient(mo, 1.0, 1.0, Schwarz);
    return compute_coulomb_gradient(mo, Schwarz) +
           compute_exchange_gradient(mo, 1.0, 0.0, Schwarz);
  }

  /// Gradient of E_J for the Coulomb operator in use: RI-J when the SCF's J
  /// is density-fitted, else four-centre.
  Mat3N compute_coulomb_gradient(const MolecularOrbitals &mo,
                                 const Mat &Schwarz = Mat()) const {
    if (m_df_engine)
      return m_df_engine->coulomb_gradient(mo);
    return compute_jk_gradient(mo, 1.0, 0.0, Schwarz);
  }

  std::vector<Mat> compute_J_list(const std::vector<MolecularOrbitals> &mo,
                                  const Mat &Schwarz = Mat()) const;

  Mat compute_kinetic_matrix() const;
  MatTriple compute_kinetic_gradient() const;

  Mat compute_overlap_matrix() const;
  Mat compute_overlap_matrix_for_basis(const occ::gto::AOBasis &basis) const;
  MatTriple compute_overlap_gradient() const;

  Mat compute_nuclear_attraction_matrix() const;
  MatTriple compute_nuclear_attraction_gradient() const;

  MatTriple compute_rinv_gradient_for_atom(size_t atom_index) const;

  Mat compute_effective_core_potential_matrix() const;

  /// Nuclear gradient of the ECP energy (3 x natoms), zero without ECPs.
  /// Contracted with `mo`'s density; the caller applies occ's factor of two.
  /// The ECP operator is spin-independent, so open-shell densities are
  /// contracted as the sum of their spin blocks. Handing the kernel the
  /// stacked 2n x n `mo.D` would silently read only the alpha block, since it
  /// indexes D.block(bf0, bf1, ...).
  inline Mat3N compute_ecp_gradient(const MolecularOrbitals &mo) const {
    if (!m_engine.have_effective_core_potentials())
      return Mat3N::Zero(3, atoms().size());
    switch (mo.kind) {
    case SpinorbitalKind::Unrestricted:
      return m_engine.effective_core_potential_gradient(
          Mat(block::a(mo.D) + block::b(mo.D)));
    case SpinorbitalKind::General:
      return m_engine.effective_core_potential_gradient(
          Mat(block::aa(mo.D) + block::bb(mo.D)));
    default:
      return m_engine.effective_core_potential_gradient(mo.D);
    }
  }
  Mat compute_point_charge_interaction_matrix(
      const PointChargeList &point_charges, double alpha = 1e16) const;

  Mat compute_wolf_interaction_matrix(
      const PointChargeList &point_charges,
      const std::vector<double> &partial_charges, double alpha,
      double rc) const;

  Mat3N electronic_electric_field_contribution(const MolecularOrbitals &mo,
                                               const Mat3N &) const;
  Vec electronic_electric_potential_contribution(const MolecularOrbitals &mo,
                                                 const Mat3N &) const;
  Mat compute_schwarz_ints() const;
  void update_core_hamiltonian(const MolecularOrbitals &mo, Mat &H) { return; }
  template <int order>
  occ::core::Multipole<order>
  compute_electronic_multipoles(const MolecularOrbitals &mo,
                                const Vec3 &o = {0.0, 0.0, 0.0}) const {
    occ::core::Multipole<order> result;
    int offset = 0;
    for (int i = 0; i <= order; i++) {
      Vec c = m_engine.multipole(i, mo, o);
      for (int j = 0; j < c.rows(); j++) {
        result.components[offset++] = c(j);
      }
    }
    result.components[0] -= m_num_frozen;
    return result;
  }

  template <unsigned int order = 1>
  auto compute_nuclear_multipoles(const Vec3 &o = {0.0, 0.0, 0.0}) const {
    auto charges = occ::core::make_point_charges(m_atoms);
    return occ::core::Multipole<order>{
        occ::core::compute_multipoles<order>(charges, o)};
  }

  template <int order>
  auto compute_multipoles(const MolecularOrbitals &mo,
                          const Vec3 &o = {0.0, 0.0, 0.0}) const {
    auto mults = compute_electronic_multipoles<order>(mo, o);
    auto nuc_mults = compute_nuclear_multipoles<order>(o);
    return mults + nuc_mults;
  }

  /// Exchange built with the attenuated operator `erf(omega r)/r`.
  ///
  /// The attenuated operator gets its own engines, fixed to it for the life of
  /// the calculation, rather than being produced by toggling omega on the
  /// shared ones and toggling it back. Omega is constant for a calculation --
  /// it comes from the functional -- so the only thing the toggling ever
  /// bought was letting one engine serve two operators, and it cost exception
  /// safety, const-correctness, and the ability to cache anything that belongs
  /// to an operator rather than to a basis.
  ///
  /// COSX is the exception and takes omega as a plain argument: nothing in it
  /// is cached per operator, so a second grid and shell-pair map would be pure
  /// duplication.
  Mat compute_K_long_range(const MolecularOrbitals &mo, double omega,
                           const Mat &Schwarz = Mat()) const;

  /// Exchange gradient with the attenuated operator, as above.
  MatTriple compute_K_gradient_long_range(const MolecularOrbitals &mo,
                                          double omega,
                                          const Mat &Schwarz = Mat()) const;

  inline std::string name() const { return m_method_name; }

private:
  /// Build the long-range engines for `omega` if they are not already there.
  /// Lazy because most calculations never need them, and keyed on omega so a
  /// changed functional cannot silently reuse the wrong operator.
  void ensure_long_range_engines(double omega) const;

  mutable std::unique_ptr<IntegralEngineDF> m_df_engine{nullptr};
  mutable std::unique_ptr<occ::qm::cosx::SemiNumericalExchange> m_cosx_engine{nullptr};
  // COSX engines per grid level for the schedule (the current one lives in
  // m_cosx_engine), built on first use.
  std::array<std::unique_ptr<occ::qm::cosx::SemiNumericalExchange>, 3>
      m_cosx_stash;
  std::optional<COSXGridSchedule> m_cosx_schedule;
  occ::numint::COSXGridLevel m_cosx_level{occ::numint::COSXGridLevel::Grid1};
  int m_cosx_stage{0}; // 0 iterate, 1 refine, 2 final
  // Frozen grid corrections on the refine stage, one per exchange operator
  // (kind, alpha, beta, omega), current while their generation matches.
  struct COSXCorrection {
    std::array<double, 4> key;
    int generation;
    Mat dK;
  };
  mutable std::vector<COSXCorrection> m_cosx_corrections;
  int m_cosx_generation{0};
  mutable bool m_cosx_last_approx{false};
  bool m_cosx_incremental{false};
  template <typename Build>
  Mat cosx_scf_exchange(const std::array<double, 4> &key, Build &&build) const;
  std::optional<occ::qm::cosx::Settings> m_cosx_settings;
  double m_cosx_exchange_scale{1.0};
  occ::qm::cosx::Settings effective_cosx_settings() const;
  void apply_cosx_settings();
  void switch_cosx_grid(occ::numint::COSXGridLevel level);
  mutable occ::qm::IntegralEngine m_engine;

  /// Twins of the above, fixed to `erf(m_lr_omega r)/r` and never mutated.
  mutable std::unique_ptr<occ::qm::IntegralEngine> m_lr_engine{nullptr};
  mutable std::unique_ptr<IntegralEngineDF> m_lr_df_engine{nullptr};
  mutable double m_lr_omega{0.0};

  /// Kept from `set_density_fitting_basis` so the long-range density-fitting
  /// engine can be built later without reloading the auxiliary basis.
  std::vector<occ::gto::Shell> m_df_aux_shells;
  mutable std::unique_ptr<IntegralEngineDF> m_approx_df;
  mutable bool m_approx_df_tried{false};
  std::string m_df_basis_name;
  double m_df_auto_threshold{0.0};

  std::string m_method_name{"HF"};

  // The COSX skeleton Hessian and Fock derivatives come from one grid sweep,
  // kept per operator for the other's call at the same density.
  struct COSXSecondDerivatives {
    const void *engine;
    double c, omega;
    Mat D;
    occ::qm::cosx::SemiNumericalExchange::SecondDerivatives terms;
  };
  mutable std::vector<COSXSecondDerivatives> m_cosx_second_derivatives;
  const occ::qm::cosx::SemiNumericalExchange::SecondDerivatives &
  cosx_second_derivatives(const MolecularOrbitals &mo, double c,
                          double omega) const;
};

} // namespace occ::qm
