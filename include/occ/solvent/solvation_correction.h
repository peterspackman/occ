#pragma once
#include <fmt/format.h>
#include <occ/core/energy_components.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/qm/cphf.h>
#include <occ/qm/expectation.h>
#include <occ/numint/grid_settings.h>
#include <occ/qm/mo.h>
#include <occ/qm/scf_method.h>
#include <occ/gto/shell.h>
#include <occ/scrf/reaction_field.h>
#include <occ/solvent/cosmo.h>
#include <occ/solvent/parameters.h>
#include <memory>
#include <optional>
#include <stdexcept>

namespace occ::solvent {

using occ::qm::MolecularOrbitals;
using occ::qm::SpinorbitalKind;
using PointChargeList = std::vector<occ::core::PointCharge>;

/// SMD-style continuum solvation model used by the HF/DFT pipeline.
///
/// Phase 2: this class is now a thin adapter over
/// `occ::scrf::ReactionFieldEngine`. The public API is preserved so callers
/// (drivers, `SolvationCorrectedProcedure<Proc>`, the cg solvation pipeline)
/// do not need to change; the engine pre-factors the COSMO A matrix once and
/// re-uses it across SCF iterations.
class ContinuumSolvationModel {
public:
  ContinuumSolvationModel(const std::vector<occ::core::Atom> &,
                          const std::string &solvent = "water",
                          double charge = 0.0, bool draco = false);

  /// Drive the engine from an explicit configuration instead of the SMD
  /// parameter tables. Used for the ideal-conductor COSMO calculation that
  /// σ-profiles are built from, where there is no SMD solvent to look up and
  /// no CDS branch.
  ContinuumSolvationModel(const std::vector<occ::core::Atom> &,
                          const occ::scrf::Options &, double charge = 0.0);

  void set_solvent(const std::string &);
  const std::string &solvent() const { return m_solvent_name; }

  const Mat3N &nuclear_positions() const { return m_nuclear_positions; }
  const Mat3N &surface_positions_coulomb() const {
    return m_engine.es_cavity().vertices;
  }
  const Mat3N &surface_positions_cds() const {
    return m_engine.cds_cavity().vertices;
  }
  const Vec &surface_areas_coulomb() const {
    return m_engine.es_cavity().areas;
  }
  const Vec &surface_areas_cds() const { return m_engine.cds_cavity().areas; }
  const Vec &nuclear_charges() const { return m_nuclear_charges; }
  size_t num_surface_points() const {
    return m_engine.num_es_surface_points();
  }
  void set_surface_potential(const Vec &);
  const Vec &apparent_surface_charge();

  double surface_polarization_energy();
  double surface_charge() const { return m_asc.array().sum(); }
  double smd_cds_energy() const { return m_engine.energy_cds(); }

  inline double charge() const { return m_charge; }
  inline void set_charge(double charge) {
    m_charge = charge;
    initialize_surfaces();
  }

  Vec surface_cds_energy_elements() const {
    return m_engine.cds_energy_elements();
  }
  Vec surface_polarization_energy_elements() const;

  /// Per-element decomposition of the latest SCF in the unified
  /// `occ::scrf::SolvationSurfaces` shape (the same one consumed by cg).
  ///
  /// The coulomb branch energies are `½ σ_i · φ_total_i` per element, which
  /// is algebraically identical to `nuc_i + elec_i + pol_i` from the per-
  /// element decomposition above. Verify:
  ///   nuc_i + elec_i + pol_i
  ///     = σ_i·φ_nuc_i + σ_i·φ_elec_i − ½ σ_i·φ_total_i
  ///     = σ_i·φ_total_i − ½ σ_i·φ_total_i = ½ σ_i·φ_total_i.
  occ::scrf::SolvationSurfaces solvation_surfaces() const {
    return m_engine.surfaces();
  }

  /// Access the underlying SCRF engine — exposes the pre-factored A matrix,
  /// surfaces, and direct ASC/V_atom accessors. Phase 2 onwards.
  const occ::scrf::ReactionFieldEngine &engine() const { return m_engine; }

  void write_surface_file(const std::string &filename);
  inline std::string name() const {
    return fmt::format("SMD(solvent='{}')", m_solvent_name);
  }

private:
  void initialize_surfaces();
  /// Returns the ES radii to use for the current geometry/charge — uses
  /// DRACO-scaled radii when `m_scale_radii`, intrinsic Coulomb radii otherwise.
  Vec compute_es_radii();

  double m_charge{0.0};
  Vec m_atomic_charges;            // EEQ charges (DRACO path only)
  /// False when constructed from explicit options: skips the SMD parameter
  /// lookup, radii and logging.
  bool m_smd{true};

  std::string m_solvent_name;
  Mat3N m_nuclear_positions;
  Vec m_nuclear_charges;
  Vec m_surface_potential;
  Vec m_asc;
  bool m_asc_needs_update{true};
  SMDSolventParameters m_params;

  occ::scrf::ReactionFieldEngine m_engine;
  bool m_scale_radii{false};
};

template <typename Proc> class SolvationCorrectedProcedure {
public:
  SolvationCorrectedProcedure(Proc &proc, const std::string &solvent = "water",
                              bool radii_scaling = false)
      : m_atoms(proc.atoms()), m_proc(proc),
        m_solvation_model(proc.atoms(), solvent, m_proc.system_charge(),
                          radii_scaling),
        m_radii_scaling(radii_scaling) {
    initialize();
  }

  /// Drive the reaction field from an explicit engine configuration rather
  /// than the SMD tables — see the matching `ContinuumSolvationModel` ctor.
  SolvationCorrectedProcedure(Proc &proc, const occ::scrf::Options &opts)
      : m_atoms(proc.atoms()), m_proc(proc),
        m_solvation_model(proc.atoms(), opts, m_proc.system_charge()),
        m_options(opts) {
    initialize();
  }

  /// The same solvated method on another basis (another geometry), owning
  /// its procedure: what finite-difference Hessians displace.
  SolvationCorrectedProcedure with_new_basis(const occ::gto::AOBasis &basis)
      const {
    auto proc = std::make_unique<Proc>(m_proc.with_new_basis(basis));
    if (m_options)
      return SolvationCorrectedProcedure(std::move(proc), *m_options);
    return SolvationCorrectedProcedure(std::move(proc),
                                       m_solvation_model.solvent(),
                                       m_radii_scaling);
  }

  occ::qm::FockBuildProperties fock_build_properties() const {
    // `update_core_hamiltonian` folds the apparent surface charge into H every
    // cycle, so H is not constant no matter what the wrapped method allows.
    auto properties = m_proc.fock_build_properties();
    properties.constant_core_hamiltonian = false;
    return properties;
  }
  // SCF hooks of the wrapped method (the COSX grid schedule), passed through
  // when it has them: the reaction field does not change how K is built.
  bool advance_scf_grid(double diis_error, bool converged)
    requires requires(Proc &p) { p.advance_scf_grid(diis_error, converged); }
  {
    return m_proc.advance_scf_grid(diis_error, converged);
  }
  bool scf_fock_approximate() const
    requires requires(const Proc &p) { p.scf_fock_approximate(); }
  {
    return m_proc.scf_fock_approximate();
  }
  void refresh_scf_fock_correction()
    requires requires(Proc &p) { p.refresh_scf_fock_correction(); }
  {
    m_proc.refresh_scf_fock_correction();
  }
  void set_scf_fock_incremental(bool incremental)
    requires requires(Proc &p) { p.set_scf_fock_incremental(incremental); }
  {
    m_proc.set_scf_fock_incremental(incremental);
  }
  double incremental_fock_floor() const
    requires requires(const Proc &p) { p.incremental_fock_floor(); }
  {
    return m_proc.incremental_fock_floor();
  }
  double final_energy_correction(const MolecularOrbitals &mo)
    requires requires(Proc &p) { p.final_energy_correction(mo); }
  {
    return m_proc.final_energy_correction(mo);
  }
  double cosx_energy_shift_from(occ::numint::COSXGridLevel other,
                                const MolecularOrbitals &mo)
    requires requires(Proc &p) { p.cosx_energy_shift_from(other, mo); }
  {
    return m_proc.cosx_energy_shift_from(other, mo);
  }

  inline bool have_effective_core_potentials() const {
    return m_proc.have_effective_core_potentials();
  }
  inline const auto &atoms() const { return m_proc.atoms(); }
  inline const auto &aobasis() const { return m_proc.aobasis(); }
  inline auto nbf() const { return m_proc.nbf(); }

  inline Vec3 center_of_mass() const { return m_proc.center_of_mass(); }

  void set_system_charge(int charge) {
    m_proc.set_system_charge(charge);
    m_solvation_model.set_charge(charge);
  }

  inline void set_precision(double precision) {
    m_proc.set_precision(precision);
  }

  inline int system_charge() const { return m_proc.system_charge(); }

  int total_electrons() const { return m_proc.total_electrons(); }
  int active_electrons() const { return m_proc.active_electrons(); }
  inline const auto &frozen_electrons() const {
    return m_proc.frozen_electrons();
  }

  bool usual_scf_energy() const { return m_proc.usual_scf_energy(); }

  double nuclear_repulsion_energy() const {
    return m_proc.nuclear_repulsion_energy();
  }

  inline double
  nuclear_point_charge_interaction_energy(const PointChargeList &pc) const {
    return m_proc.nuclear_point_charge_interaction_energy(pc);
  }

  void update_scf_energy(occ::core::EnergyComponents &energy) const {
    m_proc.update_scf_energy(energy);
    energy["solvation.electronic"] = m_electronic_solvation_energy;
    energy["solvation.nuclear"] = m_nuclear_solvation_energy;
    energy["solvation.surface"] = m_surface_solvation_energy;
    energy["solvation.CDS"] = m_cds_solvation_energy;
    energy["total"] = energy["electronic"] + energy["nuclear.repulsion"] +
                      energy["solvation.nuclear"] +
                      energy["solvation.surface"] + energy["solvation.CDS"];
  }

  auto compute_kinetic_matrix() const {
    return m_proc.compute_kinetic_matrix();
  }

  auto compute_overlap_matrix() const {
    return m_proc.compute_overlap_matrix();
  }

  auto compute_overlap_matrix_for_basis(const occ::gto::AOBasis &bs) const {
    return m_proc.compute_overlap_matrix_for_basis(bs);
  }

  auto compute_nuclear_attraction_matrix() const {
    return m_proc.compute_nuclear_attraction_matrix();
  }

  auto compute_effective_core_potential_matrix() const {
    return m_proc.compute_effective_core_potential_matrix();
  }

  auto compute_schwarz_ints() const { return m_proc.compute_schwarz_ints(); }

  inline auto
  compute_point_charge_interaction_matrix(const PointChargeList &pc) {
    return m_proc.compute_point_charge_interaction_matrix(pc);
  }

  void update_core_hamiltonian(const MolecularOrbitals &mo, occ::Mat &H) {
    occ::timing::start(occ::timing::category::solvent);
    occ::Vec v =
        (m_qn + m_proc.electronic_electric_potential_contribution(
                    mo, m_solvation_model.surface_positions_coulomb()));
    m_solvation_model.set_surface_potential(v);
    auto asc = m_solvation_model.apparent_surface_charge();
    for (int i = 0; i < m_point_charges.size(); i++) {
      m_point_charges[i].set_charge(asc(i));
    }
    double surface_energy = m_solvation_model.surface_polarization_energy();
    m_nuclear_solvation_energy = m_qn.dot(asc);
    m_surface_solvation_energy = surface_energy;
    m_electronic_solvation_energy = 0.0;
    occ::log::debug("PCM surface polarization energy: {:.12f}", surface_energy);
    occ::log::debug("PCM surface charge: {:.12f}",
                    m_solvation_model.surface_charge());
    m_X = m_proc.compute_point_charge_interaction_matrix(m_point_charges);

    switch (mo.kind) {
    case SpinorbitalKind::Restricted: {
      m_electronic_solvation_energy =
          2 * occ::qm::expectation<SpinorbitalKind::Restricted>(mo.D, H);
      H += m_X;
      m_electronic_solvation_energy =
          2 * occ::qm::expectation<SpinorbitalKind::Restricted>(mo.D, H) -
          m_electronic_solvation_energy;
      break;
    }
    case SpinorbitalKind::Unrestricted: {
      m_electronic_solvation_energy =
          2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(mo.D, H);
      occ::qm::block::a(H) += m_X;
      occ::qm::block::b(H) += m_X;
      m_electronic_solvation_energy =
          2 * occ::qm::expectation<SpinorbitalKind::Unrestricted>(mo.D, H) -
          m_electronic_solvation_energy;
      break;
    }
    case SpinorbitalKind::General: {
      m_electronic_solvation_energy =
          2 * occ::qm::expectation<SpinorbitalKind::General>(mo.D, H);
      occ::qm::block::aa(H) += m_X;
      occ::qm::block::bb(H) += m_X;
      m_electronic_solvation_energy =
          2 * occ::qm::expectation<SpinorbitalKind::General>(mo.D, H) -
          m_electronic_solvation_energy;
      break;
    }
    }
    occ::timing::stop(occ::timing::category::solvent);
  }

  Mat compute_fock(const MolecularOrbitals &mo,
                   const Mat &Schwarz = Mat()) const {
    return m_proc.compute_fock(mo, Schwarz);
  }

  Mat compute_fock_from_density(const MolecularOrbitals &mo,
                                const Mat &Schwarz = Mat()) const {
    return m_proc.compute_fock_from_density(mo, Schwarz);
  }

  Mat compute_fock_mixed_basis(const MolecularOrbitals &mo_bs,
                               const qm::AOBasis &bs, bool is_shell_diagonal) {
    return m_proc.compute_fock_mixed_basis(mo_bs, bs, is_shell_diagonal);
  }

  void set_solvent(const std::string &solvent) {
    m_solvation_model.set_solvent(solvent);
  }

  void write_surface_file(const std::string &filename) {
    m_solvation_model.write_surface_file(filename);
  }

  auto surface_positions_coulomb() const {
    return m_solvation_model.surface_positions_coulomb();
  }
  auto surface_positions_cds() const {
    return m_solvation_model.surface_positions_cds();
  }
  auto surface_areas_coulomb() const {
    return m_solvation_model.surface_areas_coulomb();
  }
  auto surface_areas_cds() const {
    return m_solvation_model.surface_areas_cds();
  }

  auto surface_cds_energy_elements() const {
    return m_solvation_model.surface_cds_energy_elements();
  }
  auto surface_polarization_energy_elements() const {
    return m_solvation_model.surface_polarization_energy_elements();
  }

  /// Per-element decomposition in the unified `occ::scrf::SolvationSurfaces`
  /// shape (same one consumed by cg, same one produced by the xTB pipeline).
  occ::scrf::SolvationSurfaces solvation_surfaces() const {
    return m_solvation_model.solvation_surfaces();
  }
  const occ::scrf::ReactionFieldEngine &engine() const {
    return m_solvation_model.engine();
  }

  template <unsigned int order = 1>
  inline auto
  compute_electronic_multipole_matrices(const Vec3 &o = {0.0, 0.0, 0.0}) const {
    return m_proc.template compute_electronic_multipole_matrices<order>(o);
  }

  template <unsigned int order = 1>
  inline auto compute_electronic_multipoles(const MolecularOrbitals &mo,
                                            const Vec3 &o = {0.0, 0.0,
                                                             0.0}) const {
    return m_proc.template compute_electronic_multipoles<order>(mo, o);
  }

  template <unsigned int order = 1>
  inline auto compute_nuclear_multipoles(const Vec3 &o = {0.0, 0.0,
                                                          0.0}) const {
    return m_proc.template compute_nuclear_multipoles<order>(o);
  }

  // --- Nuclear gradient ----------------------------------------------------
  //
  // The solvated energy is E_gas[D] + ½ σ·φ + E_cds, with φ = φ_nuc + φ_el[D]
  // the solute potential on the cavity and σ = −f A⁻¹ φ (at convergence the
  // σ in the Fock matrix is that of the final density). ½ σ·φ is stationary
  // in σ, so at fixed density its derivative is σᵀ ∂φ/∂x + (1/2f) σᵀ ∂A/∂x σ,
  // with the cavity points moving with their atoms; the Pulay term needs no
  // change, as the orbital energies already include the reaction field.

  Mat3N nuclear_repulsion_gradient() const {
    return m_proc.nuclear_repulsion_gradient();
  }
  auto compute_overlap_gradient() const {
    return m_proc.compute_overlap_gradient();
  }
  auto compute_kinetic_gradient() const {
    return m_proc.compute_kinetic_gradient();
  }
  auto compute_nuclear_attraction_gradient() const {
    return m_proc.compute_nuclear_attraction_gradient();
  }
  auto compute_rinv_gradient_for_atom(size_t atom) const {
    return m_proc.compute_rinv_gradient_for_atom(atom);
  }
  Mat3N two_electron_gradient(const MolecularOrbitals &mo,
                              const Mat &Schwarz = Mat()) const {
    return m_proc.two_electron_gradient(mo, Schwarz);
  }
  Mat3N compute_ecp_gradient(const MolecularOrbitals &mo) const {
    return m_proc.compute_ecp_gradient(mo);
  }
  Mat3N additional_atomic_gradients(const MolecularOrbitals &mo) const {
    return m_proc.additional_atomic_gradients(mo) + solvation_gradient(mo);
  }

  /// Nuclear gradient of ½ σ·φ + E_cds at the density of mo (3 x natom).
  Mat3N solvation_gradient(const MolecularOrbitals &mo) const {
    require_derivatives("Solvation gradients");
    const auto &engine = m_solvation_model.engine();
    const auto &cavity = engine.es_cavity();
    Mat3N grad = engine.cds_gradient();
    if (cavity.areas.size() == 0)
      return grad;
    const Vec phi = m_qn + m_proc.electronic_electric_potential_contribution(
                               mo, cavity.vertices);
    const Vec sigma = engine.solve_cavity(-engine.f_epsilon() * phi);
    grad += engine.es_gradient(sigma, m_nuclear_charges);
    // φ_el,k = −Σ P L^k
    const Vec g = -integrals().point_charge_potential_derivatives(
                      mo.total_density_matrix(), cavity.vertices,
                      cavity.atom_index) *
                  sigma;
    grad += Eigen::Map<const Mat3N>(g.data(), 3, grad.cols());
    return grad;
  }

  // --- Linear response and analytic Hessian ---------------------------------
  //
  // The reaction field's Fock contribution is −Σ_k σ_k L^k, L^k the
  // 1/|r − s_k| integrals, and σ follows the density: a first-order total
  // density P1 gives σ1 = f A⁻¹ (Σ P1 L) (equilibrium solvation, the static
  // response). The terms below take total densities, so they serve
  // restricted and unrestricted references alike, the same matrices entering
  // both spins.

  /// Reaction-field response −Σ_k σ1_k[P1] L^k for first-order total
  /// densities P1 (symmetric).
  std::vector<Mat> solvation_response(const std::vector<Mat> &P1) const {
    const auto &engine = m_solvation_model.engine();
    const auto &points = engine.es_cavity().vertices;
    std::vector<Mat> X;
    if (points.cols() == 0) {
      X.assign(P1.size(), Mat::Zero(nbf(), nbf()));
      return X;
    }
    const Mat V = integrals().point_charge_potentials(P1, points);
    X = integrals().point_charge_matrices(
        engine.solve_cavity(engine.f_epsilon() * V), points);
    for (auto &x : X)
      x = -x;
    return X;
  }

  /// Fixed-density second derivatives of ½ σ·φ + E_cds (3N x 3N) at total
  /// density P.
  Mat solvation_hessian(const Mat &P) const {
    require_derivatives("Analytic solvation Hessians");
    const auto &engine = m_solvation_model.engine();
    Mat H = engine.cds_hessian();
    if (engine.es_cavity().areas.size() == 0)
      return H;
    const auto r = first_order_charges(P);
    H += r.hessian + r.W.transpose() * r.sigma_x / engine.f_epsilon();
    H -= integrals().point_charge_hessian(P, r.sigma,
                                          engine.es_cavity().vertices,
                                          engine.es_cavity().atom_index);
    return H;
  }

  /// Fixed-density derivative of the reaction-field Fock matrix for every
  /// nuclear coordinate (3N matrices), −Σ σ_k ∂L^k/∂x − Σ ∂σ_k/∂x L^k.
  std::vector<Mat> solvation_derivative_fock(const Mat &P) const {
    require_derivatives("Analytic solvation Hessians");
    const auto &engine = m_solvation_model.engine();
    const auto &cavity = engine.es_cavity();
    const auto n3 = 3 * m_atoms.size();
    if (cavity.areas.size() == 0)
      return std::vector<Mat>(n3, Mat::Zero(nbf(), nbf()));
    const auto r = first_order_charges(P);
    auto F = integrals().point_charge_derivative_matrices(
        r.sigma, cavity.vertices, cavity.atom_index);
    const auto F2 = integrals().point_charge_matrices(r.sigma_x,
                                                      cavity.vertices);
    for (size_t x = 0; x < n3; x++)
      F[x] = -(F[x] + F2[x]);
    return F;
  }

  /// The CPHF response G[D1] C_o (see occ::qm::cphf::ResponseFunction).
  std::vector<Mat> compute_response_fock_occ(const MolecularOrbitals &mo,
                                             const std::vector<Mat> &Cx,
                                             const Mat &Schwarz = Mat()) const {
    auto G = m_proc.compute_response_fock_occ(mo, Cx, Schwarz);
    add_solvation_response_occ(mo, Cx, G);
    return G;
  }
  std::vector<Mat>
  compute_response_fock_occ_approx(const MolecularOrbitals &mo,
                                   const std::vector<Mat> &Cx,
                                   const Mat &Schwarz = Mat()) const {
    auto G = m_proc.compute_response_fock_occ_approx(mo, Cx, Schwarz);
    add_solvation_response_occ(mo, Cx, G);
    return G;
  }
  bool has_approximate_response() const {
    return m_proc.has_approximate_response();
  }
  /// The reaction field is variational, so only the wrapped method's
  /// non-variational exchange correction applies.
  std::vector<Mat>
  nonvariational_fock_derivative_occ(const MolecularOrbitals &mo) const {
    return m_proc.nonvariational_fock_derivative_occ(mo);
  }

  std::string analytic_hessian_unsupported(SpinorbitalKind kind) const {
    if (m_radii_scaling)
      return "DRACO-scaled cavity radii have no nuclear derivatives";
    return m_proc.analytic_hessian_unsupported(kind);
  }

  Mat compute_one_electron_hessian(const Mat &P, const Mat &Q) const {
    return m_proc.compute_one_electron_hessian(P, Q);
  }
  void compute_one_electron_derivative_matrices(std::vector<Mat> &S1,
                                                std::vector<Mat> &H1) const {
    m_proc.compute_one_electron_derivative_matrices(S1, H1);
  }
  /// Skeleton Hessian of the wrapped method plus the solvation terms, which
  /// depend on the density.
  Mat two_electron_hessian(const MolecularOrbitals &mo,
                           const Mat &Schwarz = Mat()) const {
    return m_proc.two_electron_hessian(mo, Schwarz) +
           solvation_hessian(mo.total_density_matrix());
  }
  std::vector<Mat>
  two_electron_derivative_fock_occ(const MolecularOrbitals &mo,
                                   const Mat &Schwarz = Mat()) const {
    auto G1o = m_proc.two_electron_derivative_fock_occ(mo, Schwarz);
    const Mat Co = occ::qm::cphf::occupied_orbitals(mo);
    const auto F = solvation_derivative_fock(mo.total_density_matrix());
    for (size_t k = 0; k < G1o.size(); k++)
      G1o[k] += F[k] * Co;
    return G1o;
  }

  inline std::string name() const {
    return fmt::format("{}+{}", m_proc.name(), m_solvation_model.name());
  }

private:
  SolvationCorrectedProcedure(std::unique_ptr<Proc> proc,
                              const std::string &solvent, bool radii_scaling)
      : m_owned(std::move(proc)), m_atoms(m_owned->atoms()), m_proc(*m_owned),
        m_solvation_model(m_atoms, solvent, m_proc.system_charge(),
                          radii_scaling),
        m_radii_scaling(radii_scaling) {
    initialize();
  }
  SolvationCorrectedProcedure(std::unique_ptr<Proc> proc,
                              const occ::scrf::Options &opts)
      : m_owned(std::move(proc)), m_atoms(m_owned->atoms()), m_proc(*m_owned),
        m_solvation_model(m_atoms, opts, m_proc.system_charge()),
        m_options(opts) {
    initialize();
  }

  const auto &integrals() const { return m_proc.integral_engine(); }

  void require_derivatives(const char *what) const {
    if (m_radii_scaling)
      throw std::runtime_error(
          fmt::format("{} are not available with DRACO-scaled cavity radii, "
                      "which depend on the geometry through EEQ charges",
                      what));
  }

  // σ at total density P, and its first-order change at fixed density:
  // W = f ∂φ/∂x + (∂A/∂x) σ (cavity x 3N) and σ_x = −A⁻¹ W, with the
  // fixed-σ nuclear and cavity second derivatives.
  struct FirstOrderCharges {
    Vec sigma;
    Mat W, sigma_x, hessian;
  };
  FirstOrderCharges first_order_charges(const Mat &P) const {
    const auto &engine = m_solvation_model.engine();
    const auto &cavity = engine.es_cavity();
    const double f = engine.f_epsilon();
    FirstOrderCharges r;
    const Vec phi = m_qn - integrals()
                               .point_charge_potentials({P}, cavity.vertices)
                               .col(0);
    r.sigma = engine.solve_cavity(-f * phi);
    auto d = engine.es_derivatives(r.sigma, m_nuclear_charges);
    const Mat dphi_el = -integrals()
                             .point_charge_potential_derivatives(
                                 P, cavity.vertices, cavity.atom_index)
                             .transpose();
    r.W = f * (d.dphi + dphi_el) + d.dA_sigma;
    r.sigma_x = -engine.solve_cavity(r.W);
    r.hessian = std::move(d.hessian);
    return r;
  }

  // The reaction field responds to the total first-order density, the sum
  // over channels of Cx^s C_o^s^T + C_o^s Cx^s^T (twice that, restricted),
  // and acts on both spins alike.
  void add_solvation_response_occ(const MolecularOrbitals &mo,
                                  const std::vector<Mat> &Cx,
                                  std::vector<Mat> &G) const {
    const Mat Co = occ::qm::cphf::occupied_orbitals(mo);
    const double electrons =
        mo.kind == SpinorbitalKind::Restricted ? 2.0 : 1.0;
    std::vector<Mat> P1;
    P1.reserve(Cx.size());
    for (const auto &c : Cx) {
      const Mat X = c * Co.transpose();
      P1.push_back(electrons * (X + X.transpose()));
    }
    const auto X = solvation_response(P1);
    for (size_t k = 0; k < G.size(); k++)
      G[k] += X[k] * Co;
  }

  void initialize() {
    m_nuclear_charges = Vec(m_atoms.size());
    const auto &frozen = m_proc.frozen_electrons();
    for (size_t i = 0; i < m_atoms.size(); i++)
      m_nuclear_charges(i) = m_atoms[i].atomic_number -
                             (i < frozen.size() ? frozen[i] : 0);
    m_qn = m_proc.nuclear_electric_potential_contribution(
        m_solvation_model.surface_positions_coulomb());
    m_point_charges.reserve(m_qn.rows());
    for (int i = 0; i < m_solvation_model.num_surface_points(); i++) {
      const auto &pt = m_solvation_model.surface_positions_coulomb().col(i);
      m_point_charges.emplace_back(0.0, pt);
    }
    m_cds_solvation_energy = m_solvation_model.smd_cds_energy();
  }

  std::unique_ptr<Proc> m_owned; // set by with_new_basis
  const std::vector<occ::core::Atom> &m_atoms;
  Proc &m_proc;
  ContinuumSolvationModel m_solvation_model;
  bool m_radii_scaling{false};
  std::optional<occ::scrf::Options> m_options;
  Vec m_nuclear_charges; // ECP-reduced
  std::vector<core::PointCharge> m_point_charges;
  double m_electronic_solvation_energy{0.0}, m_nuclear_solvation_energy{0.0},
      m_surface_solvation_energy, m_cds_solvation_energy;
  Mat m_X;
  Vec m_qn;
};

} // namespace occ::solvent
