#pragma once
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
class HartreeFock : public SCFMethodBase {
public:
  HartreeFock(const AOBasis &basis);
  inline const auto &aobasis() const { return m_engine.aobasis(); }
  inline auto nbf() const { return m_engine.nbf(); }

  bool usual_scf_energy() const { return true; }
  void update_scf_energy(occ::core::EnergyComponents &) const { return; }

  FockBuildProperties fock_build_properties() const {
    return {// The COSX exchange build is in fact linear in the density (its
            // grid is fixed, not density adapted), so difference-density
            // builds are valid and are what the literature does. It is
            // disabled here because the shell-pair screen keys off the
            // density: screening a difference density changes which pairs
            // survive from cycle to cycle, and that interacts with the
            // convergence floor the screen already imposes. Worth enabling
            // behind its own validation, not as a side effect.
            .linear_in_density = !m_cosx_engine,
            .constant_core_hamiltonian = true,
            // Only the conventional 4-centre build screens on the density;
            // the DF path contracts through the auxiliary basis instead.
            .density_screened = !m_df_engine && !m_cosx_engine};
  }

  inline bool have_effective_core_potentials() const {
    return m_engine.have_effective_core_potentials();
  }

  void set_density_fitting_basis(const std::string &, double auto_aux_threshold = 1e-4);
  void set_density_fitting_policy(IntegralEngineDF::Policy policy);
  void set_coulomb_method(CoulombMethod method);
  void set_cosx_exchange(occ::numint::COSXGridLevel level = occ::numint::COSXGridLevel::Grid1);
  void set_cosx_settings(const occ::qm::cosx::Settings &settings);

  inline bool using_cosx() const { return m_cosx_engine != nullptr; }
  inline bool using_density_fitting() const { return m_df_engine != nullptr; }
  
  /**
   * @brief Create a new HartreeFock instance with the same settings but different basis
   * @param new_basis The new basis set to use
   * @return New HartreeFock instance
   */
  HartreeFock with_new_basis(const AOBasis &new_basis) const;

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
  mutable occ::qm::IntegralEngine m_engine;

  /// Twins of the above, fixed to `erf(m_lr_omega r)/r` and never mutated.
  mutable std::unique_ptr<occ::qm::IntegralEngine> m_lr_engine{nullptr};
  mutable std::unique_ptr<IntegralEngineDF> m_lr_df_engine{nullptr};
  mutable double m_lr_omega{0.0};

  /// Kept from `set_density_fitting_basis` so the long-range density-fitting
  /// engine can be built later without reloading the auxiliary basis.
  std::vector<occ::gto::Shell> m_df_aux_shells;

  std::string m_method_name{"HF"};
};

} // namespace occ::qm
