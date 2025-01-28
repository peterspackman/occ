#pragma once
#include <occ/core/energy_components.h>
#include <occ/core/multipole.h>
#include <occ/core/point_charge.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/mo.h>
#include <occ/qm/scf_method.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

using occ::qm::AOBasis;
using occ::qm::MolecularOrbitals;
using PointChargeList = std::vector<occ::core::PointCharge>;
class HartreeFock : public SCFMethodBase {
public:
  HartreeFock(const AOBasis &basis);
  inline const auto &aobasis() const { return m_engine.aobasis(); }
  inline auto nbf() const { return m_engine.nbf(); }

  bool usual_scf_energy() const { return true; }
  void update_scf_energy(occ::core::EnergyComponents &energy,
                         bool incremental) const {
    return;
  }
  bool supports_incremental_fock_build() const { return !m_df_engine; }

  inline bool have_effective_core_potentials() const {
    return m_engine.have_effective_core_potentials();
  }

  void set_density_fitting_basis(const std::string &);

  inline void set_precision(double precision) {
    m_engine.set_precision(precision);

    if (m_df_engine != nullptr) {
      m_df_engine->set_precision(precision);
    }
  }

  double nuclear_point_charge_interaction_energy(const PointChargeList &) const;
  double wolf_point_charge_interaction_energy(
      const PointChargeList &, const std::vector<double> &partial_charges,
      double alpha, double rc) const;

  Mat compute_fock(const MolecularOrbitals &mo,
                   const Mat &Schwarz = Mat()) const;

  MatTriple compute_fock_gradient(const MolecularOrbitals &mo,
                                  const Mat &Schwarz = Mat()) const;

  Mat compute_fock_mixed_basis(const MolecularOrbitals &mo_minbs,
                               const qm::AOBasis &bs, bool is_shell_diagonal);
  JKPair compute_JK(const MolecularOrbitals &mo,
                    const Mat &Schwarz = Mat()) const;
  JKTriple compute_JK_gradient(const MolecularOrbitals &mo,
                               const Mat &Schwarz = Mat()) const;

  std::vector<JKPair> compute_JK_list(const std::vector<MolecularOrbitals> &mo,
                                      const Mat &Schwarz = Mat()) const;

  Mat compute_J(const MolecularOrbitals &mo, const Mat &Schwarz = Mat()) const;

  MatTriple compute_J_gradient(const MolecularOrbitals &mo,
                               const Mat &Schwarz = Mat()) const;

  std::vector<Mat> compute_J_list(const std::vector<MolecularOrbitals> &mo,
                                  const Mat &Schwarz = Mat()) const;

  Mat compute_kinetic_matrix() const;
  MatTriple compute_kinetic_gradient() const;

  Mat compute_overlap_matrix() const;
  Mat compute_overlap_matrix_for_basis(const occ::qm::AOBasis &basis) const;
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

  inline double range_separated_omega() const {
    return m_engine.range_separated_omega();
  }

  inline void set_range_separated_omega(double omega) {
    m_engine.set_range_separated_omega(omega);
    if (m_df_engine) {
      (*m_df_engine).set_range_separated_omega(omega);
    }
  }

  inline std::string name() const { return m_method_name; }

private:
  mutable std::unique_ptr<IntegralEngineDF> m_df_engine{nullptr};
  mutable occ::qm::IntegralEngine m_engine;
  std::string m_method_name{"HF"};
};

} // namespace occ::qm
