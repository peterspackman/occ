#pragma once
#include <occ/core/energy_components.h>
#include <occ/core/multipole.h>
#include <occ/core/point_charge.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/integral_engine_df.h>
#include <occ/qm/mo.h>
#include <occ/qm/spinorbital.h>
#include <optional>

namespace occ::hf {

using occ::qm::AOBasis;
using occ::qm::MolecularOrbitals;
using PointChargeList = std::vector<occ::core::PointCharge>;

/// to use precomputed shell pair data must decide on max precision a priori
const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;

class HartreeFock {
  public:
    HartreeFock(const AOBasis &basis);
    inline const auto &atoms() const { return m_atoms; }
    inline const auto &aobasis() const { return m_engine.aobasis(); }
    inline auto nbf() const { return m_engine.nbf(); }

    int system_charge() const { return m_charge; }
    int num_e() const { return m_num_e; }

    Vec3 center_of_mass() const;

    double two_electron_energy_alpha() const { return m_e_alpha; }
    double two_electron_energy_beta() const { return m_e_beta; }
    double two_electron_energy() const { return m_e_alpha + m_e_beta; }
    bool usual_scf_energy() const { return true; }
    void update_scf_energy(occ::core::EnergyComponents &energy,
                           bool incremental) const {
        return;
    }
    bool supports_incremental_fock_build() const { return !m_df_engine; }

    void set_system_charge(int charge);
    void set_density_fitting_basis(const std::string &);
    double nuclear_repulsion_energy() const;

    Mat compute_fock(const MolecularOrbitals &mo,
                     double precision = std::numeric_limits<double>::epsilon(),
                     const Mat &Schwarz = Mat()) const;

    Mat compute_fock_mixed_basis(const MolecularOrbitals &mo_bs,
                                 const qm::AOBasis &bs, bool is_shell_diagonal);
    std::pair<Mat, Mat>
    compute_JK(const MolecularOrbitals &mo,
               double precision = std::numeric_limits<double>::epsilon(),
               const Mat &Schwarz = Mat()) const;
    Mat compute_J(const MolecularOrbitals &mo,
                  double precision = std::numeric_limits<double>::epsilon(),
                  const Mat &Schwarz = Mat()) const;

    Mat compute_kinetic_matrix() const;
    Mat compute_overlap_matrix() const;
    Mat compute_nuclear_attraction_matrix() const;
    Mat compute_point_charge_interaction_matrix(
        const PointChargeList &point_charges) const;

    Mat3N nuclear_electric_field_contribution(const Mat3N &) const;
    Mat3N electronic_electric_field_contribution(const MolecularOrbitals &mo,
                                                 const Mat3N &) const;
    Vec electronic_electric_potential_contribution(const MolecularOrbitals &mo,
                                                   const Mat3N &) const;
    Vec nuclear_electric_potential_contribution(const Mat3N &) const;

    Mat compute_schwarz_ints() const;
    void update_core_hamiltonian(const MolecularOrbitals &mo, Mat &H) {
        return;
    }
    template <int order>
    occ::core::Multipole<order>
    compute_electronic_multipoles(const MolecularOrbitals &mo,
                                  const Vec3 &o = {0.0, 0.0, 0.0}) const {
        occ::core::Multipole<order> result;
        const auto &D = mo.D;
        int offset = 0;
        for (int i = 0; i <= order; i++) {
            Vec c = m_engine.multipole(i, mo, o);
            for (int j = 0; j < c.rows(); j++) {
                result.components[offset++] = c(j);
            }
        }
        return result;
    }

    template <unsigned int order = 1>
    auto compute_nuclear_multipoles(const Vec3 &o = {0.0, 0.0, 0.0}) const {
        std::array<double, 3> c{o(0), o(1), o(2)};
        auto charges = occ::core::make_point_charges(m_atoms);
        return occ::core::Multipole<order>{
            occ::core::compute_multipoles<order>(charges, c)};
    }

    template <int order>
    auto compute_multipoles(const MolecularOrbitals &mo,
                            const Vec3 &o = {0.0, 0.0, 0.0}) const {
        auto mults = compute_electronic_multipoles<order>(mo, o);
        auto nuc_mults = compute_nuclear_multipoles<order>(o);
        return mults + nuc_mults;
    }

  private:
    int m_charge{0};
    int m_num_e{0};
    std::vector<occ::core::Atom> m_atoms;
    mutable double m_e_alpha{0};
    mutable double m_e_beta{0};
    mutable std::optional<occ::qm::IntegralEngineDF> m_df_engine;
    mutable occ::qm::IntegralEngine m_engine;
};

} // namespace occ::hf
