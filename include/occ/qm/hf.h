#pragma once
#include <occ/core/energy_components.h>
#include <occ/core/multipole.h>
#include <occ/core/point_charge.h>
#include <occ/qm/density_fitting.h>
#include <occ/qm/fock.h>
#include <occ/qm/mo.h>
#include <occ/qm/spinorbital.h>

namespace occ::hf {

using occ::ints::BasisSet;
using occ::ints::compute_1body_ints;
using occ::ints::compute_1body_ints_deriv;
using occ::ints::Operator;
using occ::ints::ShellPairData;
using occ::ints::ShellPairList;
using occ::qm::MolecularOrbitals;
using occ::qm::SpinorbitalKind;

/// to use precomputed shell pair data must decide on max precision a priori
const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;

class HartreeFock {
  public:
    HartreeFock(const std::vector<occ::core::Atom> &atoms,
                const BasisSet &basis);
    const auto &shellpair_list() const { return m_shellpair_list; }
    const auto &shellpair_data() const { return m_shellpair_data; }
    const auto &atoms() const { return m_atoms; }
    const auto &basis() const { return m_basis; }

    int system_charge() const { return m_charge; }
    int num_e() const { return m_num_e; }

    double two_electron_energy_alpha() const { return m_e_alpha; }
    double two_electron_energy_beta() const { return m_e_beta; }
    double two_electron_energy() const { return m_e_alpha + m_e_beta; }
    bool usual_scf_energy() const { return true; }
    void update_scf_energy(occ::core::EnergyComponents &energy,
                           bool incremental) const {
        return;
    }
    bool supports_incremental_fock_build() const { return true; }

    void set_system_charge(int charge);
    void set_density_fitting_basis(const std::string &);
    double nuclear_repulsion_energy() const;

    Mat compute_fock(SpinorbitalKind kind, const MolecularOrbitals &mo,
                     double precision = std::numeric_limits<double>::epsilon(),
                     const Mat &Schwarz = Mat()) const;
    std::pair<Mat, Mat>
    compute_JK(SpinorbitalKind kind, const MolecularOrbitals &mo,
               double precision = std::numeric_limits<double>::epsilon(),
               const Mat &Schwarz = Mat()) const;
    Mat compute_J(SpinorbitalKind kind, const MolecularOrbitals &mo,
                  double precision = std::numeric_limits<double>::epsilon(),
                  const Mat &Schwarz = Mat()) const;

    Mat compute_kinetic_matrix() const;
    Mat compute_overlap_matrix() const;
    Mat compute_nuclear_attraction_matrix() const;
    Mat compute_point_charge_interaction_matrix(
        const std::vector<occ::core::PointCharge> &point_charges) const;

    std::vector<Mat>
    compute_kinetic_energy_derivatives(unsigned derivative) const;
    std::vector<Mat>
    compute_nuclear_attraction_derivatives(unsigned derivative) const;
    std::vector<Mat> compute_overlap_derivatives(unsigned derivative) const;

    Mat3N nuclear_electric_field_contribution(const Mat3N &) const;
    Mat3N electronic_electric_field_contribution(SpinorbitalKind kind,
                                                 const MolecularOrbitals &mo,
                                                 const Mat3N &) const;
    Vec electronic_electric_potential_contribution(SpinorbitalKind kind,
                                                   const MolecularOrbitals &mo,
                                                   const Mat3N &) const;
    Vec nuclear_electric_potential_contribution(const Mat3N &) const;

    Mat compute_shellblock_norm(const Mat &A) const;

    auto compute_schwarz_ints() const {
        return occ::ints::compute_schwarz_ints<>(m_basis);
    }

    void update_core_hamiltonian(occ::qm::SpinorbitalKind k,
                                 const MolecularOrbitals &mo, Mat &H) {
        return;
    }

    template <unsigned int order = 1>
    inline auto compute_electronic_multipole_matrices(
        const Vec3 &o = {0.0, 0.0, 0.0}) const {
        std::array<double, 3> c{o(0), o(1), o(2)};
        static_assert(
            order < 4,
            "Multipole integrals with order > 3 are not supported yet");
        constexpr std::array<libint2::Operator, 4> ops{
            Operator::overlap, Operator::emultipole1, Operator::emultipole2,
            Operator::emultipole3};

        constexpr libint2::Operator op = ops[order];
        return compute_1body_ints<op>(m_basis, m_shellpair_list, c);
    }

    template <unsigned int order = 1>
    inline auto compute_electronic_multipoles(occ::qm::SpinorbitalKind k,
                                              const MolecularOrbitals &mo,
                                              const Vec3 &o = {0.0, 0.0,
                                                               0.0}) const {
        occ::core::Multipole<order> result;
        const auto &D = mo.D;
        auto mats = compute_electronic_multipole_matrices<order>(o);
        auto ex = [&](const Mat &op) {
            switch (k) {
            case SpinorbitalKind::Unrestricted:
                return occ::qm::expectation<SpinorbitalKind::Unrestricted>(D,
                                                                           op);
            case SpinorbitalKind::General:
                return occ::qm::expectation<SpinorbitalKind::General>(D, op);
            default:
                return occ::qm::expectation<SpinorbitalKind::Restricted>(D, op);
            }
        };

        for (size_t i = 0; i < mats.size(); i++) {
            result.components[i] = -2 * ex(mats[i]);
        }
        return result;
    }

    template <unsigned int order = 1>
    inline auto compute_nuclear_multipoles(const Vec3 &o = {0.0, 0.0,
                                                            0.0}) const {
        std::array<double, 3> c{o(0), o(1), o(2)};
        auto charges = occ::core::make_point_charges(m_atoms);
        return occ::core::Multipole<order>{
            occ::core::compute_multipoles<order>(charges, c)};
    }

  private:
    int m_charge{0};
    int m_num_e{0};
    std::vector<occ::core::Atom> m_atoms;
    BasisSet m_basis;
    BasisSet m_density_fitting_basis;
    ShellPairList m_shellpair_list{}; // shellpair list for OBS
    ShellPairData m_shellpair_data{}; // shellpair data for OBS
    occ::ints::FockBuilder m_fockbuilder;
    mutable double m_e_alpha{0};
    mutable double m_e_beta{0};
    mutable std::optional<occ::df::DFFockEngine> m_df_fock_engine;
};

} // namespace occ::hf
