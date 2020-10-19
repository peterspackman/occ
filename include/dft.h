#pragma once
#include "linear_algebra.h"
#include <vector>
#include <string>
#include "dft_grid.h"
#include "density_functional.h"
#include "ints.h"
#include "spinorbital.h"
#include "timings.h"
#include "hf.h"
#include "gto.h"
#include "density.h"

namespace libint2 {
class BasisSet;
class Atom;
}

namespace tonto::dft {
using tonto::qm::SpinorbitalKind;
using tonto::qm::expectation;

using tonto::Mat3N;
using tonto::MatRM;
using tonto::MatN4;
using tonto::Vec;
using tonto::IVec;
using tonto::MatRM;
using tonto::ints::BasisSet;
using tonto::ints::compute_1body_ints;
using tonto::ints::compute_1body_ints_deriv;
using tonto::ints::Operator;
using tonto::ints::shellpair_data_t;
using tonto::ints::shellpair_list_t;


std::vector<DensityFunctional> parse_method(const std::string& method_string);

template<int derivative_order, SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
std::pair<tonto::Mat, tonto::gto::GTOValues<derivative_order>> evaluate_density_and_gtos(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::MatRM& D,
    const tonto::Mat &grid_pts)
{
    auto gto_values = tonto::gto::evaluate_basis_on_grid<derivative_order>(basis, atoms, grid_pts);
    auto rho = tonto::density::evaluate_density<derivative_order, spinorbital_kind>(D, gto_values);
    return {rho, gto_values};
}

class DFT {

public:
    DFT(const std::string&, const libint2::BasisSet&, const std::vector<libint2::Atom>&);
    const auto &shellpair_list() const { return m_hf.shellpair_list(); }
    const auto &shellpair_data() const { return m_hf.shellpair_data(); }
    const auto &atoms() const { return m_hf.atoms(); }
    const auto &basis() const { return m_hf.basis(); }

    void set_system_charge(int charge) {
        m_hf.set_system_charge(charge);
    }
    int system_charge() const { return m_hf.system_charge(); }
    int num_e() const { return m_hf.num_e(); }

    double two_electron_energy() const { return m_e_alpha + m_e_beta; }
    double two_electron_energy_alpha() const { return m_e_alpha; }
    double two_electron_energy_beta() const { return m_e_beta; }
    bool usual_scf_energy() const { return false; }

    int density_derivative() const;
    double exact_exchange_factor() const {
        return std::accumulate(m_funcs.begin(), m_funcs.end(), 0.0,
                               [&](double a, const auto& v) { return a + v.exact_exchange_factor(); });
    }

    double nuclear_repulsion_energy() const { return m_hf.nuclear_repulsion_energy(); }
    auto compute_kinetic_matrix() {
      return m_hf.compute_kinetic_matrix();
    }
    auto compute_overlap_matrix() {
      return m_hf.compute_overlap_matrix();
    }
    auto compute_nuclear_attraction_matrix() {
      return m_hf.compute_nuclear_attraction_matrix();
    }

    auto compute_kinetic_energy_derivatives(unsigned derivative) {
      return m_hf.compute_kinetic_energy_derivatives(derivative);
    }

    auto compute_nuclear_attraction_derivatives(unsigned derivative) {
      return m_hf.compute_nuclear_attraction_derivatives(derivative);
    }

    auto compute_overlap_derivatives(unsigned derivative) {
      return m_hf.compute_overlap_derivatives(derivative);
    }

    MatRM compute_shellblock_norm(const MatRM &A) const {
        return m_hf.compute_shellblock_norm(A);
    }

    auto compute_schwarz_ints() {
      return m_hf.compute_schwarz_ints();
    }

    template<int derivative_order, SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
    MatRM compute_fock_dft(const MatRM &D, double precision, const MatRM& Schwarz)
    {
        tonto::MatRM K, F;
        m_e_alpha = 0.0;
        double ecoul, exc;
        double exchange_factor = exact_exchange_factor();
        tonto::timing::start(tonto::timing::category::ints);
        if(exchange_factor != 0.0) {
            std::tie(F, K) = m_hf.compute_JK(spinorbital_kind, D, precision, Schwarz);
            ecoul = D.cwiseProduct(F).sum();
            exc = - D.cwiseProduct(K).sum() * exchange_factor;
            F -= K * exchange_factor;
        }
        else {
            F = m_hf.compute_J(D, precision, Schwarz);
        }
        tonto::timing::stop(tonto::timing::category::ints);
        K = tonto::MatRM::Zero(F.rows(), F.cols());
        const auto& basis = m_hf.basis();
        const auto& atoms = m_hf.atoms();
        constexpr size_t BLOCKSIZE = 1024;
        double total_density{0.0};
        auto D2 = 2 * D;
        DensityFunctional::Family family{DensityFunctional::Family::LDA};

        if constexpr (derivative_order == 1) {
            family = DensityFunctional::Family::GGA;
        }

        DensityFunctional::Params params;
        for(const auto& [pts, weights] : m_atom_grids) {
            size_t npt = pts.cols();
            tonto::timing::start(tonto::timing::category::grid);
            const auto [rho, gto_vals] = evaluate_density_and_gtos<derivative_order>(basis, atoms, D2, pts);
            tonto::timing::stop(tonto::timing::category::grid);
            if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) params.rho = rho.col(0);
            else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
                using Eigen::Map;
                using Eigen::Stride;
                using Eigen::Dynamic;
                params.rho = tonto::Vec(npt * 2);
                Map<tonto::Mat, 0, Stride<Dynamic, 2>>(params.rho.data(), npt, 1, Stride<Dynamic, 2>(npt * 2, 2)) = rho.alpha();
                Map<tonto::Mat, 0, Stride<Dynamic, 2>>(params.rho.data() + 1, npt, 1, Stride<Dynamic, 2>(npt * 2, 2)) = rho.beta();
            }

            if constexpr(derivative_order > 0) {
                if constexpr(spinorbital_kind == SpinorbitalKind::Restricted) {
                    const auto& rho_x = rho.col(1).array(), rho_y = rho.col(2).array(), rho_z = rho.col(3).array();
                    params.sigma = rho_x * rho_x + rho_y * rho_y + rho_z * rho_z;
                }
                else if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted) {
                    using Eigen::Map;
                    using Eigen::Stride;
                    using Eigen::Dynamic;
                    const auto& rho_alpha = rho.alpha();
                    const auto& rho_beta = rho.beta();
                    const auto& rho_alpha_x = rho_alpha.col(1).array(), rho_alpha_y = rho_alpha.col(2).array(), rho_alpha_z = rho_alpha.col(3).array();
                    const auto& rho_beta_x = rho_beta.col(1).array(), rho_beta_y = rho_beta.col(2).array(), rho_beta_z = rho_beta.col(3).array();
                    params.sigma = tonto::Vec(npt * 3);
                    Map<tonto::Mat, 0, Stride<Dynamic, 3>>(params.sigma.data(), npt, 1, Stride<Dynamic, 3>(npt * 3, 3)) =
                            rho_alpha_x * rho_alpha_x + rho_alpha_y * rho_alpha_y + rho_alpha_z * rho_alpha_z;
                    Map<tonto::Mat, 0, Stride<Dynamic, 3>>(params.sigma.data() + 1, npt, 1, Stride<Dynamic, 3>(npt * 3, 3)) =
                            rho_alpha_x * rho_beta_x + rho_alpha_y * rho_beta_y + rho_alpha_z * rho_beta_z;
                    Map<tonto::Mat, 0, Stride<Dynamic, 3>>(params.sigma.data() + 2, npt, 1, Stride<Dynamic, 3>(npt * 3, 3)) =
                            rho_beta_x * rho_beta_x + rho_beta_y * rho_beta_y + rho_beta_z * rho_beta_z;
                }
            }

            DensityFunctional::Result res(npt, family);
            tonto::timing::start(tonto::timing::category::dft);
            for(const auto& func: m_funcs) {
                res += func.evaluate(params);
            }
            tonto::timing::stop(tonto::timing::category::dft);
            Vec vwt = res.vrho.array() * weights.array();
            Vec ewt = res.exc.array() * weights.array();

            total_density += (params.rho.array() * weights.array()).array().sum();
            m_e_alpha += params.rho.dot(ewt);
            tonto::Mat phi_vrho = gto_vals.phi.array().colwise() * vwt.array();

            tonto::MatRM KK = gto_vals.phi.transpose() * phi_vrho;

            if constexpr(derivative_order > 0) {
                if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
                    Vec vsigmawt = res.vsigma.array() * weights.array();
                    tonto::Mat phi_xyz = 2 * (gto_vals.phi_x.array().colwise() * rho.col(1).array())
                                       + 2 * (gto_vals.phi_y.array().colwise() * rho.col(2).array())
                                       + 2 * (gto_vals.phi_z.array().colwise() * rho.col(3).array());
                    tonto::Mat phi_vsigma = gto_vals.phi.array().colwise() * vsigmawt.array();
                    tonto::Mat ktmp = phi_vsigma.transpose() * phi_xyz;
                    KK.noalias() += ktmp + ktmp.transpose();
                } else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
                    // TODO
                }
            }
            K.noalias() += KK;
        }
     //   tonto::log::debug("E_coul: {}, E_x: {}, E_xc = {}, E_XC = {}", ecoul, exc, m_e_alpha, m_e_alpha + exc);
        fmt::print("\nGTO  {:10.5f}\nfunc {:10.5f}\nfock {:10.5f}\n\n",
                        tonto::timing::total(tonto::timing::category::grid),
                        tonto::timing::total(tonto::timing::category::dft),
                        tonto::timing::total(tonto::timing::category::ints));
        tonto::timing::clear_all();
        m_e_alpha += expectation<spinorbital_kind>(D, F);
        F += K;
        return F;
    }


    MatRM compute_fock(SpinorbitalKind kind, const MatRM& D, double precision, const MatRM& Schwarz)
    {
        int deriv = density_derivative();
        switch (kind) {
        case SpinorbitalKind::Unrestricted: {
            switch (deriv) {
                case 0: return compute_fock_dft<0, SpinorbitalKind::Unrestricted>(D, precision, Schwarz);
                case 1: return compute_fock_dft<1, SpinorbitalKind::Unrestricted>(D, precision, Schwarz);
                default: throw std::runtime_error("Not implemented: DFT for derivative order > 1");
            }
        }
        case SpinorbitalKind::Restricted: {
            switch (deriv) {
                case 0: return compute_fock_dft<0, SpinorbitalKind::Restricted>(D, precision, Schwarz);
                case 1: return compute_fock_dft<1, SpinorbitalKind::Restricted>(D, precision, Schwarz);
                default: throw std::runtime_error("Not implemented: DFT for derivative order > 1");
            }
        }
        default: throw std::runtime_error("Not implemented: DFT for General spinorbitals");
        }
    }
private:


    tonto::hf::HartreeFock m_hf;
    DFTGrid m_grid;
    std::vector<DensityFunctional> m_funcs;
    std::vector<std::pair<tonto::Mat3N, tonto::Vec>> m_atom_grids;
    mutable double m_e_alpha;
    mutable double m_e_beta;
};
}
