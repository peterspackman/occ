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
#include "parallel.h"

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


std::vector<DensityFunctional> parse_method(const std::string& method_string, bool polarized = false);

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
    DFT(const std::string&, const libint2::BasisSet&, const std::vector<libint2::Atom>&, const SpinorbitalKind kind = SpinorbitalKind::Restricted);
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
        using tonto::parallel::nthreads;
        tonto::MatRM K, F;
        m_e_alpha = 0.0;
        double ecoul, exc;
        double exchange_factor = exact_exchange_factor();
        tonto::timing::start(tonto::timing::category::ints);
        if(exchange_factor != 0.0) {
            std::tie(F, K) = m_hf.compute_JK(spinorbital_kind, D, precision, Schwarz);
            ecoul = expectation<spinorbital_kind>(D, F);
            exc = - expectation<spinorbital_kind>(D, K) * exchange_factor;
            F.noalias() -= K * exchange_factor;
        }
        else {
            F = m_hf.compute_J(spinorbital_kind, D, precision, Schwarz);
            ecoul = expectation<spinorbital_kind>(D, F);
        }
        tonto::timing::stop(tonto::timing::category::ints);
        K = tonto::MatRM::Zero(F.rows(), F.cols());
        const auto& basis = m_hf.basis();
        const auto& atoms = m_hf.atoms();
        constexpr size_t BLOCKSIZE = 128;
        double total_density_a{0.0}, total_density_b{0.0};
        const auto& D2 = 2 * D;
        DensityFunctional::Family family{DensityFunctional::Family::LDA};
        if constexpr (derivative_order == 1) {
            family = DensityFunctional::Family::GGA;
        }

        std::vector<tonto::MatRM> Kt(tonto::parallel::nthreads, tonto::Mat::Zero(D.rows(), D.cols()));
        std::vector<double> energies(tonto::parallel::nthreads, 0.0);
        const auto& funcs = m_funcs;
        for(const auto& [atom_pts, atom_weights] : m_atom_grids) {
            const size_t npt_total = atom_pts.cols();
            const size_t num_blocks = npt_total / BLOCKSIZE + 1;

            auto lambda = [&](int thread_id)
            {
                for(size_t block = 0; block < num_blocks; block++) {
                    if(block % nthreads != thread_id) continue;
                    Eigen::Index l = block * BLOCKSIZE;
                    Eigen::Index u = std::min(npt_total - 1, (block + 1) * BLOCKSIZE);
                    Eigen::Index npt = u - l;
                    if(npt == 0) continue;
                    auto& k = Kt[thread_id];
                    const auto& pts_block = atom_pts.middleCols(l, npt);
                    const auto& weights_block = atom_weights.segment(l, npt);
                    DensityFunctional::Params params(npt, family, spinorbital_kind);
                    const auto [rho, gto_vals] = evaluate_density_and_gtos<derivative_order, spinorbital_kind>(basis, atoms, D2, pts_block);

                    if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
                        params.rho.col(0) = rho.col(0);
                        total_density_a += rho.col(0).dot(weights_block);
                    }
                    else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
                        // correct assignment
                        params.rho.col(0) = rho.alpha().col(0);
                        params.rho.col(1) = rho.beta().col(0);
                        total_density_a += rho.alpha().col(0).dot(weights_block);
                        total_density_b += rho.beta().col(0).dot(weights_block);
                    }


                    if constexpr(derivative_order > 0) {
                        if constexpr(spinorbital_kind == SpinorbitalKind::Restricted) {
                            const auto& dx_rho = rho.col(1).array(), dy_rho = rho.col(2).array(), dz_rho = rho.col(3).array();
                            params.sigma.col(0) = dx_rho * dx_rho + dy_rho * dy_rho + dz_rho * dz_rho;
                        }
                        else if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted) {

                            const auto& rho_alpha = rho.alpha();
                            const auto& rho_beta = rho.beta();
                            const auto& dx_rho_a = rho_alpha.col(1).array();
                            const auto& dy_rho_a = rho_alpha.col(2).array();
                            const auto& dz_rho_a = rho_alpha.col(3).array();
                            const auto& dx_rho_b = rho_beta.col(1).array();
                            const auto& dy_rho_b = rho_beta.col(2).array();
                            const auto& dz_rho_b = rho_beta.col(3).array();
                            params.sigma.col(0) = dx_rho_a * dx_rho_a + dy_rho_a * dy_rho_a + dz_rho_a * dz_rho_a;
                            params.sigma.col(1) = dx_rho_a * dx_rho_b + dy_rho_a * dy_rho_b + dz_rho_a * dz_rho_b;
                            params.sigma.col(2) = dx_rho_b * dx_rho_b + dy_rho_b * dy_rho_b + dz_rho_b * dz_rho_b;
                        }
                    }

                    DensityFunctional::Result res(npt, family, spinorbital_kind);
                    for(const auto& func: funcs) {
                        res += func.evaluate(params);
                    }

                    tonto::MatRM KK = tonto::MatRM::Zero(k.rows(), k.cols());

                    // Weight the arrays by the grid weights
                    res.weight_by(weights_block);
                    if constexpr(spinorbital_kind == SpinorbitalKind::Restricted) {
                        energies[thread_id] += rho.col(0).dot(res.exc);
                        tonto::Mat phi_vrho = gto_vals.phi.array().colwise() * res.vrho.col(0).array();
                        KK = gto_vals.phi.transpose() * phi_vrho;
                    } else if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted) {
                        energies[thread_id] += res.exc.dot(rho.alpha().col(0)) + res.exc.dot(rho.beta().col(0));
                        tonto::Mat phi_vrho_a = gto_vals.phi.array().colwise() * res.vrho.col(0).array();
                        tonto::Mat phi_vrho_b = gto_vals.phi.array().colwise() * res.vrho.col(1).array();

                        KK.alpha().noalias() = gto_vals.phi.transpose() * phi_vrho_a;
                        KK.beta().noalias() = gto_vals.phi.transpose() * phi_vrho_b;
                    }

                    if constexpr(derivative_order > 0) {
                        if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
                            auto g = rho.block(0, 1, npt, 3).array().colwise() * (2 * res.vsigma.col(0).array());
                            tonto::Mat gamma = gto_vals.phi_x.array().colwise() * g.col(0).array()
                                             + gto_vals.phi_y.array().colwise() * g.col(1).array()
                                             + gto_vals.phi_z.array().colwise() * g.col(2).array();
                            tonto::Mat ktmp = gto_vals.phi.transpose() * gamma;
                            KK.noalias() += ktmp + ktmp.transpose();

                        } else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
                            auto ga = rho.alpha().block(0, 1, npt, 3).array().colwise() * (2 * res.vsigma.col(0).array()) +
                                            rho.beta().block(0, 1, npt, 3).array().colwise() * res.vsigma.col(1).array();
                            auto gb = rho.beta().block(0, 1, npt, 3).array().colwise() * (2 * res.vsigma.col(2).array()) +
                                            rho.alpha().block(0, 1, npt, 3).array().colwise() * res.vsigma.col(1).array();

                            tonto::Mat gamma_a = gto_vals.phi_x.array().colwise() * ga.col(0).array()
                                               + gto_vals.phi_y.array().colwise() * ga.col(1).array()
                                               + gto_vals.phi_z.array().colwise() * ga.col(2).array();
                            tonto::Mat gamma_b = gto_vals.phi_x.array().colwise() * gb.col(0).array()
                                               + gto_vals.phi_y.array().colwise() * gb.col(1).array()
                                               + gto_vals.phi_z.array().colwise() * gb.col(2).array();
                            tonto::Mat ktmp_a = (gto_vals.phi.transpose() * gamma_a);
                            tonto::Mat ktmp_b = (gto_vals.phi.transpose() * gamma_b);
                            KK.alpha().noalias() += ktmp_a + ktmp_a.transpose();
                            KK.beta().noalias() += ktmp_b + ktmp_b.transpose();
                        }
                    }
                    k.noalias() += KK;
                }
            };
            tonto::parallel::parallel_do(lambda);
            //tonto::log::debug("E_coul: {}, E_x: {}, E_xc = {}, E_XC = {}", ecoul, exc, m_e_alpha, m_e_alpha + exc);
        }

        for(size_t i = 0; i < nthreads; i++) {
            K += Kt[i];
            m_e_alpha += energies[i];
        }
        m_e_alpha += exc + ecoul;
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

    SpinorbitalKind m_spinorbital_kind;
    tonto::hf::HartreeFock m_hf;
    DFTGrid m_grid;
    std::vector<DensityFunctional> m_funcs;
    std::vector<std::pair<tonto::Mat3N, tonto::Vec>> m_atom_grids;
    mutable double m_e_alpha;
    mutable double m_e_beta;
};
}
