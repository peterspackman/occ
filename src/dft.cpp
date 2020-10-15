#include "dft.h"
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include "logger.h"
#include "density.h"
#include "timings.h"
#include "gto.h"

namespace tonto::dft {

template<int derivative_order>
std::pair<tonto::Mat, tonto::gto::GTOValues<derivative_order>> evaluate_density_and_gtos(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::MatRM& D,
    const tonto::Mat4N &grid_pts)
{
    auto gto_values = tonto::gto::evaluate_basis_on_grid<derivative_order>(basis, atoms, grid_pts);
    auto rho = tonto::density::evaluate_density<derivative_order>(D, gto_values);
    return {rho, gto_values};
}

int DFT::density_derivative() const {
    int deriv = 0;
    for(const auto& func: m_funcs) {
        deriv = std::max(deriv, func.derivative_order());
    }
    return deriv;
}

DFT::DFT(const std::string& method, const libint2::BasisSet& basis, const std::vector<libint2::Atom>& atoms) :
    m_hf(atoms, basis), m_grid(basis, atoms)
{
    tonto::log::debug("start calculating atom grids... ");
    m_grid.set_max_angular_points(530);
    m_grid.set_min_angular_points(80);
    for(size_t i = 0; i < atoms.size(); i++) {
        m_atom_grids.push_back(m_grid.grid_points(i).transpose());
    }
    size_t num_grid_points = std::accumulate(m_atom_grids.begin(), m_atom_grids.end(), 0.0, [&](double tot, const auto& grid) { return tot + grid.rows(); });
    tonto::log::debug("finished calculating atom grids ({} points)", num_grid_points);
    m_funcs.push_back(DensityFunctional("xc_hyb_gga_xc_b3lyp"));
    for(const auto& func: m_funcs) {
        tonto::log::debug("Functional: {} {} {}, exact exchange = {}", func.name(), func.kind_string(), func.family_string(), func.exact_exchange_factor());
    }
    tonto::log::debug("Total exchange factor: {}", exact_exchange_factor());
}


MatRM DFT::compute_2body_fock(const MatRM &D, double precision, const MatRM &Schwarz) const
{
    int d = density_derivative();
    switch(d) {
    case 0: return compute_2body_fock_d0(D, precision, Schwarz);
    case 1: return compute_2body_fock_d1(D, precision, Schwarz);
    case 2: return compute_2body_fock_d2(D, precision, Schwarz);
    default: throw std::runtime_error(fmt::format("Density derivative order {} not implemented", d));
    }
}

MatRM DFT::compute_2body_fock_d0(const MatRM &D, double precision, const MatRM &Schwarz) const
{
    auto J = m_hf.compute_J(D, precision, Schwarz);
    tonto::MatRM K = tonto::MatRM::Zero(J.rows(), J.cols());
    const auto& basis = m_hf.basis();
    const auto& atoms = m_hf.atoms();
    size_t nbf = basis.nbf();
    double total_density{0.0};
    m_e_alpha = 0.0;
    auto D2 = 2 * D;
    DensityFunctional::Params params;
    for(const auto& pts : m_atom_grids) {
        size_t npt = pts.rows();
        auto [rho, gto_vals] = evaluate_density_and_gtos<0>(basis, atoms, D2, pts);
        DensityFunctional::Result res(npt, DensityFunctional::Family::LDA);
        params.rho = rho.col(0);
        for(const auto& func: m_funcs) {
            res += func.evaluate(params);
        }
        // add weights contribution
        auto vwt = res.vrho.array() * pts.col(3).array();
        auto ewt = res.exc.array() * pts.col(3).array();
        total_density += (params.rho.array() * pts.col(3).array()).array().sum();
        m_e_alpha += (params.rho * ewt).sum();
        tonto::Mat phi_vrho = gto_vals.phi;
        for(size_t bf = 0; bf < nbf; bf++) {
            phi_vrho.col(bf).array() *= vwt;
        }
        K.noalias() += (gto_vals.phi * phi_vrho.transpose());
    }
//    fmt::print("Total density {}\n", total_density);
    m_e_alpha += D.cwiseProduct(J).sum();
    auto F = J + K;
    return F;
}

MatRM DFT::compute_2body_fock_d1(const MatRM &D, double precision, const MatRM &Schwarz) const
{
    tonto::MatRM K, F;
    m_e_alpha = 0.0;
    double ecoul, exc;
    double exchange_factor = exact_exchange_factor();
    tonto::timing::start(tonto::timing::category::ints);
    if(exchange_factor != 0.0) {
        std::tie(F, K) = m_hf.compute_JK(D, precision, Schwarz);
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
    size_t nbf = basis.nbf();
    double total_density{0.0};
    auto D2 = 2 * D;
    DensityFunctional::Params params;
    for(const auto& pts : m_atom_grids) {
        size_t npt = pts.rows();
        // Need to sort out Row/Column Major mixing...
        tonto::timing::start(tonto::timing::category::grid);
        auto [rho, gto_vals] = evaluate_density_and_gtos<1>(basis, atoms, D2, pts);
        tonto::timing::stop(tonto::timing::category::grid);
        params.rho = rho.col(0);
        const auto& rho_x = rho.col(1).array(), rho_y = rho.col(2).array(), rho_z = rho.col(3).array();
        params.sigma = rho_x * rho_x + rho_y * rho_y + rho_z * rho_z;
        DensityFunctional::Result res(npt, DensityFunctional::Family::GGA);
        tonto::timing::start(tonto::timing::category::dft);
        for(const auto& func: m_funcs) {
            res += func.evaluate(params);
        }
        tonto::timing::stop(tonto::timing::category::dft);
        // add weights contribution
        auto vwt = res.vrho.array() * pts.col(3).array();
        auto ewt = res.exc.array() * pts.col(3).array();
        auto vsigmawt = res.vsigma.array() * pts.col(3).array();
        total_density += (params.rho.array() * pts.col(3).array()).array().sum();
        m_e_alpha += (params.rho * ewt).sum();
        tonto::Mat& phi = gto_vals.phi;
        tonto::Mat phi_vrho = phi;
        #pragma omp parallel for
        for(size_t bf = 0; bf < nbf; bf++) {
            phi_vrho.row(bf).array() *= vwt;
        }
        K += phi * phi_vrho.transpose();
        // especially here
        tonto::Mat& phi_x = gto_vals.phi_x;
        tonto::Mat& phi_y = gto_vals.phi_y;
        tonto::Mat& phi_z = gto_vals.phi_z;
        #pragma omp parallel for
        for(size_t bf = 0; bf < nbf; bf++) {
            phi.col(bf).array() *= vsigmawt;
            phi_x.col(bf).array() *= 2 * rho_x;
            phi_y.col(bf).array() *= 2 * rho_y;
            phi_z.col(bf).array() *= 2 * rho_z;
        }
        tonto::Mat ktmp((phi * phi_x.transpose()) + (phi * phi_y.transpose()) + (phi * phi_z.transpose()));
        tonto::Mat KK = ktmp + ktmp.transpose();
        K.noalias() += KK;
    }
    //tonto::log::debug("E_coul: {}, E_x: {}, E_xc = {}", ecoul, exc, m_e_alpha);
    fmt::print("\nGTO  {:10.5f}\nfunc {:10.5f}\nfock {:10.5f}\n\n",
                      tonto::timing::total(tonto::timing::category::grid),
                      tonto::timing::total(tonto::timing::category::dft),
                      tonto::timing::total(tonto::timing::category::ints));
    tonto::timing::clear_all();
    m_e_alpha += D.cwiseProduct(F).sum();
    F += K;
    return F;
}

MatRM DFT::compute_2body_fock_d2(const MatRM &D, double precision, const MatRM &Schwarz) const
{
    throw std::runtime_error("Density derivative d2 fock not implemented");
}

std::pair<MatRM, MatRM> DFT::compute_JK(const MatRM &D, double precision, const MatRM &Schwarz) const
{
    return m_hf.compute_JK(D, precision, Schwarz);
}

}
