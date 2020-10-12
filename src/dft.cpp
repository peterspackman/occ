#include "dft.h"
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include "logger.h"
#include "density.h"

namespace tonto::dft {

template<int derivative_order>
std::pair<tonto::Mat, tonto::Mat> evaluate_density_and_gtos(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::MatRM& D,
    const tonto::MatN4 &grid_pts)
{
    int n_components = tonto::density::num_components(derivative_order);
    tonto::Mat rho = tonto::Mat::Zero(n_components, grid_pts.rows());
    Eigen::Map<tonto::Vec> flat(rho.data(), rho.size());
    auto gto_vals = tonto::density::evaluate_gtos(basis, atoms, grid_pts, derivative_order);
    for(int bf1 = 0; bf1 < gto_vals.rows(); bf1++) {
        for(int bf2 = bf1; bf2 < gto_vals.rows(); bf2++) {
            if(bf1 == bf2) flat.array() += D(bf1, bf2) * gto_vals.row(bf1).array() * gto_vals.row(bf2).array();
            else flat.array() += 2 * D(bf1, bf2) * gto_vals.row(bf1).array() * gto_vals.row(bf2).array();
        }
    }
    return std::make_pair(rho.transpose(), gto_vals);
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
    m_grid.set_max_angular_points(30);
    m_grid.set_min_angular_points(10);
    for(size_t i = 0; i < atoms.size(); i++) {
        m_atom_grids.push_back(m_grid.grid_points(i));
    }
    tonto::log::debug("finished calculating atom grids");
    m_funcs.push_back(DensityFunctional("xc_lda_x"));
    m_funcs.push_back(DensityFunctional("xc_lda_c_vwn"));

    for(const auto& func: m_funcs) {
        fmt::print("Functional: {} {} {}\n", func.name(), func.kind_string(), func.family_string());
    }
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
        tonto::Vec rho;
        tonto::Mat gto_vals;
        std::tie(rho, gto_vals) = evaluate_density_and_gtos<0>(basis, atoms, D2, pts);
        DensityFunctional::Result res(npt, DensityFunctional::Family::LDA);
        params.rho = rho;
        for(const auto& func: m_funcs) {
            res += func.evaluate(params);
        }
        // add weights contribution
        auto vwt = res.vrho.array() * pts.col(3).array();
        auto ewt = res.exc.array() * pts.col(3).array();
        total_density += (rho.array() * pts.col(3).array()).array().sum();
        for(size_t bf1 = 0; bf1 < nbf; bf1++) {
            for(size_t bf2 = bf1; bf2 < nbf; bf2++) {
                double val = 0.0;
                double wal = 0.0;
                double Dab = D2(bf1, bf2);
                for(size_t pt = 0; pt < npt; pt++) {
                    double gab = gto_vals(bf1, pt) * gto_vals(bf2, pt);
                    val += gab * vwt(pt);
                    wal += gab * ewt(pt);
                }
                m_e_alpha += Dab * wal;
                K(bf1, bf2) += val;
                if(bf1 != bf2) {
                    K(bf2, bf1) += val;
                    m_e_alpha += Dab * wal;
                }
            }
        }
    }
    fmt::print("Total density {}\n", total_density);
    m_e_alpha += D.cwiseProduct(J).sum();
    auto F = J + K;
    return F;
}

MatRM DFT::compute_2body_fock_d1(const MatRM &D, double precision, const MatRM &Schwarz) const
{
    tonto::log::debug("In compute_2body_fock_d1");
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
        tonto::Mat rho;
        tonto::Mat gto_vals;
        std::tie(rho, gto_vals) = evaluate_density_and_gtos<1>(basis, atoms, D2, pts);
        fmt::print("rho.shape {} {}\n", rho.rows(), rho.cols());
        params.rho = rho.col(0);
        params.sigma.resize(rho.rows(), 1);
        for(Eigen::Index i = 0; i < rho.rows(); i++) {
            params.sigma(i) = rho(i, 1) * rho(i, 1) + rho(i, 2) * rho(i, 2) + rho(i, 3) * rho(i, 3);
        }
        DensityFunctional::Result res(npt, DensityFunctional::Family::GGA);
        for(const auto& func: m_funcs) {
            res += func.evaluate(params);
        }
        // add weights contribution
        auto vwt = res.vrho * pts.col(3).array();
        auto ewt = res.exc * pts.col(3).array();
        auto vsigmawt = res.vsigma * pts.col(3).array();
        total_density += (params.rho * pts.col(3).array()).array().sum();
        for(size_t bf1 = 0; bf1 < nbf; bf1++) {
            for(size_t bf2 = bf1; bf2 < nbf; bf2++) {
                double val = 0.0;
                double wal = 0.0;
                double Dab = D2(bf1, bf2);
                for(size_t pt = 0; pt < npt; pt++) {
                    double gax = gto_vals(bf1, pt + 1), gay = gto_vals(bf1, pt + 2), gaz = gto_vals(bf1, pt + 3);
                    double gbx = gto_vals(bf2, pt + 1), gby = gto_vals(bf1, pt + 2), gbz = gto_vals(bf1, pt + 3);
                    double ga = gto_vals(bf1, pt), gb = gto_vals(bf2, pt);
                    double gab = ga * gb;
                    val += gab * vwt(pt) + 2 * vsigmawt(pt) * (
                        ga * (rho(pt, 1) * gbx + rho(pt, 2) * gby + rho(pt, 3) * gbz) +
                        gb * (rho(pt, 1) * gax + rho(pt, 2) * gay + rho(pt, 3) * gaz)
                    );
                    wal += gab * ewt(pt);
                }
                m_e_alpha += Dab * wal;
                K(bf1, bf2) += val;
                if(bf1 != bf2) {
                    K(bf2, bf1) += val;
                    m_e_alpha += Dab * wal;
                }
            }
        }
    }
    fmt::print("Total density: {}\n", total_density);
    m_e_alpha += D.cwiseProduct(J).sum();
    auto F = J + K;
    fmt::print("D\n{}\n", D);
    fmt::print("J\n{}\n", J);
    fmt::print("K\n{}\n", K);
    fmt::print("F\n{}\n", F);
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
