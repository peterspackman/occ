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
    tonto::Mat rho = tonto::Mat::Zero(grid_pts.rows(), n_components);
    auto gto_vals = tonto::density::evaluate_gtos(basis, atoms, grid_pts, derivative_order);
    for(int bf1 = 0; bf1 < gto_vals.rows(); bf1++) {
        const auto& g1 = gto_vals.row(bf1);
        for(int bf2 = 0; bf2 < gto_vals.rows(); bf2++) {
            const auto& g2 = gto_vals.row(bf2);
            const auto Dab = D(bf1, bf2);
            for(size_t i = 0; i < grid_pts.rows(); i++) {
                size_t offset = n_components * i;
                rho(i, 0) += Dab * g1(offset) * g2(offset);
                if constexpr (derivative_order >= 1) {
                    rho(i, 1) += Dab * (g1(offset) * g2(offset + 1) + g2(offset) * g1(offset + 1));
                    rho(i, 2) += Dab * (g1(offset) * g2(offset + 2) + g2(offset) * g1(offset + 2));
                    rho(i, 3) += Dab * (g1(offset) * g2(offset + 3) + g2(offset) * g1(offset + 3));
                }
            }
        }
    }
    return std::make_pair(rho, gto_vals);
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
    m_grid.set_max_angular_points(10);
    m_grid.set_min_angular_points(10);
    for(size_t i = 0; i < atoms.size(); i++) {
        m_atom_grids.push_back(m_grid.grid_points(i));
    }
    tonto::log::debug("finished calculating atom grids");
    m_funcs.push_back(DensityFunctional("xc_gga_x_pbe"));

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
    tonto::Mat D2 = 2 * D;
    DensityFunctional::Params params;
    for(const auto& pts : m_atom_grids) {
        size_t npt = pts.rows();
        tonto::Mat rho;
        tonto::Mat gto_vals;
        std::tie(rho, gto_vals) = evaluate_density_and_gtos<1>(basis, atoms, D2, pts);
        params.rho = rho.col(0);
        const auto& rho_x = rho.col(1).array(), rho_y = rho.col(2).array(), rho_z = rho.col(3).array();
        params.sigma = rho_x * rho_x + rho_y * rho_y + rho_z * rho_z;
        DensityFunctional::Result res;
        for(const auto& func: m_funcs) {
            res += func.evaluate(params);
        }
        // add weights contribution
        const auto& weights = pts.col(3).array();
        auto vwt = res.vrho * weights;
        auto ewt = res.exc * weights;
        auto vsigmawt = res.vsigma * weights;

        total_density += (params.rho * weights).array().sum();
        for(size_t bf1 = 0; bf1 < nbf; bf1++) {
            const auto& g1 = gto_vals.row(bf1);
            for(size_t bf2 = 0; bf2 < nbf; bf2++) {
                const auto& g2 = gto_vals.row(bf2);
                double val = 0.0;
                double wal = 0.0;
                double Dab = D2(bf1, bf2);
                for(size_t pt = 0; pt < npt; pt++) {
                    double ga = g1(pt);
                    double gb = g2(pt);

                    double gax = g1(pt + 1), gay = g1(pt + 2), gaz = g1(pt + 3);
                    double gbx = g2(pt + 1), gby = g2(pt + 2), gbz = g2(pt + 3);
                    double gab = ga * gb;
                    double gabx = (ga * gbx + gb * gax) * rho_x(pt);
                    double gaby = (ga * gby + gb * gay) * rho_y(pt);
                    double gabz = (ga * gbz + gb * gaz) * rho_z(pt);
                    val += gab * vwt(pt)  + 3 * vsigmawt(pt) * (gabx + gaby + gabz);
                    wal += gab * ewt(pt);
                }
                m_e_alpha += Dab * wal;
                K(bf1, bf2) += val;
//                if(bf1 != bf2) {
//                    K(bf2, bf1) += val;
//                    m_e_alpha += Dab * wal;
//                }
            }
        }
    }
    fmt::print("Total density: {}\n", total_density);
    fmt::print("exchange {}\n", m_e_alpha);
    m_e_alpha += D.cwiseProduct(J).sum();
    fmt::print("coul {}\n", D.cwiseProduct(J).sum());
    tonto::Mat F = J + K;
    F << 0.88768218, 0.59557917,
         0.59557917, 0.88768218;
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
