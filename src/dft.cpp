#include "dft.h"
#include <libint2/basis.h>
#include <libint2/atom.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include "logger.h"
#include "density.h"

namespace tonto::dft {

std::pair<tonto::Vec, tonto::Mat> evaluate_density_and_gtos(
    const libint2::BasisSet &basis,
    const std::vector<libint2::Atom> &atoms,
    const tonto::MatRM& D,
    const tonto::MatN4 &grid_pts,
    int derivative)
{
    int n_components = tonto::density::num_components(derivative);
    tonto::Vec rho = tonto::Vec::Zero(grid_pts.rows()* n_components);
    auto gto_vals = tonto::density::evaluate_gtos(basis, atoms, grid_pts, derivative);
    for(int bf1 = 0; bf1 < gto_vals.rows(); bf1++) {
        for(int bf2 = bf1; bf2 < gto_vals.rows(); bf2++) {
            if(bf1 == bf2) rho.array() += D(bf1, bf2) * gto_vals.row(bf1).array() * gto_vals.row(bf2).array();
            else rho.array() += 2 * D(bf1, bf2) * gto_vals.row(bf1).array() * gto_vals.row(bf2).array();
        }
    }
    return std::make_pair(rho, gto_vals);
}

DFT::DFT(const std::string& method, const libint2::BasisSet& basis, const std::vector<libint2::Atom>& atoms) :
    m_hf(atoms, basis), m_grid(basis, atoms)
{
    tonto::log::debug("start calculating atom grids... ");
    m_grid.set_max_angular_points(131);
    m_grid.set_min_angular_points(33);
    for(size_t i = 0; i < atoms.size(); i++) {
        m_atom_grids.push_back(m_grid.grid_points(i));
    }
    tonto::log::debug("finished calculating atom grids");
    m_funcs.push_back(DensityFunctional("LDA"));
}


MatRM DFT::compute_2body_fock(const MatRM &D, double precision, const MatRM &Schwarz) const
{
    auto J = m_hf.compute_J(D, precision, Schwarz);
    const auto& basis = m_hf.basis();
    const auto& atoms = m_hf.atoms();
    size_t nbf = basis.nbf();
    fmt::print("xc_func: {}\n", m_funcs[0].family_string());
    for(const auto& pts : m_atom_grids) {
        tonto::Vec rho;
        tonto::Mat gto_vals;
        std::tie(rho, gto_vals) = evaluate_density_and_gtos(basis, atoms, D, pts, 0);
        rho.array() = rho.array() * 2;
        fmt::print("n_funcs: {}, rho max: {}, rho.shape {} {}\n", m_funcs.size(), rho.maxCoeff(), rho.rows(), rho.cols());
        fmt::print("rho(0): {} rho.data()[0] {}\n", rho[0], rho.data()[0]);
        tonto::Vec v = m_funcs[0].potential(rho);
        v.array() *= pts.col(3).array();
        fmt::print("v_max = {} v_min = {}, v.shape {} {}\n", v.maxCoeff(), v.minCoeff(), v.rows(), v.cols());
        for(size_t bf1 = 0; bf1 < nbf; bf1++) {
            const auto x1 = gto_vals.row(bf1);
            for(size_t bf2 = bf1; bf2 < nbf; bf2++) {
                const auto x2 = gto_vals.row(bf2);
                double coeff = (x1.array() * x2.array() * v.array()).sum();
                J(bf1, bf2) += coeff;
                if(bf1 != bf2) J(bf2, bf1) += coeff;
            }
        }
    }
    return J;
}

std::pair<MatRM, MatRM> DFT::compute_JK(const MatRM &D, double precision, const MatRM &Schwarz) const
{
    return m_hf.compute_JK(D, precision, Schwarz);
}

}
