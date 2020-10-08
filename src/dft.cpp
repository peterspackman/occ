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
    m_funcs.push_back(DensityFunctional("lda"));
    m_funcs.push_back(DensityFunctional("vwn"));

    for(const auto& func: m_funcs) {
        fmt::print("Functional: {} {} {}\n", func.name(), func.kind_string(), func.family_string());
    }
}


MatRM DFT::compute_2body_fock(const MatRM &D, double precision, const MatRM &Schwarz) const
{
    auto J = m_hf.compute_J(D, precision, Schwarz);
    tonto::MatRM K = tonto::MatRM::Zero(J.rows(), J.cols());
    const auto& basis = m_hf.basis();
    const auto& atoms = m_hf.atoms();
    size_t nbf = basis.nbf();
    double total_density{0.0};
    m_e_alpha = 0.0;
    auto D2 = 2 * D;
    for(const auto& pts : m_atom_grids) {
        size_t npt = pts.rows();
        tonto::Vec rho;
        tonto::Mat gto_vals;
        std::tie(rho, gto_vals) = evaluate_density_and_gtos(basis, atoms, D2, pts, 0);
        tonto::Vec v = tonto::Vec::Zero(rho.rows()), e = tonto::Vec::Zero(rho.rows());
        tonto::Vec vtmp(rho.rows()), etmp(rho.rows());
        for(const auto& func: m_funcs) {
            func.add_energy_potential(rho, etmp, vtmp);
            v += vtmp; e += etmp;
        }
        // add weights contribution
        auto vwt = v.array() * pts.col(3).array();
        auto ewt = e.array() * pts.col(3).array();
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
    m_e_alpha += D.cwiseProduct(J).sum();
    auto F = J + K;
    return F;
}

std::pair<MatRM, MatRM> DFT::compute_JK(const MatRM &D, double precision, const MatRM &Schwarz) const
{
    return m_hf.compute_JK(D, precision, Schwarz);
}

}
