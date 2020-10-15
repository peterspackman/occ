#pragma once
#include "linear_algebra.h"
#include "gto.h"

namespace tonto::density {

inline int num_components(int deriv_order) {
    switch(deriv_order) {
    case 0: return 1;
    case 1: return 4;
    case 2: return 10;
    }
    return 1;
}

    void eval_shell(const libint2::Shell &s1, const Eigen::Ref<const tonto::Mat>& dists, Eigen::Ref<tonto::Mat>& result, int derivative=0);
    tonto::MatRM evaluate_gtos(
            const libint2::BasisSet &basis, const std::vector<libint2::Atom> &atoms,
            const tonto::MatN4 &grid_pts, int derivative=0);
    tonto::Mat evaluate(
            const libint2::BasisSet &basis, const std::vector<libint2::Atom> &atoms,
            const tonto::MatRM& D, const tonto::MatN4 &grid_pts, int derivative=0);


    template<size_t max_derivative>
    tonto::Mat evaluate_density(const tonto::MatRM &D, const tonto::gto::GTOValues<max_derivative>& gto_values)
    {
        tonto::Mat Dphi = gto_values.phi * D;
        tonto::Mat rho(Dphi.rows(), num_components(max_derivative));
        rho.col(0).array() = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
        if constexpr(max_derivative > 0) {
            rho.col(1).array() = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
            rho.col(2).array() = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
            rho.col(3).array() = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
        }
        return rho;
    }

    template<size_t max_derivative>
    tonto::Mat evaluate_density_on_grid(
        const libint2::BasisSet &basis, const std::vector<libint2::Atom> &atoms,
        const tonto::MatRM& D, const tonto::Mat &grid_pts)
    {
        auto gto_values = tonto::gto::evaluate_basis_on_grid<max_derivative>(basis, atoms, grid_pts);
        return evaluate_density<max_derivative>(D, gto_values);
    }

}
