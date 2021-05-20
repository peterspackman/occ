#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/gto/gto.h>
#include <occ/qm/spinorbital.h>

namespace occ::density {
    using occ::qm::SpinorbitalKind;
    using occ::qm::BasisSet;

    constexpr int num_components(int deriv_order) {
        switch(deriv_order) {
        case 0: return 1;
        case 1: return 4;
        case 2: return 10;
        }
        return 1;
    }

    template<size_t max_derivative, SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
    occ::Mat evaluate_density(const occ::MatRM &D, const occ::gto::GTOValues<max_derivative>& gto_values)
    {
        if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted) {
            // alpha part first
            occ::Mat Dphi = gto_values.phi * D.alpha();
            occ::Mat rho(Dphi.rows() * 2, num_components(max_derivative));
            rho.block(0, 0, Dphi.rows(), 1).array() = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
            /*
             * If we wish to get the values interleaved for say libxc, use an Eigen::Map as follows:
             *  Map<occ::Mat, 0, Stride<Dynamic, 2>>(rho.col(1).data(), Dphi.rows(), Dphi.cols(), Stride<Dynamic, 2>(2*Dphi.rows(), 2)) = RHS
             */
            if constexpr(max_derivative > 0) {
                rho.block(0, 1, Dphi.rows(), 1).array() = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
                rho.block(0, 2, Dphi.rows(), 1).array() = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
                rho.block(0, 3, Dphi.rows(), 1).array() = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
            }
            // beta part
            Dphi = gto_values.phi * D.beta();
            rho.block(Dphi.rows(), 0, Dphi.rows(), 1).array() = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
            if constexpr(max_derivative > 0) {
                rho.block(Dphi.rows(), 1, Dphi.rows(), 1).array() = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
                rho.block(Dphi.rows(), 2, Dphi.rows(), 1).array() = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
                rho.block(Dphi.rows(), 3, Dphi.rows(), 1).array() = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
            }
            return rho;
        }
        else {
            occ::Mat Dphi = gto_values.phi * D;
            occ::Mat rho(Dphi.rows(), num_components(max_derivative));
            rho.col(0).array() = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
            if constexpr(max_derivative > 0) {
                rho.col(1).array() = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
                rho.col(2).array() = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
                rho.col(3).array() = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
            }
            return rho;
        }
    }

    template<size_t max_derivative, SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
    occ::Mat evaluate_density_on_grid(
        const BasisSet &basis, const std::vector<libint2::Atom> &atoms,
        const occ::MatRM& D, const occ::Mat &grid_pts)
    {
        auto gto_values = occ::gto::evaluate_basis_on_grid<max_derivative>(basis, atoms, grid_pts);
        return evaluate_density<max_derivative, spinorbital_kind>(D, gto_values);
    }

}