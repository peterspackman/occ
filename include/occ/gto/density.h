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
    void evaluate_density(const Mat &D, const occ::gto::GTOValues& gto_values, Mat &rho)
    {
        //use a MatRM as a row major temporary, selfadjointView also speeds things up a little.
        if constexpr(spinorbital_kind == SpinorbitalKind::Unrestricted) {
            // alpha part first
            MatRM Dphi = gto_values.phi * D.alpha().selfadjointView<Eigen::Upper>();
            if(rho.rows() != gto_values.phi.rows() * 2) rho.resize(gto_values.phi.rows() * 2, num_components(max_derivative));
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
            Dphi = gto_values.phi * D.beta().selfadjointView<Eigen::Upper>();
            rho.block(gto_values.phi.rows(), 0, gto_values.phi.rows(), 1).array() = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
            if constexpr(max_derivative > 0) {
                rho.block(Dphi.rows(), 1, Dphi.rows(), 1).array() = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
                rho.block(Dphi.rows(), 2, Dphi.rows(), 1).array() = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
                rho.block(Dphi.rows(), 3, Dphi.rows(), 1).array() = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
            }
        }
        else {
            MatRM Dphi = gto_values.phi * D.selfadjointView<Eigen::Upper>();
            if (rho.rows() != gto_values.phi.rows()) rho.resize(gto_values.phi.rows(), num_components(max_derivative));
            rho.col(0).array() = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
            if constexpr(max_derivative > 0) {
                rho.col(1).array() = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
                rho.col(2).array() = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
                rho.col(3).array() = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
            }
        }

    }

    template<size_t max_derivative, SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
    Mat evaluate_density(const Mat &D, const occ::gto::GTOValues& gto_values)
    {
        occ::Mat rho;
        evaluate_density<max_derivative, spinorbital_kind>(D, gto_values, rho);
        return rho;
    }

    template<size_t max_derivative, SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
    Mat evaluate_density_on_grid(
        const BasisSet &basis, const std::vector<occ::core::Atom> &atoms,
        const Mat& D, const occ::Mat &grid_pts)
    {
        auto gto_values = occ::gto::evaluate_basis(basis, atoms, grid_pts, max_derivative);
        return evaluate_density<max_derivative, spinorbital_kind>(D, gto_values);
    }

}
