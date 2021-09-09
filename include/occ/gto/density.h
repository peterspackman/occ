#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/gto/gto.h>
#include <occ/qm/spinorbital.h>

namespace occ::density {
using occ::qm::BasisSet;
using occ::qm::SpinorbitalKind;

constexpr int num_components(int deriv_order) {
    switch (deriv_order) {
    case 0:
        return 1;
    case 1:
        return 4;
    case 2:
        return 6;
    }
    return 1;
}

template <size_t max_derivative,
          SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
void evaluate_density(Eigen::Ref<const Mat> D,
                      const occ::gto::GTOValues &gto_values, Mat &rho) {
    // use a MatRM as a row major temporary, selfadjointView also speeds things
    // up a little.
    auto npt = gto_values.phi.rows();
    auto nbf = gto_values.phi.cols();
    if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
        // alpha part first
        auto Da = occ::qm::block::a(D);
        MatRM Dphi = gto_values.phi * Da;
        if (rho.rows() != npt * 2)
            rho.resize(npt * 2, num_components(max_derivative));
        auto rho_a = occ::qm::block::a(rho);
        rho_a.col(0) =
            (gto_values.phi.array() * Dphi.array()).rowwise().sum();
        /*
         * If we wish to get the values interleaved for say libxc, use an
         * Eigen::Map as follows: Map<occ::Mat, 0, Stride<Dynamic,
         * 2>>(rho.col(1).data(), Dphi.rows(), Dphi.cols(), Stride<Dynamic,
         * 2>(2*Dphi.rows(), 2)) = RHS
         */
        if constexpr (max_derivative > 0) {
            rho_a.col(1) =
                2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
            rho_a.col(2) =
                2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
            rho_a.col(3) =
                2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
        }
        if constexpr (max_derivative > 1) {
            // laplacian
            rho_a.col(4) =
                2 * ((gto_values.phi_xx.array() + gto_values.phi_yy.array() +
                      gto_values.phi_zz.array()) *
                     Dphi.array())
                        .rowwise()
                        .sum();
            // tau
            Dphi = gto_values.phi_x * Da;
            rho_a.col(5) =
                (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
            Dphi = gto_values.phi_y * Da;
            rho_a.col(5).array() +=
                (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
            Dphi = gto_values.phi_z * Da;
            rho_a.col(5).array() +=
                (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
            rho_a.col(4).array() += 2 * rho.col(5).array();
            rho_a.col(5).array() *= 0.5;
        }
        // beta part
        auto Db = occ::qm::block::b(D);
        Dphi = gto_values.phi * Db;
        auto rho_b = occ::qm::block::b(rho);
        rho_b.col(0) =
            (gto_values.phi.array() * Dphi.array()).rowwise().sum();
        if constexpr (max_derivative > 0) {
            rho_b.col(1) =
                2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
            rho_b.col(2) =
                2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
            rho_b.col(3) =
                2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
        }
        if constexpr (max_derivative > 1) {
            // laplacian
            rho_b.col(4) =
                2 * ((gto_values.phi_xx.array() + gto_values.phi_yy.array() +
                      gto_values.phi_zz.array()) *
                     Dphi.array())
                        .rowwise()
                        .sum();
            // tau
            Dphi = gto_values.phi_x * Db;
            rho_b.col(5) =
                (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
            Dphi = gto_values.phi_y * Db;
            rho_b.col(5).array() +=
                (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
            Dphi = gto_values.phi_z * Db;
            rho_b.col(5).array() +=
                (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
            rho_b.col(4).array() += 2 * rho.col(5).array();
            rho_b.col(5).array() *= 0.5;
        }
    } else {
        MatRM Dphi = gto_values.phi * D.selfadjointView<Eigen::Upper>();
        if (rho.rows() != npt)
            rho.resize(npt, num_components(max_derivative));
        rho.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
        if constexpr (max_derivative > 0) {
            rho.col(1) =
                2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
            rho.col(2) =
                2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
            rho.col(3) =
                2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
        }
        if constexpr (max_derivative > 1) {
            // laplacian
            rho.col(4) =
                2 * ((gto_values.phi_xx.array() + gto_values.phi_yy.array() +
                      gto_values.phi_zz.array()) *
                     Dphi.array())
                        .rowwise()
                        .sum();
            // tau
            Dphi = gto_values.phi_x * D;
            rho.col(5) =
                (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
            Dphi = gto_values.phi_y * D;
            rho.col(5).array() +=
                (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
            Dphi = gto_values.phi_z * D;
            rho.col(5).array() +=
                (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();

            rho.col(4).array() += 2 * rho.col(5).array();
            rho.col(5).array() *= 0.5;
        }
    }
}

template <size_t max_derivative,
          SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
Mat evaluate_density(const Mat &D, const occ::gto::GTOValues &gto_values) {
    occ::Mat rho;
    evaluate_density<max_derivative, spinorbital_kind>(D, gto_values, rho);
    return rho;
}

template <size_t max_derivative,
          SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted>
Mat evaluate_density_on_grid(const BasisSet &basis,
                             const std::vector<occ::core::Atom> &atoms,
                             const Mat &D, const occ::Mat &grid_pts) {
    auto gto_values =
        occ::gto::evaluate_basis(basis, atoms, grid_pts, max_derivative);
    return evaluate_density<max_derivative, spinorbital_kind>(D, gto_values);
}

} // namespace occ::density
