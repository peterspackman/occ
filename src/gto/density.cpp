#include <occ/gto/density.h>

namespace occ::density {

constexpr auto R = occ::qm::SpinorbitalKind::Restricted;
constexpr auto U = occ::qm::SpinorbitalKind::Unrestricted;

template <>
void evaluate_density<0, R>(MatConstRef D,
                            const occ::gto::GTOValues &gto_values, MatRef rho) {
    // use a MatRM as a row major temporary, selfadjointView also speeds things
    // up a little.
    // Genuine bottleneck ~ 50% of the time for DFT XC is spent here for hybrids
    MatRM Dphi = gto_values.phi * D.selfadjointView<Eigen::Upper>();
    rho.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
}

template <>
void evaluate_density<0, U>(MatConstRef D,
                            const occ::gto::GTOValues &gto_values, MatRef rho) {
    // alpha part first
    auto Da = occ::qm::block::a(D);
    MatRM Dphi = gto_values.phi * Da;
    auto rho_a = occ::qm::block::a(rho);
    rho_a.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
    auto Db = occ::qm::block::b(D);
    Dphi = gto_values.phi * Db;
    auto rho_b = occ::qm::block::b(rho);
    rho_b.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
}

template <>
void evaluate_density<1, R>(MatConstRef D,
                            const occ::gto::GTOValues &gto_values, MatRef rho) {
    // use a MatRM as a row major temporary, selfadjointView also speeds things
    // up a little.
    // Genuine bottleneck ~ 50% of the time for DFT XC is spent here for hybrids
    MatRM Dphi = gto_values.phi * D.selfadjointView<Eigen::Upper>();
    rho.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
    rho.col(1) = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
    rho.col(2) = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
    rho.col(3) = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
}

template <>
void evaluate_density<1, U>(MatConstRef D,
                            const occ::gto::GTOValues &gto_values, MatRef rho) {
    // alpha part first
    auto Da = occ::qm::block::a(D);
    MatRM Dphi = gto_values.phi * Da;
    auto rho_a = occ::qm::block::a(rho);
    rho_a.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
    /*
     * If we wish to get the values interleaved for say libxc, use an
     * Eigen::Map as follows: Map<occ::Mat, 0, Stride<Dynamic,
     * 2>>(rho.col(1).data(), Dphi.rows(), Dphi.cols(), Stride<Dynamic,
     * 2>(2*Dphi.rows(), 2)) = RHS
     */
    rho_a.col(1) =
        2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
    rho_a.col(2) =
        2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
    rho_a.col(3) =
        2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
    // beta part
    auto Db = occ::qm::block::b(D);
    Dphi = gto_values.phi * Db;
    auto rho_b = occ::qm::block::b(rho);
    rho_b.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
    rho_b.col(1) =
        2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
    rho_b.col(2) =
        2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
    rho_b.col(3) =
        2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
}

template <>
void evaluate_density<2, R>(MatConstRef D,
                            const occ::gto::GTOValues &gto_values, MatRef rho) {
    // use a MatRM as a row major temporary, selfadjointView also speeds things
    // up a little.
    // Genuine bottleneck ~ 50% of the time for DFT XC is spent here for hybrids
    MatRM Dphi = gto_values.phi * D.selfadjointView<Eigen::Upper>();
    rho.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
    rho.col(1) = 2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
    rho.col(2) = 2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
    rho.col(3) = 2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
    // laplacian
    rho.col(4) = 2 * ((gto_values.phi_xx.array() + gto_values.phi_yy.array() +
                       gto_values.phi_zz.array()) *
                      Dphi.array())
                         .rowwise()
                         .sum();
    // tau
    Dphi = gto_values.phi_x * D;
    rho.col(5) = (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
    Dphi = gto_values.phi_y * D;
    rho.col(5).array() +=
        (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
    Dphi = gto_values.phi_z * D;
    rho.col(5).array() +=
        (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();

    rho.col(4).array() += 2 * rho.col(5).array();
    rho.col(5).array() *= 0.5;
}

template <>
void evaluate_density<2, U>(MatConstRef D,
                            const occ::gto::GTOValues &gto_values, MatRef rho) {
    occ::timing::start(occ::timing::category::fft);
    // use a MatRM as a row major temporary, selfadjointView also speeds things
    // up a little.
    // Genuine bottleneck ~ 50% of the time for DFT XC is spent here for hybrids
    // alpha part first
    auto Da = occ::qm::block::a(D);
    MatRM Dphi = gto_values.phi * Da;
    auto rho_a = occ::qm::block::a(rho);
    rho_a.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
    /*
     * If we wish to get the values interleaved for say libxc, use an
     * Eigen::Map as follows: Map<occ::Mat, 0, Stride<Dynamic,
     * 2>>(rho.col(1).data(), Dphi.rows(), Dphi.cols(), Stride<Dynamic,
     * 2>(2*Dphi.rows(), 2)) = RHS
     */
    rho_a.col(1) =
        2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
    rho_a.col(2) =
        2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
    rho_a.col(3) =
        2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
    // laplacian
    rho_a.col(4) = 2 * ((gto_values.phi_xx.array() + gto_values.phi_yy.array() +
                         gto_values.phi_zz.array()) *
                        Dphi.array())
                           .rowwise()
                           .sum();
    // tau
    Dphi = gto_values.phi_x * Da;
    rho_a.col(5) = (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
    Dphi = gto_values.phi_y * Da;
    rho_a.col(5).array() +=
        (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
    Dphi = gto_values.phi_z * Da;
    rho_a.col(5).array() +=
        (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
    rho_a.col(4).array() += 2 * rho_a.col(5).array();
    rho_a.col(5).array() *= 0.5;
    // beta part
    auto Db = occ::qm::block::b(D);
    Dphi = gto_values.phi * Db;
    auto rho_b = occ::qm::block::b(rho);
    rho_b.col(0) = (gto_values.phi.array() * Dphi.array()).rowwise().sum();
    rho_b.col(1) =
        2 * (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
    rho_b.col(2) =
        2 * (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
    rho_b.col(3) =
        2 * (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
    // laplacian
    rho_b.col(4) = 2 * ((gto_values.phi_xx.array() + gto_values.phi_yy.array() +
                         gto_values.phi_zz.array()) *
                        Dphi.array())
                           .rowwise()
                           .sum();
    // tau
    Dphi = gto_values.phi_x * Db;
    rho_b.col(5) = (gto_values.phi_x.array() * Dphi.array()).rowwise().sum();
    Dphi = gto_values.phi_y * Db;
    rho_b.col(5).array() +=
        (gto_values.phi_y.array() * Dphi.array()).rowwise().sum();
    Dphi = gto_values.phi_z * Db;
    rho_b.col(5).array() +=
        (gto_values.phi_z.array() * Dphi.array()).rowwise().sum();
    rho_b.col(4).array() += 2 * rho_b.col(5).array();
    rho_b.col(5).array() *= 0.5;
}

} // namespace occ::density
