#include <occ/core/timings.h>
#include <occ/dft/xc_potential_matrix.h>

namespace occ::dft {
using occ::qm::SpinorbitalKind::Restricted;
using occ::qm::SpinorbitalKind::Unrestricted;
namespace block = occ::qm::block;

template <>
void xc_potential_matrix<Restricted, 0>(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const occ::gto::GTOValues &gto_vals,
                                        MatRef Vxc, double &energy) {
    energy += rho.col(0).dot(res.exc);
    Vxc = gto_vals.phi.transpose() *
          (gto_vals.phi.array().colwise() * res.vrho.col(0).array()).matrix();
}

template <>
void xc_potential_matrix<Restricted, 1>(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const occ::gto::GTOValues &gto_vals,
                                        MatRef Vxc, double &energy) {
    const auto &phi = gto_vals.phi;
    const auto &phi_x = gto_vals.phi_x.array();
    const auto &phi_y = gto_vals.phi_y.array();
    const auto &phi_z = gto_vals.phi_z.array();

    energy += rho.col(0).dot(res.exc);
    const auto &vsigma = res.vsigma.col(0).array();
    Mat ktmp = phi.transpose() *
               (0.5 * (phi.array().colwise() * res.vrho.col(0).array()) +
                2 * (phi_x.colwise() * (rho.col(1).array() * vsigma)) +
                2 * (phi_y.colwise() * (rho.col(2).array() * vsigma)) +
                2 * (phi_z.colwise() * (rho.col(3).array() * vsigma)))
                   .matrix();
    Vxc.noalias() += ktmp + ktmp.transpose();
}

template <>
void xc_potential_matrix<Restricted, 2>(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const occ::gto::GTOValues &gto_vals,
                                        MatRef Vxc, double &energy) {
    xc_potential_matrix<Restricted, 1>(res, rho, gto_vals, Vxc, energy);
    // unsure about factors for vtau, vlaplacian
    // xx + yy + zz = rho(4)
    Array t2 = 0.5 * res.vtau.col(0);
    Vxc.noalias() += gto_vals.phi_x.transpose() *
                     (gto_vals.phi_x.array().colwise() * t2).matrix();
    Vxc.noalias() += gto_vals.phi_y.transpose() *
                     (gto_vals.phi_y.array().colwise() * t2).matrix();
    Vxc.noalias() += gto_vals.phi_z.transpose() *
                     (gto_vals.phi_z.array().colwise() * t2).matrix();
}

template <>
void xc_potential_matrix<Unrestricted, 0>(const DensityFunctional::Result &res,
                                          MatConstRef rho,
                                          const occ::gto::GTOValues &gto_vals,
                                          MatRef Vxc, double &energy) {
    double e_alpha = res.exc.dot(block::a(rho).col(0));
    double e_beta = res.exc.dot(block::b(rho).col(0));
    energy += e_alpha + e_beta;
    Mat phi_vrho_a = gto_vals.phi.array().colwise() * res.vrho.col(0).array();
    Mat phi_vrho_b = gto_vals.phi.array().colwise() * res.vrho.col(1).array();
    block::a(Vxc).noalias() = gto_vals.phi.transpose() * phi_vrho_a;
    block::b(Vxc).noalias() = gto_vals.phi.transpose() * phi_vrho_b;
}

template <>
void xc_potential_matrix<Unrestricted, 1>(const DensityFunctional::Result &res,
                                          MatConstRef rho,
                                          const occ::gto::GTOValues &gto_vals,
                                          MatRef Vxc, double &energy) {
    Eigen::Index npt = res.npts;
    // LDA into K0
    xc_potential_matrix<Unrestricted, 0>(res, rho, gto_vals, Vxc, energy);

    const auto &phi = gto_vals.phi;
    const auto &phi_x = gto_vals.phi_x;
    const auto &phi_y = gto_vals.phi_y;
    const auto &phi_z = gto_vals.phi_z;

    // factor of 2 for vsigma up up, 1 for up down
    auto ga = block::a(rho).block(0, 1, npt, 3).array().colwise() *
                  (2 * res.vsigma.col(0).array()) +
              block::b(rho).block(0, 1, npt, 3).array().colwise() *
                  res.vsigma.col(1).array();
    auto gb = block::b(rho).block(0, 1, npt, 3).array().colwise() *
                  (2 * res.vsigma.col(2).array()) +
              block::a(rho).block(0, 1, npt, 3).array().colwise() *
                  res.vsigma.col(1).array();

    Mat gamma_a = phi_x.array().colwise() * ga.col(0).array() +
                  phi_y.array().colwise() * ga.col(1).array() +
                  phi_z.array().colwise() * ga.col(2).array();
    Mat gamma_b = phi_x.array().colwise() * gb.col(0).array() +
                  phi_y.array().colwise() * gb.col(1).array() +
                  phi_z.array().colwise() * gb.col(2).array();
    Mat ktmp = (phi.transpose() * gamma_a);
    block::a(Vxc).noalias() += (ktmp + ktmp.transpose());
    ktmp = (phi.transpose() * gamma_b);
    block::b(Vxc).noalias() += (ktmp + ktmp.transpose());
}

template <>
void xc_potential_matrix<Unrestricted, 2>(const DensityFunctional::Result &res,
                                          MatConstRef rho,
                                          const occ::gto::GTOValues &gto_vals,
                                          MatRef Vxc, double &energy) {
    xc_potential_matrix<Unrestricted, 1>(res, rho, gto_vals, Vxc, energy);

    const auto &phi_x = gto_vals.phi_x;
    const auto &phi_y = gto_vals.phi_y;
    const auto &phi_z = gto_vals.phi_z;

    Array t2 = 0.5 * res.vtau.col(0); // alpha
    block::a(Vxc).noalias() += phi_x.transpose() *
                               (phi_x.array().colwise() * t2).matrix();
    block::a(Vxc).noalias() += phi_y.transpose() *
                               (phi_y.array().colwise() * t2).matrix();
    block::a(Vxc).noalias() += phi_z.transpose() *
                               (phi_z.array().colwise() * t2).matrix();

    t2 = 0.5 * res.vtau.col(1); // beta
    block::b(Vxc).noalias() += phi_x.transpose() *
                               (phi_x.array().colwise() * t2).matrix();
    block::b(Vxc).noalias() += phi_y.transpose() *
                               (phi_y.array().colwise() * t2).matrix();
    block::b(Vxc).noalias() += phi_z.transpose() *
                               (phi_z.array().colwise() * t2).matrix();
}

} // namespace occ::dft
