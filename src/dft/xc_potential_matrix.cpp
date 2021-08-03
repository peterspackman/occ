#include <occ/dft/xc_potential_matrix.h>

namespace occ::dft {
using occ::qm::SpinorbitalKind::Restricted;
using occ::qm::SpinorbitalKind::Unrestricted;

template<>
void xc_potential_matrix<Restricted, 0>(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals,
        Mat& Vxc, 
        double &energy)
{
    energy += rho.col(0).dot(res.exc);
    Mat phi_vrho = gto_vals.phi.array().colwise() * res.vrho.col(0).array();
    Vxc = gto_vals.phi.transpose() * phi_vrho;
}


template<>
void xc_potential_matrix<Restricted, 1>(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals,
        Mat& Vxc, 
        double &energy)
{
    Eigen::Index npt = res.npts;
    // LDA into K0
    xc_potential_matrix<Restricted, 0>(res, rho, gto_vals, Vxc, energy);

    const auto& phi = gto_vals.phi;
    const auto& phi_x = gto_vals.phi_x;
    const auto& phi_y = gto_vals.phi_y;
    const auto& phi_z = gto_vals.phi_z;

    const auto& vsigma = res.vsigma.col(0);
    auto g = rho.block(0, 1, npt, 3).array().colwise() * (2 * vsigma.array());
    Mat gamma = 
        phi_x.array().colwise() * g.col(0).array() +
        phi_y.array().colwise() * g.col(1).array() +
        phi_z.array().colwise() * g.col(2).array();
    Mat ktmp = phi.transpose() * gamma;
    Vxc.noalias() += ktmp + ktmp.transpose();
}

template<>
void xc_potential_matrix<Restricted, 2>(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals,
        Mat& Vxc, 
        double &energy)
{
    Eigen::Index npt = res.npts;
    xc_potential_matrix<Restricted, 1>(res, rho, gto_vals, Vxc, energy);

    const auto& phi = gto_vals.phi;
    const auto& phi_x = gto_vals.phi_x;
    const auto& phi_y = gto_vals.phi_y;
    const auto& phi_z = gto_vals.phi_z;

    // unsure about factors for vtau, vlaplacian
    // xx + yy + zz = rho(4) 
    auto t = gto_vals.phi_xx + gto_vals.phi_yy + gto_vals.phi_zz;
    Mat tmp = gto_vals.phi.transpose() * (t.array().colwise() * res.vlaplacian.col(0).array()).matrix();
    Vxc.noalias() += 0.5 * (tmp + tmp.transpose());
    Eigen::ArrayXd t2 = 0.25 * res.vtau.col(0) + res.vlaplacian.col(0);
    tmp = gto_vals.phi_x.transpose() * (gto_vals.phi_x.array().colwise() * t2.array()).matrix();
    tmp += gto_vals.phi_y.transpose() * (gto_vals.phi_y.array().colwise() * t2.array()).matrix();
    tmp += gto_vals.phi_z.transpose() * (gto_vals.phi_z.array().colwise() * t2.array()).matrix();
    Vxc.noalias() += (tmp + tmp.transpose());
}

template<>
void xc_potential_matrix<Unrestricted, 0>(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals, Mat& Vxc, double &energy)
{
    double e_alpha = res.exc.dot(rho.alpha().col(0));
    double e_beta = res.exc.dot(rho.beta().col(0));
    energy +=  e_alpha + e_beta;
    Mat phi_vrho_a = gto_vals.phi.array().colwise() * res.vrho.col(0).array();
    Mat phi_vrho_b = gto_vals.phi.array().colwise() * res.vrho.col(1).array();
    Vxc.alpha().noalias() = gto_vals.phi.transpose() * phi_vrho_a;
    Vxc.beta().noalias() = gto_vals.phi.transpose() * phi_vrho_b;
}


template<>
void xc_potential_matrix<Unrestricted, 1>(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals, Mat& Vxc, double &energy)
{
    Eigen::Index npt = res.npts;
    // LDA into K0
    xc_potential_matrix<Unrestricted, 0>(res, rho, gto_vals, Vxc, energy);

    const auto& phi = gto_vals.phi;
    const auto& phi_x = gto_vals.phi_x;
    const auto& phi_y = gto_vals.phi_y;
    const auto& phi_z = gto_vals.phi_z;

    // factor of 2 for vsigma up up, 1 for up down
    auto ga = rho.alpha().block(0, 1, npt, 3).array().colwise() * (2 * res.vsigma.col(0).array()) +
                    rho.beta().block(0, 1, npt, 3).array().colwise() * res.vsigma.col(1).array();
    auto gb = rho.beta().block(0, 1, npt, 3).array().colwise() * (2 * res.vsigma.col(2).array()) +
                    rho.alpha().block(0, 1, npt, 3).array().colwise() * res.vsigma.col(1).array();

    Mat gamma_a = gto_vals.phi_x.array().colwise() * ga.col(0).array()
                       + gto_vals.phi_y.array().colwise() * ga.col(1).array()
                       + gto_vals.phi_z.array().colwise() * ga.col(2).array();
    Mat gamma_b = gto_vals.phi_x.array().colwise() * gb.col(0).array()
                       + gto_vals.phi_y.array().colwise() * gb.col(1).array()
                       + gto_vals.phi_z.array().colwise() * gb.col(2).array();
    Mat ktmp = (gto_vals.phi.transpose() * gamma_a);
    Vxc.alpha().noalias() += (ktmp + ktmp.transpose());
    ktmp = (gto_vals.phi.transpose() * gamma_b);
    Vxc.beta().noalias() += (ktmp + ktmp.transpose());
}

template<>
void xc_potential_matrix<Unrestricted, 2>(const DensityFunctional::Result &res, const Mat &rho,
        const occ::gto::GTOValues &gto_vals, Mat& Vxc, double &energy)
{
    Eigen::Index npt = res.npts;
    xc_potential_matrix<Unrestricted, 1>(res, rho, gto_vals, Vxc, energy);

    const auto& phi = gto_vals.phi;
    const auto& phi_x = gto_vals.phi_x;
    const auto& phi_y = gto_vals.phi_y;
    const auto& phi_z = gto_vals.phi_z;

    // xx + yy + zz = rho(4) 
    auto t = gto_vals.phi_xx + gto_vals.phi_yy + gto_vals.phi_zz;
    Mat tmp = gto_vals.phi.transpose() * (t.array().colwise() * res.vlaplacian.col(0).array()).matrix();
    Vxc.alpha().noalias() += 0.5 * (tmp + tmp.transpose());
    tmp = gto_vals.phi.transpose() * (t.array().colwise() * res.vlaplacian.col(1).array()).matrix();
    Vxc.beta().noalias() += 0.5 * (tmp + tmp.transpose());

    auto t2a = 0.25 * res.vtau.col(0) + res.vlaplacian.col(0);
    tmp = gto_vals.phi_x.transpose() * (gto_vals.phi_x.array().colwise() * t2a.array()).matrix();
    tmp.noalias() += gto_vals.phi_y.transpose() * (gto_vals.phi_y.array().colwise() * t2a.array()).matrix();
    tmp.noalias() += gto_vals.phi_z.transpose() * (gto_vals.phi_z.array().colwise() * t2a.array()).matrix();
    Vxc.alpha().noalias() += (tmp + tmp.transpose());

    auto t2b = 0.25 * res.vtau.col(1) + res.vlaplacian.col(1);
    tmp = gto_vals.phi_x.transpose() * (gto_vals.phi_x.array().colwise() * t2b.array()).matrix();
    tmp.noalias() += gto_vals.phi_y.transpose() * (gto_vals.phi_y.array().colwise() * t2b.array()).matrix();
    tmp.noalias() += gto_vals.phi_z.transpose() * (gto_vals.phi_z.array().colwise() * t2b.array()).matrix();
    Vxc.beta().noalias() += (tmp + tmp.transpose());
}

}