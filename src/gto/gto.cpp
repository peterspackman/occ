#include <occ/gto/gto.h>

namespace occ::gto {

void evaluate_basis(const BasisSet &basis,
                    const std::vector<occ::core::Atom> &atoms,
                    const occ::Mat &grid_pts,
                    GTOValues &gto_values,
                    int max_derivative)
{
    occ::timing::start(occ::timing::category::gto);
    size_t nbf = basis.nbf();
    size_t npts = grid_pts.cols();
    size_t natoms = atoms.size();
    gto_values.reserve(nbf, npts, max_derivative);
    gto_values.set_zero();
    auto shell2bf = basis.shell2bf();
    auto atom2shell = basis.atom2shell(atoms);
    for(size_t i = 0; i < natoms; i++)
    {
        for(const auto& shell_idx: atom2shell[i]) {
            occ::timing::start(occ::timing::category::gto_shell);
            size_t bf = shell2bf[shell_idx];
            double * output = gto_values.phi.col(bf).data();
            const double * xyz = grid_pts.data();
            long int xyz_stride = 3;
            const auto& sh = basis[shell_idx];
            const double * coeffs = sh.contr[0].coeff.data();
            const double * alpha = sh.alpha.data();
            const double * center = sh.O.data();
            int L = sh.contr[0].l;
            int order = (sh.contr[0].pure) ? GG_SPHERICAL_CCA : GG_CARTESIAN_CCA;
            if (max_derivative == 0)
            {
                gg_collocation(L,
                               npts, xyz, xyz_stride,
                               sh.nprim(), coeffs, alpha, center, order,
                               output);   
            }
            else if (max_derivative == 1)
            {
                double * x_out = gto_values.phi_x.col(bf).data();
                double * y_out = gto_values.phi_y.col(bf).data();
                double * z_out = gto_values.phi_z.col(bf).data();
                gg_collocation_deriv1(L,
                        npts, xyz, xyz_stride,
                                      sh.nprim(), coeffs, alpha, center, order,
                                      output, x_out, y_out, z_out);   
            }
            else if (max_derivative == 2)
            {
                double * x_out = gto_values.phi_x.col(bf).data();
                double * y_out = gto_values.phi_y.col(bf).data();
                double * z_out = gto_values.phi_z.col(bf).data();
                double *xx_out = gto_values.phi_xx.col(bf).data();
                double *xy_out = gto_values.phi_xy.col(bf).data();
                double *xz_out = gto_values.phi_xz.col(bf).data();
                double *yy_out = gto_values.phi_yy.col(bf).data();
                double *yz_out = gto_values.phi_yz.col(bf).data();
                double *zz_out = gto_values.phi_zz.col(bf).data();
                gg_collocation_deriv2(L,
                        npts, xyz, xyz_stride,
                        sh.nprim(), coeffs, alpha, center, order,
                        output, x_out, y_out, z_out,
                        xx_out, xy_out, xz_out, yy_out, yz_out, zz_out);
            }
            occ::timing::stop(occ::timing::category::gto_shell);
        }
    }
    occ::timing::stop(occ::timing::category::gto);
}


Mat spherical_to_cartesian_transformation_matrix(int l)
{
    Mat result = Mat::Zero(num_subshells(false, l), num_subshells(true, l));
    switch(l)
    {
        case 0:
        {
            result(0, 0) = 1.0;
            break;
        }
        case 1:
        {
            result(0, 2) = 1;
            result(1, 0) = 1;
            result(2, 1) = 1;
            break;
        }
        case 2:
        {
            const double c3 = std::sqrt(3) / 2;
            result(0, 0) = - 1/2;
            result(0, 3) = - 1/2;
            result(0, 5) = 1;
            result(1, 2) = 1;
            result(2, 4) = 1;
            result(3, 0) = c3;
            result(3, 3) = -c3;
            result(4, 1) = 1;
            break;
        }
        case 3:
        {
            const double c2 = 3 * std::sqrt(2) / 4;
            const double c3 = std::sqrt(3) / 2;
            const double c5 = 3 * std::sqrt(5) / 10;
            const double c6 = std::sqrt(6) / 4;
            const double c10 = std::sqrt(10) / 4;
            const double c30 = std::sqrt(30) / 20;
            result(0, 2) = -c5; result(0, 7) = -c5; result(9) = 1;
            result(1, 0) = -c6; result(1, 3) = -c30; result(1, 5) = 4 * c30;
            result(2, 1) = -c30; result(2, 6) = -c6; result(2, 8) = 4 * c30;
            result(3, 2) = c3; result(3, 7) = -c3;
            result(4, 4) = 1;
            result(5, 0) = c10; result(5, 3) = -c2;
            result(6, 1) = c2; result(6, 6) = -c10;
            break;
        }

/*
const static type_sparse_el cptf4[] = {
    { 0,  0, 0.375},
    { 0,  3, 0.21957751641341996535},
    { 0,  5, -0.87831006565367986142},
    { 0, 10, 0.375},
    { 0, 12, -0.87831006565367986142},
    { 0, 14, 1.0},
    { 1,  2, -0.89642145700079522998},
    { 1,  7, -0.40089186286863657703},
    { 1,  9, 1.19522860933439364},
    { 2,  4, -0.40089186286863657703},
    { 2, 11, -0.89642145700079522998},
    { 2, 13, 1.19522860933439364},
    { 3,  0, -0.5590169943749474241},
    { 3,  5, 0.9819805060619657157},
    { 3, 10, 0.5590169943749474241},
    { 3, 12, -0.9819805060619657157},
    { 4,  1, -0.42257712736425828875},
    { 4,  6, -0.42257712736425828875},
    { 4,  8, 1.1338934190276816816},
    { 5,  2, 0.790569415042094833},
    { 5,  7, -1.0606601717798212866},
    { 6,  4, 1.0606601717798212866},
    { 6, 11, -0.790569415042094833},
    { 7,  0, 0.73950997288745200532},
    { 7,  3, -1.2990381056766579701},
    { 7, 10, 0.73950997288745200532},
    { 8,  1, 1.1180339887498948482},
    { 8,  6, -1.1180339887498948482},
};
*/
    }
    return result;
}

Mat cartesian_to_spherical_transformation_matrix(int l)
{
    return spherical_to_cartesian_transformation_matrix(l).transpose();
}

}


