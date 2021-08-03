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


}
