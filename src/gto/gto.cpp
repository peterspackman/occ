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
                                      sh.nprim(), coeffs, alpha, center, GG_CARTESIAN_CCA,
                                      output, x_out, y_out, z_out);   
            }
            occ::timing::stop(occ::timing::category::gto_shell);
        }
    }
    occ::timing::stop(occ::timing::category::gto);
}


}
