#include <occ/core/logger.h>
#include <occ/gto/gto.h>
#include <occ/qm/basisset.h>
#include <occ/qm/mo.h>

namespace occ::qm {

void MolecularOrbitals::rotate(const BasisSet &basis, const Mat3 &rotation) {

    const auto shell2bf = basis.shell2bf();
    occ::log::debug("Rotating MO coefficients");
    for (size_t s = 0; s < basis.size(); s++) {
        const auto &shell = basis[s];
        size_t bf_first = shell2bf[s];
        size_t shell_size = shell.size();
        int l = shell.contr[0].l;
        bool pure = shell.contr[0].pure;
        Mat rot;
        switch (l) {
        case 0:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<0>(rotation);
            break;
        case 1:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<1>(rotation);
            break;
        case 2:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<2>(rotation);
            break;
        case 3:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<3>(rotation);
            break;
        case 4:
            rot = occ::gto::cartesian_gaussian_rotation_matrix<4>(rotation);
            break;
        default:
            throw std::runtime_error(
                "MO rotation not implemented for angular momentum > 4");
        }
        if (pure) {
            Mat c = occ::gto::cartesian_to_spherical_transformation_matrix(l);
            Mat cinv =
                occ::gto::spherical_to_cartesian_transformation_matrix(l);
            rot = c * rot * cinv;
        }
        if (kind == SpinorbitalKind::Restricted) {
            occ::log::debug("Restricted MO rotation");
            C.block(bf_first, 0, shell_size, C.cols()) =
                rot * C.block(bf_first, 0, shell_size, C.cols());
        } else {
            occ::log::debug("Unrestricted MO rotation");
            auto ca = block::a(C);
            auto cb = block::b(C);
            ca.block(bf_first, 0, shell_size, ca.cols()) =
                rot * ca.block(bf_first, 0, shell_size, ca.cols());
            cb.block(bf_first, 0, shell_size, cb.cols()) =
                rot * cb.block(bf_first, 0, shell_size, cb.cols());
        }
    }
}

} // namespace occ::qm
