#include "basisset.h"
#include "util.h"
#include "gto.h"

namespace tonto::qm {


tonto::MatRM rotate_molecular_orbitals(const BasisSet& basis, const tonto::Mat3& rotation, const tonto::MatRM& C)
{
    const auto shell2bf = basis.shell2bf();
    tonto::MatRM result(C.rows(), C.cols());
    for(size_t s = 0; s < basis.size(); s++) {
        const auto& shell = basis[s];
        size_t bf_first = shell2bf[s];
        size_t shell_size = shell.size();
        int l = shell.contr[0].l;
        tonto::MatRM rot;
        if(l < 1) {
            result.row(bf_first) = C.row(bf_first);
            continue;
        }
        else if (l == 1) {
            rot = rotation;
        }
        else if (l == 2) {
            rot = tonto::gto::cartesian_gaussian_rotation_matrix<2>(rotation);
        }
        else if (l == 3) {
            rot = tonto::gto::cartesian_gaussian_rotation_matrix<3>(rotation);
        }
        else if (l == 4) {
            rot = tonto::gto::cartesian_gaussian_rotation_matrix<4>(rotation);
        }
        else if (l == 5) {
            rot = tonto::gto::cartesian_gaussian_rotation_matrix<5>(rotation);
        }
        else if (l == 6) {
            rot = tonto::gto::cartesian_gaussian_rotation_matrix<6>(rotation);
        }
        else {
            throw std::runtime_error("MO rotation not implemented for angular momentum > 6");
        }
        result.block(bf_first, 0, shell_size, C.cols()) = rot * C.block(bf_first, 0, shell_size, C.cols());
    }
    return result;
}

}
