#pragma once
#include <occ/qm/spinorbital.h>
#include <occ/core/linear_algebra.h>

namespace occ::qm::orb {

inline auto occupied_restricted(const occ::MatRM &orbitals, size_t num_occ)
{
    return orbitals.leftCols(num_occ);

}

inline auto occupied_unrestricted(const occ::MatRM &orbitals, size_t num_alpha, size_t num_beta)
{
    size_t nbf = orbitals.rows() / 2;
    occ::MatRM occ = occ::MatRM::Zero(2 * nbf, std::max(num_alpha, num_beta));
    occ.block(0, 0, nbf, num_alpha) = orbitals.alpha().leftCols(num_alpha);
    occ.block(nbf, 0, nbf, num_beta) = orbitals.beta().leftCols(num_beta);
    return occ;
}

inline auto density_matrix_restricted(const occ::MatRM &occupied_orbitals)
{
    return occupied_orbitals * occupied_orbitals.transpose();
}

inline auto density_matrix_unrestricted(const occ::MatRM &occupied_orbitals, size_t num_alpha, size_t num_beta)
{
    size_t rows, cols;
    size_t nbf = occupied_orbitals.rows() / 2;
    std::tie(rows, cols) = occ::qm::matrix_dimensions<SpinorbitalKind::Unrestricted>(nbf);
    occ::MatRM D(rows, cols);
    D.alpha() = occupied_orbitals.block(0, 0, nbf, num_alpha) * occupied_orbitals.block(0, 0, nbf, num_alpha).transpose();
    D.beta() = occupied_orbitals.block(nbf, 0, nbf, num_beta) * occupied_orbitals.block(nbf, 0, nbf, num_beta).transpose();
    D *= 0.5;
    return D;
}

}
