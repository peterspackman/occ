#pragma once
#include <occ/qm/spinorbital.h>

namespace occ::qm {

class BasisSet;

struct MolecularOrbitals {
    SpinorbitalKind kind{SpinorbitalKind::Restricted};
    size_t n_alpha{0}, n_beta{0}, n_ao{0};
    Mat C;
    Mat Cocc;
    Mat D;
    Vec energies;
    void update(const Mat &ortho, const Mat &potential);
    void update_density_matrix();
    void rotate(const BasisSet &basis, const occ::Mat3 &rotation);
};

} // namespace occ::qm
