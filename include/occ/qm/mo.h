#pragma once
#include <occ/qm/spinorbital.h>

namespace occ::qm {

class BasisSet;

struct MolecularOrbitals {
    SpinorbitalKind kind{SpinorbitalKind::Restricted};
    size_t n_occ;
    Mat C;
    Mat Cocc;
    Mat D;
    Vec energies;
    void rotate(const BasisSet &basis, const occ::Mat3 &rotation);
};

} // namespace occ::qm
