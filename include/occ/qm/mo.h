#pragma once
#include <occ/qm/spinorbital.h>

namespace occ::qm {

struct MolecularOrbitals {
    SpinorbitalKind kind{SpinorbitalKind::Restricted};
    size_t n_occ;
    Mat C;
    Mat Cocc;
    Mat D;
    Vec energies;
};

} // namespace occ::qm
