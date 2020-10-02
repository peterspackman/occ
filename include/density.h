#pragma once
#include "linear_algebra.h"

namespace libint2 {
    class Shell;
    class BasisSet;
    class Atom;
}

namespace tonto::density {
    tonto::Vec eval_gto(const libint2::Shell &s1, const double x[3]);
    tonto::Vec evaluate(
            const libint2::BasisSet &basis, const std::vector<libint2::Atom> &atoms,
            const tonto::MatRM& D, const tonto::Mat4N &grid_pts);
}
