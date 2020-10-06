#pragma once
#include "linear_algebra.h"

namespace libint2 {
    class Shell;
    class BasisSet;
    class Atom;
}

namespace tonto::density {
    void eval_shell(const libint2::Shell &s1, const Eigen::Ref<const tonto::Mat>& dists, Eigen::Ref<tonto::Mat>& result);
    tonto::Mat evaluate_gtos(
            const libint2::BasisSet &basis, const std::vector<libint2::Atom> &atoms,
            const tonto::MatN4 &grid_pts);
    tonto::Vec evaluate(
            const libint2::BasisSet &basis, const std::vector<libint2::Atom> &atoms,
            const tonto::MatRM& D, const tonto::MatN4 &grid_pts);
}
