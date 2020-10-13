#pragma once
#include "linear_algebra.h"

namespace libint2 {
    class Shell;
    class BasisSet;
    class Atom;
}

namespace tonto::density {

inline int num_components(int deriv_order) {
    switch(deriv_order) {
    case 0: return 1;
    case 1: return 4;
    case 2: return 10;
    }
    return 1;
}

    void eval_shell(const libint2::Shell &s1, const Eigen::Ref<const tonto::Mat>& dists, Eigen::Ref<tonto::Mat>& result, int derivative=0);
    tonto::Mat evaluate_gtos(
            const libint2::BasisSet &basis, const std::vector<libint2::Atom> &atoms,
            const tonto::MatN4 &grid_pts, int derivative=0);
    tonto::Mat evaluate(
            const libint2::BasisSet &basis, const std::vector<libint2::Atom> &atoms,
            const tonto::MatRM& D, const tonto::MatN4 &grid_pts, int derivative=0);
}
