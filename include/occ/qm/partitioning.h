#pragma once
#include <fmt/ostream.h>
#include <occ/core/linear_algebra.h>
#include <occ/qm/basisset.h>

namespace occ::qm {

template <SpinorbitalKind kind>
Vec mulliken_partition(const BasisSet &basis,
                       const std::vector<occ::core::Atom> &atoms, const Mat &D,
                       const Mat &op) {
    auto nbf = basis.nbf();
    Vec N = Vec::Zero(atoms.size());
    auto bf2atom = basis.bf2atom(atoms);
    Vec pbf = Vec::Zero(nbf);
    if constexpr (kind == Restricted) {
        Vec pop = (D * op).diagonal();
        for (int u = 0; u < nbf; u++) {
            N(bf2atom[u]) += pop(u);
        }
    } else if constexpr (kind == Unrestricted) {
        Vec pop_a = (D.alpha() * op).diagonal();
        Vec pop_b = (D.beta() * op).diagonal();
        for (int u = 0; u < nbf; u++) {
            N(bf2atom[u]) += (pop_a(u) + pop_b(u));
        }
    } else if constexpr (kind == General) {
        Vec pop_aa = (D.alpha_alpha() * op).diagonal();
        Vec pop_ab = (D.alpha_beta() * op).diagonal();
        Vec pop_ba = (D.beta_alpha() * op).diagonal();
        Vec pop_bb = (D.beta_beta() * op).diagonal();
        for (int u = 0; u < nbf; u++) {
            N(bf2atom[u]) += pop_aa(u) + pop_ab(u) + pop_ba(u) + pop_bb(u);
        }
    }
    return N;
}

} // namespace occ::qm
