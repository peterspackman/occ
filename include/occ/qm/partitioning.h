#pragma once
#include <fmt/ostream.h>
#include <occ/core/linear_algebra.h>
#include <occ/qm/occshell.h>
#include <occ/qm/spinorbital.h>

namespace occ::qm {

template <SpinorbitalKind kind>
Vec mulliken_partition(const AOBasis &basis, const Mat &D, const Mat &op) {
    auto nbf = basis.nbf();
    Vec N = Vec::Zero(basis.atoms().size());
    auto bf2atom = basis.bf_to_atom();
    if constexpr (kind == SpinorbitalKind::Restricted) {
        Vec pop = (D * op).diagonal();
        for (int u = 0; u < nbf; u++) {
            N(bf2atom[u]) += pop(u);
        }
    } else if constexpr (kind == SpinorbitalKind::Unrestricted) {
        Vec pop_a = (block::a(D) * op).diagonal();
        Vec pop_b = (block::b(D) * op).diagonal();
        for (int u = 0; u < nbf; u++) {
            N(bf2atom[u]) += (pop_a(u) + pop_b(u));
        }
    } else if constexpr (kind == SpinorbitalKind::General) {
        Vec pop_aa = (block::aa(D) * op).diagonal();
        Vec pop_ab = (block::ab(D) * op).diagonal();
        Vec pop_ba = (block::ba(D) * op).diagonal();
        Vec pop_bb = (block::bb(D) * op).diagonal();
        for (int u = 0; u < nbf; u++) {
            N(bf2atom[u]) += pop_aa(u) + pop_ab(u) + pop_ba(u) + pop_bb(u);
        }
    }
    return N;
}

} // namespace occ::qm
