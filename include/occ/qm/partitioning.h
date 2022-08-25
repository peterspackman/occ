#pragma once
#include <fmt/ostream.h>
#include <occ/core/linear_algebra.h>
#include <occ/qm/shell.h>
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
        Vec pop = ((block::b(D) + block::a(D)) * op).diagonal();
        for (int u = 0; u < nbf; u++) {
            N(bf2atom[u]) += pop(u);
        }
    } else if constexpr (kind == SpinorbitalKind::General) {
        Vec pop =
            ((block::aa(D) + block::ab(D) + block::ba(D) + block::bb(D)) * op)
                .diagonal();
        for (int u = 0; u < nbf; u++) {
            N(bf2atom[u]) += pop(u);
        }
    }
    return N;
}

} // namespace occ::qm
