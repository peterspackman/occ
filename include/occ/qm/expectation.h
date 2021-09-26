#pragma once
#include <occ/qm/spinorbital.h>

namespace occ::qm {

template <SpinorbitalKind kind, typename TA>
typename TA::Scalar expectation(const TA &left, const TA &right) {
    if constexpr (kind == Unrestricted) {
        if (right.rows() < left.rows()) {
            return block::a(left).cwiseProduct(right).sum() +
                   block::b(left).cwiseProduct(right).sum();
        }
        return block::a(left).cwiseProduct(block::a(right)).sum() +
               block::b(left).cwiseProduct(block::b(right)).sum();
    } else if constexpr (kind == General) {
        if (right.rows() < left.rows()) {
            return block::aa(left).cwiseProduct(right).sum() +
                   block::bb(left).cwiseProduct(right).sum();
        }
        return left.cwiseProduct(right).sum();
    } else {
        return left.cwiseProduct(right).sum();
    }
}

} // namespace occ::qm
