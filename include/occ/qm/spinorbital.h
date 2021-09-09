#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/qm/opmatrix.h>

namespace occ::qm {

enum SpinorbitalKind { Restricted, Unrestricted, General };

template <SpinorbitalKind kind>
constexpr std::pair<size_t, size_t> matrix_dimensions(size_t nbf) {
    switch (kind) {
    case Restricted:
        return {nbf, nbf};
    case Unrestricted:
        return {2 * nbf, nbf};
    case General:
        return {2 * nbf, 2 * nbf};
    }
}

template <SpinorbitalKind kind, typename TA>
typename TA::Scalar expectation(const TA &left, const TA &right) {
    namespace block = occ::qm::block;
    if constexpr (kind == Unrestricted) {
        if (right.rows() < left.rows()) {
            return block::a(left).cwiseProduct(right).sum() +
                   block::b(left).cwiseProduct(right).sum();
        } else {
            return block::a(left).cwiseProduct(block::a(right)).sum() +
                   block::b(left).cwiseProduct(block::b(right)).sum();
        }
    } else if constexpr (kind == General) {
        if (right.rows() < left.rows()) {
            return block::aa(left).cwiseProduct(right).sum() +
                   block::bb(left).cwiseProduct(right).sum();
        } else {
            return left.cwiseProduct(right).sum();
        }
    } else {
        return left.cwiseProduct(right).sum();
    }
}

} // namespace occ::qm
