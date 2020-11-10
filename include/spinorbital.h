#pragma once
#include "linear_algebra.h"

namespace tonto::qm {

enum SpinorbitalKind {
    Restricted,
    Unrestricted,
    General
};

template<SpinorbitalKind kind>
constexpr std::pair<size_t, size_t> matrix_dimensions(size_t nbf) {
    switch (kind) {
    case Restricted: return {nbf, nbf};
    case Unrestricted: return {2 * nbf, nbf};
    case General: return {2 * nbf, 2 * nbf};
    }
}

template<SpinorbitalKind kind, typename TA>
typename TA::Scalar expectation(const TA& left, const TA& right) {
    if constexpr(kind == Unrestricted) {
        if(right.rows() < left.rows()) {
            return left.alpha().cwiseProduct(right).sum() +
                   left.beta().cwiseProduct(right).sum();
        }
        else {
            return left.alpha().cwiseProduct(right.alpha()).sum() +
                   left.beta().cwiseProduct(right.beta()).sum();
        }
    }
    else {
        return left.cwiseProduct(right).sum();
    }
}

}
