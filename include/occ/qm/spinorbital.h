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

inline constexpr std::pair<size_t, size_t>
matrix_dimensions(SpinorbitalKind kind, size_t nbf) {
    switch (kind) {
    case Unrestricted:
        return {2 * nbf, nbf};
    case General:
        return {2 * nbf, 2 * nbf};
    default:
        return {nbf, nbf};
    }
}

} // namespace occ::qm
