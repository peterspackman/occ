#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/util.h>
#include <occ/qm/opmatrix.h>

namespace occ::qm {

enum SpinorbitalKind { Restricted, Unrestricted, General };

inline bool get_spinorbital_kind_from_string(const std::string &s,
                                             SpinorbitalKind &sk) {
    std::string lc = occ::util::to_lower_copy(s);
    bool valid = false;
    if ((lc == "r") || (lc == "rhf") || (lc == "restricted") || (lc == "spin-restricted")) {
        sk = SpinorbitalKind::Restricted;
        valid = true;
    } else if ((lc == "u") || (lc == "uhf") || (lc == "unrestricted") ||
               (lc == "spin-unrestricted")) {
        sk = SpinorbitalKind::Unrestricted;
        valid = true;
    } else if ((lc == "g") || (lc == "ghf") || (lc == "general")) {
        sk = SpinorbitalKind::General;
        valid = true;
    }
    return valid;
}

constexpr const char *spinorbital_kind_to_string(const SpinorbitalKind &sk) {
    if (sk == SpinorbitalKind::Restricted)
        return "restricted";
    else if (sk == SpinorbitalKind::Unrestricted)
        return "unrestricted";
    else
        return "general";
}

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
