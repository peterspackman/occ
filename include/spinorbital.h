#pragma once

namespace tonto::qm {

enum SpinorbitalKind {
    Restricted,
    Unrestricted,
    General
};

template<SpinorbitalKind kind>
constexpr std::pair<size_t, size_t> density_matrix_dimensions(size_t nbf) {
    switch (kind) {
    case Restricted: return {nbf, nbf};
    case Unrestricted: return {2 * nbf, nbf};
    case General: return {2 * nbf, 2 * nbf};
    }
}

inline auto alpha_block(size_t nbf, tonto::MatRM& mat) {
    return mat.block(0, 0, nbf, nbf);
}

inline auto alpha_block(size_t nbf, const tonto::MatRM& mat) {
    return mat.block(0, 0, nbf, nbf);
}

inline auto beta_block(size_t nbf, tonto::MatRM& mat) {
    return mat.block(nbf, 0, nbf, nbf);
}

inline auto beta_block(size_t nbf, const tonto::MatRM& mat) {
    return mat.block(nbf, 0, nbf, nbf);
}

inline auto alpha_alpha_block(size_t nbf, tonto::MatRM& mat) {
    return mat.block(0, 0, nbf, nbf);
}

inline auto alpha_alpha_block(size_t nbf, const tonto::MatRM& mat) {
    return mat.block(0, 0, nbf, nbf);
}

inline auto alpha_beta_block(size_t nbf, tonto::MatRM& mat) {
    return mat.block(0, nbf, nbf, nbf);
}

inline auto alpha_beta_block(size_t nbf, const tonto::MatRM& mat) {
    return mat.block(0, nbf, nbf, nbf);
}

inline auto beta_alpha_block(size_t nbf, tonto::MatRM& mat) {
    return mat.block(nbf, 0, nbf, nbf);
}

inline auto beta_alpha_block(size_t nbf, const tonto::MatRM& mat) {
    return mat.block(nbf, 0, nbf, nbf);
}

inline auto beta_beta_block(size_t nbf, tonto::MatRM& mat) {
    return mat.block(nbf, nbf, nbf, nbf);
}

inline auto beta_beta_block(size_t nbf, const tonto::MatRM& mat) {
    return mat.block(nbf, nbf, nbf, nbf);
}
}
