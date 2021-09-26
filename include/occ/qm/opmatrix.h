#pragma once
#include <occ/qm/spinorbital.h>

namespace occ::qm::block {

template <typename Derived> auto a(Eigen::DenseBase<Derived> &mat) {
    return mat.block(0, 0, mat.rows() / 2, mat.cols());
}

template <typename Derived> auto b(Eigen::DenseBase<Derived> &mat) {
    return mat.block(mat.rows() / 2, 0, mat.rows() / 2, mat.cols());
}

template <typename Derived> auto aa(Eigen::DenseBase<Derived> &mat) {
    return mat.block(0, 0, mat.rows() / 2, mat.cols() / 2);
}

template <typename Derived> auto ab(Eigen::DenseBase<Derived> &mat) {
    return mat.block(mat.rows() / 2, 0, mat.rows() / 2, mat.cols() / 2);
}
template <typename Derived> auto ba(Eigen::DenseBase<Derived> &mat) {
    return mat.block(0, mat.cols() / 2, mat.rows() / 2, mat.cols() / 2);
}

template <typename Derived> auto bb(Eigen::DenseBase<Derived> &mat) {
    return mat.block(mat.rows() / 2, mat.cols() / 2, mat.rows() / 2,
                     mat.cols() / 2);
}

template <typename Derived> const auto a(const Eigen::DenseBase<Derived> &mat) {
    return mat.block(0, 0, mat.rows() / 2, mat.cols());
}

template <typename Derived> const auto b(const Eigen::DenseBase<Derived> &mat) {
    return mat.block(mat.rows() / 2, 0, mat.rows() / 2, mat.cols());
}

template <typename Derived>
const auto aa(const Eigen::DenseBase<Derived> &mat) {
    return mat.block(0, 0, mat.rows() / 2, mat.cols() / 2);
}

template <typename Derived>
const auto ab(const Eigen::DenseBase<Derived> &mat) {
    return mat.block(mat.rows() / 2, 0, mat.rows() / 2, mat.cols() / 2);
}
template <typename Derived>
const auto ba(const Eigen::DenseBase<Derived> &mat) {
    return mat.block(0, mat.cols() / 2, mat.rows() / 2, mat.cols() / 2);
}

template <typename Derived>
const auto bb(const Eigen::DenseBase<Derived> &mat) {
    return mat.block(mat.rows() / 2, mat.cols() / 2, mat.rows() / 2,
                     mat.cols() / 2);
}

} // namespace occ::qm::block
