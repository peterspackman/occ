#pragma once
#include <occ/3rdparty/nanoflann.hpp>
#include <occ/core/linear_algebra.h>

namespace occ::core {

template <typename NumericType>
using KDTree = nanoflann::KDTreeEigenMatrixAdaptor<
    Eigen::Matrix<NumericType, 3, Eigen::Dynamic>, 3, nanoflann::metric_L2,
    false>;

constexpr size_t max_leaf = 10;

} // namespace occ::core
