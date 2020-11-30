#pragma once

#include <Eigen/Dense>
#include <tonto/3rdparty/nanoflann.hpp>

namespace cx {

template <typename NumericType>
using KDTree = nanoflann::KDTreeEigenMatrixAdaptor<
    Eigen::Matrix<NumericType, 3, Eigen::Dynamic>, 3, nanoflann::metric_L2,
    false>;

constexpr size_t max_leaf = 10;

} // namespace cx
