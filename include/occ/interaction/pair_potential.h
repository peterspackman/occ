#pragma once
#include <occ/core/dimer.h>
#include <occ/core/linear_algebra.h>

namespace occ::interaction {

template <int derivative_order = 0>
Mat lennard_jones(Eigen::Ref<const Mat3N> positions,
                  Eigen::Ref<const Mat> params);

template <>
Mat lennard_jones<0>(Eigen::Ref<const Mat3N> positions,
                     Eigen::Ref<const Mat> params);

template <>
Mat lennard_jones<1>(Eigen::Ref<const Mat3N> positions,
                     Eigen::Ref<const Mat> params);

double dreiding_type_hb_correction(double eps, double sigma,
                                   const occ::core::Dimer &dimer);

} // namespace occ::interaction
