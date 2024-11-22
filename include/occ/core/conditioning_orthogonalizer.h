#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::core {

struct ConditioningOrthogonalizerResult {
  Mat result;
  Mat result_inverse;
  double result_condition_number;
};

ConditioningOrthogonalizerResult
conditioning_orthogonalizer(Eigen::Ref<const Mat>, double);

} // namespace occ::core
