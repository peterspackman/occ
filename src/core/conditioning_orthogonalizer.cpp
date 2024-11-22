#include <occ/core/conditioning_orthogonalizer.h>
#include <occ/core/gensqrtinv.h>
#include <occ/core/log.h>

namespace occ::core {

ConditioningOrthogonalizerResult
conditioning_orthogonalizer(Eigen::Ref<const Mat> S,
                            double condition_number_threshold) {
  assert(S.rows() == S.cols());

  auto g = gensqrtinv(S, false, condition_number_threshold);
  auto obs_nbf_omitted =
      static_cast<long>(S.rows()) - static_cast<long>(g.n_cond);
  log::debug("Overlap condition number = {}", g.condition_number);

  if (obs_nbf_omitted > 0) {
    occ::log::debug(" (dropped {} {} to reduce to {})", obs_nbf_omitted,
                    obs_nbf_omitted > 1 ? "fns" : "fn",
                    g.result_condition_number);
  }

  if (obs_nbf_omitted > 0) {
    Mat should_be_I = g.result.transpose() * S * g.result;
    Mat I = Mat::Identity(should_be_I.rows(), should_be_I.cols());
    occ::log::debug("||X^t * S * X - I||_2 = {} (should be 0)\n",
                    (should_be_I - I).norm());
  }

  return {g.result, g.result_inverse, g.result_condition_number};
}

} // namespace occ::core
