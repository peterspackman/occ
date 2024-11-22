#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::core {

struct GenSqrtInvResult {
  Mat result;
  Mat result_inverse;
  size_t n_cond;
  double condition_number;
  double result_condition_number;
};

// returns {X,X^{-1},rank,A_condition_number,result_A_condition_number}, where
// X is the generalized square-root-inverse such that X.transpose() * A * X = I
//
// if symmetric is true, produce "symmetric" sqrtinv: X = U . A_evals_sqrtinv .
// U.transpose()),
// else produce "canonical" sqrtinv: X = U . A_evals_sqrtinv
// where U are eigenvectors of A
// rows and cols of symmetric X are equivalent; for canonical X the rows are
// original basis (AO),
// cols are transformed basis ("orthogonal" AO)
//
// A is conditioned to max_condition_number
GenSqrtInvResult gensqrtinv(Eigen::Ref<const Mat>, bool symmetric = false,
                            double max_condition_number = 1e8);

} // namespace occ::core
