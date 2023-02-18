#include <occ/core/gensqrtinv.h>

namespace occ::core {

GenSqrtInvResult gensqrtinv(Eigen::Ref<const Mat> S, bool symmetric,
                            double max_condition_number) {
    Eigen::SelfAdjointEigenSolver<Mat> eig_solver(S);
    auto U = eig_solver.eigenvectors();
    auto s = eig_solver.eigenvalues();
    auto s_max = s.maxCoeff();
    auto condition_number = std::min(
        s_max / std::max(s.minCoeff(), std::numeric_limits<double>::min()),
        1.0 / std::numeric_limits<double>::epsilon());
    auto threshold = s_max / max_condition_number;
    long n = s.rows();
    long n_cond = 0;
    for (long i = n - 1; i >= 0; --i) {
        if (s(i) >= threshold) {
            ++n_cond;
        } else
            i = 0; // skip rest since eigenvalues are in ascending order
    }

    auto sigma = s.bottomRows(n_cond);
    auto result_condition_number = sigma.maxCoeff() / sigma.minCoeff();
    auto sigma_sqrt = sigma.array().sqrt().matrix().asDiagonal();
    auto sigma_invsqrt = sigma.array().sqrt().inverse().matrix().asDiagonal();

    // make canonical X/Xinv
    auto U_cond = U.block(0, n - n_cond, n, n_cond);
    Mat X = U_cond * sigma_invsqrt;
    Mat Xinv = U_cond * sigma_sqrt;
    // convert to symmetric, if needed
    if (symmetric) {
        X = X * U_cond.transpose();
        Xinv = Xinv * U_cond.transpose();
    }
    return {X, Xinv, static_cast<size_t>(n_cond), condition_number,
            result_condition_number};
}
} // namespace occ::core
