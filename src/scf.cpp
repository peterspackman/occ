#include "scf.h"

namespace craso::scf {

    std::tuple<RowMajorMatrix, RowMajorMatrix, size_t, double, double> gensqrtinv(const RowMajorMatrix &S, bool symmetric, double max_condition_number)
    {
        Eigen::SelfAdjointEigenSolver<RowMajorMatrix> eig_solver(S);
        auto U = eig_solver.eigenvectors();
        auto s = eig_solver.eigenvalues();
        auto s_max = s.maxCoeff();
        auto condition_number = std::min(
            s_max / std::max(s.minCoeff(), std::numeric_limits<double>::min()),
            1.0 / std::numeric_limits<double>::epsilon());
        auto threshold = s_max / max_condition_number;
        long n = s.rows();
        long n_cond = 0;
        for (long i = n - 1; i >= 0; --i)
        {
            if (s(i) >= threshold)
            {
                ++n_cond;
            }
            else
                i = 0; // skip rest since eigenvalues are in ascending order
        }

        auto sigma = s.bottomRows(n_cond);
        auto result_condition_number = sigma.maxCoeff() / sigma.minCoeff();
        auto sigma_sqrt = sigma.array().sqrt().matrix().asDiagonal();
        auto sigma_invsqrt = sigma.array().sqrt().inverse().matrix().asDiagonal();

        // make canonical X/Xinv
        auto U_cond = U.block(0, n - n_cond, n, n_cond);
        RowMajorMatrix X = U_cond * sigma_invsqrt;
        RowMajorMatrix Xinv = U_cond * sigma_sqrt;
        // convert to symmetric, if needed
        if (symmetric)
        {
            X = X * U_cond.transpose();
            Xinv = Xinv * U_cond.transpose();
        }
        return std::make_tuple(X, Xinv, size_t(n_cond), condition_number,
                               result_condition_number);
    }

    std::tuple<RowMajorMatrix, RowMajorMatrix, double> conditioning_orthogonalizer(const RowMajorMatrix &S, double S_condition_number_threshold)
    {
        size_t obs_rank;
        double S_condition_number;
        double XtX_condition_number;
        RowMajorMatrix X, Xinv;

        assert(S.rows() == S.cols());

        std::tie(X, Xinv, obs_rank, S_condition_number, XtX_condition_number) =
            gensqrtinv(S, false, S_condition_number_threshold);
        auto obs_nbf_omitted = (long)S.rows() - (long)obs_rank;
        fmt::print("Overlap condition number = {}", S_condition_number);
        if (obs_nbf_omitted > 0) {
            fmt::print(
                " (dropped {} {} to reduce to {})",
                obs_nbf_omitted,
                obs_nbf_omitted > 1 ? "fns" : "fn",
                XtX_condition_number
            );
        }
        std::cout << std::endl;

        if (obs_nbf_omitted > 0)
        {
            RowMajorMatrix should_be_I = X.transpose() * S * X;
            RowMajorMatrix I = RowMajorMatrix::Identity(should_be_I.rows(), should_be_I.cols());
            fmt::print("||X^t * S * X - I||_2 = {} (should be 0)\n", (should_be_I - I).norm());
        }

        return std::make_tuple(X, Xinv, XtX_condition_number);
    }

}
