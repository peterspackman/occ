#include <occ/core/linear_algebra.h>
#include <occ/core/logger.h>

namespace occ {

std::tuple<Mat, Mat, size_t, double, double>
gensqrtinv(const Mat &S, bool symmetric, double max_condition_number) {
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
    return std::make_tuple(X, Xinv, size_t(n_cond), condition_number,
                           result_condition_number);
}

std::tuple<Mat, Mat, double>
conditioning_orthogonalizer(const Mat &S, double S_condition_number_threshold) {
    size_t obs_rank;
    double S_condition_number;
    double XtX_condition_number;
    Mat X, Xinv;

    assert(S.rows() == S.cols());

    std::tie(X, Xinv, obs_rank, S_condition_number, XtX_condition_number) =
        gensqrtinv(S, false, S_condition_number_threshold);
    auto obs_nbf_omitted = (long)S.rows() - (long)obs_rank;
    occ::log::debug("Overlap condition number = {}", S_condition_number);

    if (obs_nbf_omitted > 0) {
        occ::log::debug(" (dropped {} {} to reduce to {})", obs_nbf_omitted,
                        obs_nbf_omitted > 1 ? "fns" : "fn",
                        XtX_condition_number);
    }

    if (obs_nbf_omitted > 0) {
        Mat should_be_I = X.transpose() * S * X;
        Mat I = Mat::Identity(should_be_I.rows(), should_be_I.cols());
        occ::log::debug("||X^t * S * X - I||_2 = {} (should be 0)\n",
                        (should_be_I - I).norm());
    }

    return std::make_tuple(X, Xinv, XtX_condition_number);
}

Mat3 inertia_tensor(Eigen::Ref<const Vec> masses,
                    Eigen::Ref<const Mat3N> positions) {
    Mat3 result;
    double total_mass = masses.array().sum();
    Vec3 center_of_mass =
        (positions.array().rowwise() * masses.transpose().array())
            .rowwise()
            .sum() /
        total_mass;
    Mat3N d = positions.colwise() - center_of_mass;
    Mat3N md = d.array().rowwise() * masses.transpose().array();
    Mat3N d2 = d.array() * d.array();
    Mat3N md2 = d2.array().rowwise() * masses.transpose().array();

    result(0, 0) = (md2.row(1).array() + md2.row(2).array()).array().sum();
    result(1, 1) = (md2.row(0).array() + md2.row(2).array()).array().sum();
    result(2, 2) = (md2.row(0).array() + md2.row(1).array()).array().sum();
    result(0, 1) = -md.row(0).dot(d.row(1));
    result(1, 0) = result(0, 1);
    result(0, 2) = -md.row(0).dot(d.row(2));
    result(2, 0) = result(0, 2);
    result(1, 2) = -md.row(1).dot(d.row(2));
    result(2, 1) = result(1, 2);

    return result;
}

std::pair<Mat, Mat> meshgrid(const Vec &x, const Vec &y) {
    Mat g0 = x.replicate(1, y.rows()).transpose();
    Mat g1 = y.replicate(1, x.rows());
    return {g0, g1};
}

} // namespace occ
