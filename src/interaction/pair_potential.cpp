#include <occ/interaction/pair_potential.h>

namespace occ::interaction {

template <>
Mat lennard_jones<0>(Eigen::Ref<const Mat3N> positions,
                     Eigen::Ref<const Mat> params) {
    const int num_particles = positions.cols();
    Mat result = Mat::Zero(num_particles, 1);

    Vec rwork(num_particles);
    Vec uwork(num_particles);

    for (int i = 0; i < positions.cols(); i++) {
        Vec3 p = positions.col(i);
        const int N = num_particles - i - 1;
        // r2
        rwork.bottomRows(N).array() =
            (positions.bottomRows(N).colwise() - p).squaredNorm();

        // r6
        rwork.bottomRows(N) *= rwork.bottomRows(N) * rwork.bottomRows(N);

        uwork.bottomRows(N).array() -=
            params.block(i + 1, 0, N, 1).array() / rwork.bottomRows(N).array();

        rwork.bottomRows(N) *= rwork.bottomRows(N);

        uwork.bottomRows(N).array() +=
            params.block(i + 1, 1, N, 1).array() / rwork.bottomRows(N).array();

        result.bottomRows(N).array() += uwork.bottomRows(N).array();
        result(i) += uwork.bottomRows(N).array().sum();
    }
    return result;
}

} // namespace occ::interaction
