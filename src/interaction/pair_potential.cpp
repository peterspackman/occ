#include <cmath>
#include <fmt/core.h>
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

inline double dreiding_hb(double r, double cos_theta, double sigma,
                          double power1, double power2, double eps) {
    return eps *
           (4 * std::pow(sigma / r, power1) - 6 * std::pow(sigma / r, power2)) *
           std::pow(cos_theta, 4);
}

Mat pairwise_distance(const Mat3N &mat1, const Mat3N &mat2) {
    int n1 = mat1.cols();
    int n2 = mat2.cols();

    Mat result(n1, n2);

    for (int i = 0; i < n1; ++i) {
        result.row(i) = (mat2.colwise() - mat1.col(i)).colwise().norm();
    }

    return result;
}

double dreiding_type_hb_correction(double eps, double sigma,
                                   const occ::core::Dimer &dimer) {

    double power1 = 12;
    double power2 = 10;

    const auto &elements_a = dimer.a().atomic_numbers();
    const auto &elements_b = dimer.b().atomic_numbers();
    const auto &pos_a = dimer.a().positions();
    const auto &pos_b = dimer.b().positions();
    const int Na = elements_a.rows();
    const int Nb = elements_b.rows();

    auto is_hb_donor = [](int i) { return i == 7 || i == 8 || i == 9; };

    // this is the slow way for now, can obviously cache this rather than
    // recalculate

    Mat dist = pairwise_distance(dimer.a().positions(), dimer.b().positions());

    double result = 0.0;
    for (int i = 0; i < Na; i++) {
        if (elements_a(i) != 1)
            continue;
        Vec distances_a = (pos_a.colwise() - pos_a.col(i)).colwise().norm();
        distances_a(i) = 1e8;
        Eigen::Index bonded_idx;
        double vab_norm = distances_a.minCoeff(&bonded_idx);

        if (!is_hb_donor(elements_a(bonded_idx)))
            continue;
        Vec3 vab = pos_a.col(i) - pos_a.col(bonded_idx);

        for (int j = 0; j < Nb; j++) {
            if (!is_hb_donor(elements_b(j)))
                continue;
            Vec3 vbc = pos_b.col(j) - pos_a.col(i);
            double vbc_norm = vbc.norm();
            double r = (pos_b.col(j) - pos_a.col(bonded_idx)).norm();
            double cos_theta = vab.dot(vbc) / vab_norm / vbc_norm;
            result += dreiding_hb(r, cos_theta, sigma, power1, power2, eps);
        }
    }

    for (int i = 0; i < Nb; i++) {
        if (elements_b(i) != 1)
            continue;
        Vec distances_b = (pos_b.colwise() - pos_b.col(i)).colwise().norm();
        distances_b(i) = 1e8;
        Eigen::Index bonded_idx;
        double vab_norm = distances_b.minCoeff(&bonded_idx);

        if (!is_hb_donor(elements_b(bonded_idx)))
            continue;
        Vec3 vab = pos_b.col(i) - pos_b.col(bonded_idx);

        for (int j = 0; j < Na; j++) {
            if (!is_hb_donor(elements_a(j)))
                continue;
            Vec3 vbc = pos_a.col(j) - pos_b.col(i);
            double vbc_norm = vbc.norm();
            double r = (pos_a.col(j) - pos_b.col(bonded_idx)).norm();
            double cos_theta = vab.dot(vbc) / vab_norm / vbc_norm;
            result += dreiding_hb(r, cos_theta, sigma, power1, power2, eps);
        }
    }
    return result;
}

} // namespace occ::interaction
