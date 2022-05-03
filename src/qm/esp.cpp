#include <cmath>
#include <fmt/core.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/timings.h>
#include <occ/qm/esp.h>

namespace occ {

Vec boys(int N, double x) {
    static const double factorial[]{1,   1,    2,     6,      24,     120,
                                    720, 5040, 40320, 362880, 3628800};

    Vec result = Vec::Zero(N + 1);
    const double sqrt_x = std::sqrt(x);
    const double xerf = std::erf(sqrt_x);
    const double sqrt_pi = std::sqrt(M_PI);

    for (int n = 0; n <= N; n++) {
        const double front = 0.5 * factorial[2 * n] / factorial[n];
        const double num1 = sqrt_pi * xerf;
        const double den1 = std::pow(2, 2 * n) * std::pow(x, n + 0.5);
        result(n) = num1 / den1;

        const double fac2 = std::exp(-x);
        for (int k = 0; k < n; k++) {
            const double num2 = factorial[n - k];
            const double den2 = factorial[2 * n - 2 * k] * std::pow(x, k + 1) *
                                std::pow(2, 2 * k);
            result(n) -= fac2 * num2 / den2;
        }
        result(n) *= front;
    }
    return result;
}

Vec boys_accum(int N, double x) {
    Vec result = Vec::Zero(N + 1);
    Vec ak = Vec::Zero(N + 1);
    Vec denominator = Vec(N + 1);
    // g_n(x) = exp(x) F_n(x) = \sum_k=0^\inf a_k
    // a_0 = 1 / (2n + 1)
    // a_{k + 1} = a_k * 2 X / (2n + 2k + 3)
    // after k > 2 X - n, sum of all neglected terms is < a_k
    constexpr double precision{1e-14};

    for (int i = 0; i <= N; i++) {
        ak(i) = 1.0 / (2 * i + 1);
        denominator(i) = 2 * i + 3;
    }

    result.array() = ak.array();
    double x2 = 2 * x;
    for (int k = 0; k < 15; k++) {
        denominator.array() += 2 * k;
        ak.array() = ak.array() * x2 / denominator.array();
        result.array() += ak.array();
    }
    return result * std::exp(-x);
}

double boys_0(double x) {
    if (x < 1e-2)
        return 1;
    const double sqrt_x = std::sqrt(x);
    const double xerf = std::erf(sqrt_x);
    static const double sqrt_pi = std::sqrt(M_PI);
    return 0.5 * (sqrt_pi * xerf) / sqrt_x;
}

} // namespace occ

namespace occ::ints {

Vec compute_electric_potential2(const Mat &D, const BasisSet &obs,
                                const ShellPairList &shellpair_list,
                                const occ::Mat3N &positions) {

    /*
     * TODO this is a work in progress, currently only implemented for the SS
     * case, and even there it is incomplete.
     * - (2*M_PI Z_c / \gamma_p) * exp(-eta(A - B)^2) * boys_0(-gamma_p * (P -
     * r_c)^2)
     */

    static const double two_pi = 2 * M_PI;

    Vec boys_values = Vec::Zero(5);
    occ::timing::start(occ::timing::category::gto_dist);
    for (int i = 1; i < 1000; i++) {
        boys_values += occ::boys(4, 0.001 * i);
    }
    occ::timing::stop(occ::timing::category::gto_dist);

    Vec boys_values2 = Vec::Zero(5);
    occ::timing::start(occ::timing::category::gto_gen);
    for (int i = 1; i < 1000; i++) {
        boys_values2 += occ::boys_accum(4, 0.001 * i);
    }
    occ::timing::stop(occ::timing::category::gto_gen);

    fmt::print("Boys: ({})\n{}\n",
               occ::timing::total(occ::timing::category::gto_dist),
               boys_values);
    fmt::print("Boys accum: ({})\n{}\n",
               occ::timing::total(occ::timing::category::gto_gen),
               boys_values2);

    Vec result = Vec::Zero(positions.cols());

    for (size_t s1 = 0; s1 < obs.size(); s1++) {
        const auto &shell1 = obs[s1];
        size_t l1 = shell1.contr[0].l;
        const Vec3 p1{shell1.O[0], shell1.O[1], shell1.O[2]};

        size_t nprim1 = shell1.contr[0].size();
        const auto &alpha1 = shell1.alpha;
        const auto &c1 = shell1.contr[0].coeff;

        for (const auto s2 : shellpair_list.at(s1)) {
            const auto &shell2 = obs[s2];
            size_t l1 = shell2.contr[0].l;
            const Vec3 p2{shell2.O[0], shell2.O[1], shell2.O[2]};
            size_t nprim2 = shell2.contr[0].size();
            const auto &alpha2 = shell2.alpha;
            const auto &c2 = shell2.contr[0].coeff;

            double r2 = (p1 - p2).squaredNorm();

            for (size_t i = 0; i < nprim1; i++) {
                for (size_t j = 0; j < nprim2; j++) {
                    double gammap = alpha1[i] + alpha2[j];
                    double etap = (alpha1[i] * alpha2[j]) / gammap;

                    double left_term =
                        c1[i] * c2[j] * two_pi / gammap * std::exp(-etap * r2);
                    Vec3 P = (alpha1[i] * p1 + alpha2[j] * p2) / gammap;
                    for (size_t pt = 0; pt < positions.cols(); pt++) {
                        double x =
                            gammap * (P - positions.col(pt)).squaredNorm();
                        result(pt) -= left_term * boys_0(x);
                    }
                }
            }
        }
    }
    return result;
}

} // namespace occ::ints
