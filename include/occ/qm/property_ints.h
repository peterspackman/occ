#pragma once
#include <libint2/engine.h>
#include <occ/3rdparty/robin_hood.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/parallel.h>
#include <occ/qm/basisset.h>
#include <vector>

namespace occ::ints {

using libint2::BraKet;
using libint2::Operator;
using libint2::Shell;
using occ::qm::BasisSet;
using occ::qm::ShellPairData;
using occ::qm::ShellPairList;
using occ::qm::SpinorbitalKind;

template <SpinorbitalKind kind>
occ::Vec compute_electric_potential(const Mat &D, const BasisSet &obs,
                                    const ShellPairList &shellpair_list,
                                    const occ::Mat3N &positions) {
    occ::timing::start(occ::timing::category::ints1e);

    using occ::qm::expectation;
    const auto n = obs.nbf();
    const auto nshells = obs.size();
    using occ::parallel::nthreads;
    typedef std::array<Mat, libint2::operator_traits<Operator::nuclear>::nopers>
        result_type;
    const unsigned int nopers =
        libint2::operator_traits<Operator::nuclear>::nopers;

    occ::Vec result(positions.cols());
    std::vector<libint2::Engine> engines(nthreads);
    engines[0] =
        libint2::Engine(Operator::nuclear, obs.max_nprim(), obs.max_l(), 0);
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }
    std::vector<Mat> opmats(nthreads, Mat::Zero(n, n));
    auto shell2bf = obs.shell2bf();

    auto compute = [&](int thread_id) {
        for (size_t pt = 0; pt < positions.cols(); pt++) {
            if (pt % nthreads != thread_id)
                continue;
            opmats[thread_id].setZero();
            std::vector<std::pair<double, std::array<double, 3>>> chgs{
                {1, {positions(0, pt), positions(1, pt), positions(2, pt)}}};
            engines[thread_id].set_params(chgs);

            const auto &buf = engines[thread_id].results();

            // loop over unique shell pairs, {s1,s2} such that s1 >= s2
            // this is due to the permutational symmetry of the real integrals
            // over Hermitian operators: (1|2) = (2|1)
            for (size_t s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
                size_t bf1 = shell2bf[s1]; // first basis function in this shell
                size_t n1 = obs[s1].size();

                size_t s1_offset = s1 * (s1 + 1) / 2;
                for (size_t s2 : shellpair_list.at(s1)) {
                    size_t s12 = s1_offset + s2;
                    size_t bf2 = shell2bf[s2];
                    size_t n2 = obs[s2].size();

                    // compute shell pair; return is the pointer to the buffer
                    engines[thread_id].compute(obs[s1], obs[s2]);

                    Eigen::Map<const MatRM> buf_mat(buf[0], n1, n2);
                    opmats[thread_id].block(bf1, bf2, n1, n2) = buf_mat;
                    if (s1 !=
                        s2) // if s1 >= s2, copy {s1,s2} to the corresponding
                        // {s2,s1} block, note the transpose!
                        opmats[thread_id].block(bf2, bf1, n2, n1) =
                            buf_mat.transpose();
                }
            }
            result(pt) = 2 * expectation<kind>(D, opmats[thread_id]);
        }
    }; // compute lambda
    occ::parallel::parallel_do(compute);
    occ::timing::stop(occ::timing::category::ints1e);
    return result;
}

template <SpinorbitalKind kind>
occ::Mat3N compute_electric_field(const Mat &D, const BasisSet &obs,
                                  const ShellPairList &shellpair_list,
                                  const Mat3N &positions) {
    occ::timing::start(occ::timing::category::ints1e);
    using occ::parallel::nthreads;
    using occ::qm::expectation;
    // seems to be correct for very small molecules but has issues -- use finite
    // differences for now

    const auto n = obs.nbf();
    const auto nshells = obs.size();
    const auto nresults = libint2::num_geometrical_derivatives(1, 1);
    occ::Mat3N result(3, positions.cols());

    std::vector<Mat> xmats(nthreads, Mat::Zero(n, n));
    std::vector<Mat> ymats(nthreads, Mat::Zero(n, n));
    std::vector<Mat> zmats(nthreads, Mat::Zero(n, n));

    // construct the 1-body integrals engine
    std::vector<libint2::Engine> engines(nthreads);
    engines[0] =
        libint2::Engine(Operator::nuclear, obs.max_nprim(), obs.max_l(), 1);

    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }
    auto shell2bf = obs.shell2bf();

    auto compute = [&](int thread_id) {
        for (size_t pt = 0; pt < positions.cols(); pt++) {
            if (pt % nthreads != thread_id)
                continue;
            std::vector<std::pair<double, std::array<double, 3>>> chgs{
                {1, {positions(0, pt), positions(1, pt), positions(2, pt)}}};
            engines[thread_id].set_params(chgs);
            const auto &buf = engines[thread_id].results();
            for (size_t s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
                size_t bf1 = shell2bf[s1]; // first basis function in this shell
                size_t n1 = obs[s1].size();
                size_t s1_offset = s1 * (s1 + 1) / 2;
                for (size_t s2 : shellpair_list.at(s1)) {
                    size_t s12 = s1_offset + s2;
                    size_t bf2 = shell2bf[s2];
                    size_t n2 = obs[s2].size();
                    engines[thread_id].compute(obs[s1], obs[s2]);

                    Eigen::Map<const MatRM> buf_mat_x(buf[0], n1, n2),
                        buf_mat_y(buf[1], n1, n2), buf_mat_z(buf[2], n1, n2);
                    Eigen::Map<const MatRM> buf_mat_x2(buf[3], n1, n2),
                        buf_mat_y2(buf[4], n1, n2), buf_mat_z2(buf[5], n1, n2);
                    xmats[thread_id].block(bf1, bf2, n1, n2) =
                        -(buf_mat_x + buf_mat_x2);
                    ymats[thread_id].block(bf1, bf2, n1, n2) =
                        -(buf_mat_y + buf_mat_y2);
                    zmats[thread_id].block(bf1, bf2, n1, n2) =
                        -(buf_mat_z + buf_mat_z2);
                    if (s1 != s2) {
                        xmats[thread_id].block(bf2, bf1, n2, n1) =
                            -(buf_mat_x + buf_mat_x2).transpose();
                        ymats[thread_id].block(bf2, bf1, n2, n1) =
                            -(buf_mat_y + buf_mat_y2).transpose();
                        zmats[thread_id].block(bf2, bf1, n2, n1) =
                            -(buf_mat_z + buf_mat_z2).transpose();
                    }
                }
            }
            result(0, pt) = -2 * expectation<kind>(D, xmats[thread_id]);
            result(1, pt) = -2 * expectation<kind>(D, ymats[thread_id]);
            result(2, pt) = -2 * expectation<kind>(D, zmats[thread_id]);
        }
    };
    occ::parallel::parallel_do(compute);
    occ::timing::stop(occ::timing::category::ints1e);
    return result;
}

} // namespace occ::ints
