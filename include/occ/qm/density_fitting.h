#pragma once
#include <libint2.hpp>
#include <occ/qm/ints.h>

namespace occ::df {
using occ::qm::BasisSet;
using occ::ints::shellpair_data_t;
using occ::ints::shellpair_list_t;

struct DFFockEngine {

    BasisSet obs;
    BasisSet dfbs;
    Mat Vinv;
    Mat Linv_t;
    Eigen::LLT<Mat> V_LLt;
    const size_t nbf{0}, ndf{0};

    DFFockEngine(const BasisSet &_obs, const BasisSet &_dfbs);

    std::vector<Mat> ints;
    bool ints_populated{false};

    // a DF-based builder, using coefficients of occupied MOs
    Mat compute_2body_fock_dfC(const Mat &Cocc);
    Mat compute_J(const Mat &D);
    Mat compute_J_direct(const Mat &D) const;
    std::pair<Mat, Mat> compute_JK_direct(const Mat &Cocc);

    size_t integral_storage_max_size() const {
        return nbf * nbf * ndf;
    }

    const auto &shellpair_list() const { return m_shellpair_list; }
    const auto &shellpair_data() const { return m_shellpair_data; }


  private:
    shellpair_list_t m_shellpair_list{}; // shellpair list for OBS
    shellpair_data_t m_shellpair_data{}; // shellpair data for OBS

    mutable std::vector<libint2::Engine> m_engines;
    template <typename T> void three_center_integral_helper(T &func) const {
        using occ::parallel::nthreads;

        const auto nshells = obs.size();
        const auto nshells_df = dfbs.size();
        const auto &unitshell = libint2::Shell::unit();

        // construct the 2-electron 3-center repulsion integrals engine
        // since the code assumes (xx|xs) braket, and Engine/libint only
        // produces (xs|xx), use 4-center engine

        auto shell2bf = obs.shell2bf();
        auto shell2bf_df = dfbs.shell2bf();

        auto lambda = [&](int thread_id) {
            auto &engine = m_engines[thread_id];
            const auto &results = engine.results();

            for (auto s1 = 0l, s123 = 0l; s1 != nshells_df; ++s1) {
                if (s1 % nthreads != thread_id)
                    continue;
                auto bf1_first = shell2bf_df[s1];
                auto n1 = dfbs[s1].size();

                for (auto s2 = 0l; s2 != nshells; s2++) {
                    auto bf2 = shell2bf[s2]; // first basis function in this shell
                    auto n2 = obs[s2].size();
                    auto bf2_first = shell2bf[s2];

                    auto s2_offset = s2 * (s2 + 1) / 2;
                    for (auto s3 : m_shellpair_list.at(s2)) {
                        auto bf2 = shell2bf[s3];
                        auto n3 = obs[s3].size();
                        auto bf3_first = shell2bf[s3];

                        engine.compute2<libint2::Operator::coulomb,
                                        libint2::BraKet::xs_xx, 0>(
                            dfbs[s1], unitshell, obs[s2], obs[s3]);
                        const auto *buf = results[0];
                        if (buf == nullptr)
                            continue;

                        func(thread_id, bf1_first, n1, bf2_first, n2, bf3_first,
                             n3, buf);
                    }
                }
            }
        }; // lambda

        occ::parallel::parallel_do(lambda);
    }
};

} // namespace occ::df
