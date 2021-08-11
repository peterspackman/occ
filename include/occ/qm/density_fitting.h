#pragma once
#include <occ/qm/ints.h>
#include <libint2.hpp>


namespace occ::df {
using occ::qm::BasisSet;

struct DFFockEngine
{

    BasisSet obs;
    BasisSet dfbs;
    Mat Linv_t;
    const size_t nbf{0}, ndf{0};

    DFFockEngine(const BasisSet& _obs, const BasisSet& _dfbs);

    std::array<size_t, 3> xyK_dims;
    std::vector<double> xyK;

    // a DF-based builder, using coefficients of occupied MOs
    Mat compute_2body_fock_dfC(const Mat& Cocc);

private:
    template <typename T>
    void three_center_integral_helper(T& func)
    {
        using occ::parallel::nthreads;

        const auto nshells = obs.size();
        const auto nshells_df = dfbs.size();
        const auto& unitshell = libint2::Shell::unit();

        // construct the 2-electron 3-center repulsion integrals engine
        // since the code assumes (xx|xs) braket, and Engine/libint only produces
        // (xs|xx), use 4-center engine
        std::vector<libint2::Engine> engines(nthreads);
        engines[0] = libint2::Engine(libint2::Operator::coulomb,
                std::max(obs.max_nprim(), dfbs.max_nprim()),
                std::max(obs.max_l(), dfbs.max_l()), 0);
        engines[0].set(libint2::BraKet::xs_xx);
        for (size_t i = 1; i != nthreads; ++i)
        {
            engines[i] = engines[0];
        }

        auto shell2bf = obs.shell2bf();
        auto shell2bf_df = dfbs.shell2bf();

        auto lambda = [&](int thread_id)
        {
            auto& engine = engines[thread_id];
            const auto& results = engine.results();

            for (auto s1 = 0l, s123 = 0l; s1 != nshells_df; ++s1)
            {
                auto bf1_first = shell2bf_df[s1];
                auto n1 = dfbs[s1].size();

                for (auto s2 = 0; s2 < nshells; ++s2)
                {
                    auto bf2_first = shell2bf[s2];
                    auto n2 = obs[s2].size();
                    const auto n12 = n1 * n2;

                    for (auto s3 = 0; s3 < nshells; ++s3, ++s123)
                    {
                        if (s123 % nthreads != thread_id) continue;

                        auto bf3_first = shell2bf[s3];
                        auto n3 = obs[s3].size();
                        const auto n123 = n12 * n3;

                        engine.compute2<libint2::Operator::coulomb, libint2::BraKet::xs_xx, 0>(
                                dfbs[s1], unitshell, obs[s2], obs[s3]);
                        const auto* buf = results[0];
                        if (buf == nullptr)
                            continue;

                        func(thread_id, bf1_first, n1, bf2_first, n2, bf3_first, n3, buf);
                    }
                }
            }

        };  // lambda

        occ::parallel::parallel_do(lambda);

    }

};

}
