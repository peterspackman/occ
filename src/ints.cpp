#include "ints.h"

namespace craso::ints
{
    RowMajorMatrix compute_2body_2index_ints(const BasisSet &bs)
    {
        using craso::parallel::nthreads;
        using libint2::Engine;
        using libint2::Shell;
        using libint2::BraKet;

        const auto n = bs.nbf();
        const auto nshells = bs.size();
        RowMajorMatrix result = RowMajorMatrix::Zero(n, n);

        // build engines for each thread

        std::vector<Engine> engines(nthreads);
        engines[0] =
            Engine(libint2::Operator::coulomb, bs.max_nprim(), bs.max_l(), 0);
        engines[0].set(BraKet::xs_xs);
        for (size_t i = 1; i != nthreads; ++i)
        {
            engines[i] = engines[0];
        }

        auto shell2bf = bs.shell2bf();
        auto unitshell = Shell::unit();

        auto compute = [&](int thread_id) {
            auto &engine = engines[thread_id];
            const auto &buf = engine.results();

            // loop over unique shell pairs, {s1,s2} such that s1 >= s2
            // this is due to the permutational symmetry of the real integrals over
            // Hermitian operators: (1|2) = (2|1)
            for (auto s1 = 0l, s12 = 0l; s1 != nshells; ++s1)
            {
                auto bf1 = shell2bf[s1]; // first basis function in this shell
                auto n1 = bs[s1].size();

                for (auto s2 = 0; s2 <= s1; ++s2, ++s12)
                {
                    if (s12 % nthreads != thread_id)
                        continue;

                    auto bf2 = shell2bf[s2];
                    auto n2 = bs[s2].size();

                    // compute shell pair; return is the pointer to the buffer
                    engine.compute(bs[s1], bs[s2]);
                    if (buf[0] == nullptr)
                        continue; // if all integrals screened out, skip to next shell set

                    // "map" buffer to a const Eigen Matrix, and copy it to the
                    // corresponding blocks of the result
                    Eigen::Map<const RowMajorMatrix> buf_mat(buf[0], n1, n2);
                    result.block(bf1, bf2, n1, n2) = buf_mat;
                    if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1}
                                  // block, note the transpose!
                        result.block(bf2, bf1, n2, n1) = buf_mat.transpose();
                }
            }
        }; // compute lambda

        craso::parallel::parallel_do(compute);

        return result;
    }

    RowMajorMatrix compute_shellblock_norm(const BasisSet &obs, const RowMajorMatrix &A)
    {
        const auto nsh = obs.size();
        RowMajorMatrix Ash(nsh, nsh);

        auto shell2bf = obs.shell2bf();
        for (size_t s1 = 0; s1 != nsh; ++s1)
        {
            const auto &s1_first = shell2bf[s1];
            const auto &s1_size = obs[s1].size();
            for (size_t s2 = 0; s2 != nsh; ++s2)
            {
                const auto &s2_first = shell2bf[s2];
                const auto &s2_size = obs[s2].size();

                Ash(s1, s2) = A.block(s1_first, s2_first, s1_size, s2_size)
                                  .lpNorm<Eigen::Infinity>();
            }
        }

        return Ash;
    }

    RowMajorMatrix compute_2body_fock(const BasisSet &obs, const shellpair_list_t &shellpair_list,
                                      const shellpair_data_t &shellpair_data, const RowMajorMatrix &D,
                                      double precision,
                                      const RowMajorMatrix &Schwarz)
    {
        const auto n = obs.nbf();
        const auto nshells = obs.size();
        using craso::parallel::nthreads;
        std::vector<RowMajorMatrix> G(nthreads, RowMajorMatrix::Zero(n, n));

        const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
        RowMajorMatrix D_shblk_norm =
            compute_shellblock_norm(obs, D); // matrix of infty-norms of shell blocks

        auto fock_precision = precision;
        // engine precision controls primitive truncation, assume worst-case scenario
        // (all primitive combinations add up constructively)
        auto max_nprim = obs.max_nprim();
        auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
        auto engine_precision = std::min(fock_precision / D_shblk_norm.maxCoeff(),
                                         std::numeric_limits<double>::epsilon()) /
                                max_nprim4;
        assert(engine_precision > max_engine_precision &&
               "using precomputed shell pair data limits the max engine precision"
               " ... make max_engine_precision smalle and recompile");

        // construct the 2-electron repulsion integrals engine pool
        using libint2::Engine;
        std::vector<Engine> engines(nthreads);
        engines[0] = Engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);
        engines[0].set_precision(engine_precision); // shellset-dependent precision
                                                    // control will likely break
                                                    // positive definiteness
                                                    // stick with this simple recipe
        //std::cout << "compute_2body_fock:precision = " << precision << std::endl;
        //std::cout << "Engine::precision = " << engines[0].precision() << std::endl;
        for (size_t i = 1; i != nthreads; ++i)
        {
            engines[i] = engines[0];
        }
        std::atomic<size_t> num_ints_computed{0};

#if defined(REPORT_INTEGRAL_TIMINGS)
        std::vector<libint2::Timers<1>> timers(nthreads);
#endif

        auto shell2bf = obs.shell2bf();

        auto lambda = [&](int thread_id) {
            auto &engine = engines[thread_id];
            auto &g = G[thread_id];
            const auto &buf = engine.results();

#if defined(REPORT_INTEGRAL_TIMINGS)
            auto &timer = timers[thread_id];
            timer.clear();
            timer.set_now_overhead(25);
#endif

            // loop over permutationally-unique set of shells
            for (auto s1 = 0l, s1234 = 0l; s1 != nshells; ++s1)
            {
                auto bf1_first = shell2bf[s1]; // first basis function in this shell
                auto n1 = obs[s1].size();      // number of basis functions in this shell

                auto sp12_iter = shellpair_data.at(s1).begin();

                for (const auto &s2 : shellpair_list.at(s1))
                {
                    auto bf2_first = shell2bf[s2];
                    auto n2 = obs[s2].size();

                    const auto *sp12 = sp12_iter->get();
                    ++sp12_iter;

                    const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

                    for (auto s3 = 0; s3 <= s1; ++s3)
                    {
                        auto bf3_first = shell2bf[s3];
                        auto n3 = obs[s3].size();

                        const auto Dnorm123 =
                            do_schwarz_screen
                                ? std::max(D_shblk_norm(s1, s3),
                                           std::max(D_shblk_norm(s2, s3), Dnorm12))
                                : 0.;

                        auto sp34_iter = shellpair_data.at(s3).begin();

                        const auto s4_max = (s1 == s3) ? s2 : s3;
                        for (const auto &s4 : shellpair_list.at(s3))
                        {
                            if (s4 > s4_max)
                                break; // for each s3, s4 are stored in monotonically increasing
                                       // order

                            // must update the iter even if going to skip s4
                            const auto *sp34 = sp34_iter->get();
                            ++sp34_iter;

                            if ((s1234++) % nthreads != thread_id)
                                continue;

                            const auto Dnorm1234 =
                                do_schwarz_screen
                                    ? std::max(
                                          D_shblk_norm(s1, s4),
                                          std::max(D_shblk_norm(s2, s4),
                                                   std::max(D_shblk_norm(s3, s4), Dnorm123)))
                                    : 0.;

                            if (do_schwarz_screen &&
                                Dnorm1234 * Schwarz(s1, s2) * Schwarz(s3, s4) <
                                    fock_precision)
                                continue;

                            auto bf4_first = shell2bf[s4];
                            auto n4 = obs[s4].size();

                            num_ints_computed += n1 * n2 * n3 * n4;

                            // compute the permutational degeneracy (i.e. # of equivalents) of
                            // the given shell set
                            auto s12_deg = (s1 == s2) ? 1 : 2;
                            auto s34_deg = (s3 == s4) ? 1 : 2;
                            auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
                            auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

#if defined(REPORT_INTEGRAL_TIMINGS)
                            timer.start(0);
#endif

                            engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                                obs[s1], obs[s2], obs[s3], obs[s4], sp12, sp34);
                            const auto *buf_1234 = buf[0];
                            if (buf_1234 == nullptr)
                                continue; // if all integrals screened out, skip to next quartet

#if defined(REPORT_INTEGRAL_TIMINGS)
                            timer.stop(0);
#endif

                            // 1) each shell set of integrals contributes up to 6 shell sets of
                            // the Fock matrix:
                            //    F(a,b) += (ab|cd) * D(c,d)
                            //    F(c,d) += (ab|cd) * D(a,b)
                            //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
                            //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
                            //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
                            //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
                            // 2) each permutationally-unique integral (shell set) must be
                            // scaled by its degeneracy,
                            //    i.e. the number of the integrals/sets equivalent to it
                            // 3) the end result must be symmetrized
                            for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1)
                            {
                                const auto bf1 = f1 + bf1_first;
                                for (auto f2 = 0; f2 != n2; ++f2)
                                {
                                    const auto bf2 = f2 + bf2_first;
                                    for (auto f3 = 0; f3 != n3; ++f3)
                                    {
                                        const auto bf3 = f3 + bf3_first;
                                        for (auto f4 = 0; f4 != n4; ++f4, ++f1234)
                                        {
                                            const auto bf4 = f4 + bf4_first;

                                            const auto value = buf_1234[f1234];

                                            const auto value_scal_by_deg = value * s1234_deg;

                                            g(bf1, bf2) += D(bf3, bf4) * value_scal_by_deg;
                                            g(bf3, bf4) += D(bf1, bf2) * value_scal_by_deg;
                                            g(bf1, bf3) -= 0.25 * D(bf2, bf4) * value_scal_by_deg;
                                            g(bf2, bf4) -= 0.25 * D(bf1, bf3) * value_scal_by_deg;
                                            g(bf1, bf4) -= 0.25 * D(bf2, bf3) * value_scal_by_deg;
                                            g(bf2, bf3) -= 0.25 * D(bf1, bf4) * value_scal_by_deg;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }; // end of lambda

        craso::parallel::parallel_do(lambda);

        // accumulate contributions from all threads
        for (size_t i = 1; i != nthreads; ++i)
        {
            G[0] += G[i];
        }

#if defined(REPORT_INTEGRAL_TIMINGS)
        double time_for_ints = 0.0;
        for (auto &t : timers)
        {
            time_for_ints += t.read(0);
        }
        std::cout << "time for integrals = " << time_for_ints << std::endl;
        for (int t = 0; t != nthreads; ++t)
            engines[t].print_timers();
        std::cout << "# of integrals = " << num_ints_computed << std::endl;
#endif

        // symmetrize the result and return
        RowMajorMatrix GG = 0.5 * (G[0] + G[0].transpose());
        return GG;
    }

    RowMajorMatrix compute_2body_fock_general(const BasisSet &obs, const RowMajorMatrix &D,
                                              const BasisSet &D_bs, bool D_is_shelldiagonal,
                                              double precision)
    {
        const auto n = obs.nbf();
        const auto nshells = obs.size();
        const auto n_D = D_bs.nbf();
        assert(D.cols() == D.rows() && D.cols() == n_D);

        using craso::parallel::nthreads;
        std::vector<RowMajorMatrix> G(nthreads, RowMajorMatrix::Zero(n, n));

        // construct the 2-electron repulsion integrals engine
        using libint2::Engine;
        std::vector<Engine> engines(nthreads);
        engines[0] = Engine(libint2::Operator::coulomb,
                            std::max(obs.max_nprim(), D_bs.max_nprim()),
                            std::max(obs.max_l(), D_bs.max_l()), 0);
        engines[0].set_precision(precision); // shellset-dependent precision control
                                             // will likely break positive
                                             // definiteness
                                             // stick with this simple recipe
        for (size_t i = 1; i != nthreads; ++i)
        {
            engines[i] = engines[0];
        }
        auto shell2bf = obs.shell2bf();
        auto shell2bf_D = D_bs.shell2bf();

        auto lambda = [&](int thread_id) {
            auto &engine = engines[thread_id];
            auto &g = G[thread_id];
            const auto &buf = engine.results();

            // loop over permutationally-unique set of shells
            for (auto s1 = 0l, s1234 = 0l; s1 != nshells; ++s1)
            {
                auto bf1_first = shell2bf[s1]; // first basis function in this shell
                auto n1 = obs[s1].size();      // number of basis functions in this shell

                for (auto s2 = 0; s2 <= s1; ++s2)
                {
                    auto bf2_first = shell2bf[s2];
                    auto n2 = obs[s2].size();

                    for (auto s3 = 0; s3 < D_bs.size(); ++s3)
                    {
                        auto bf3_first = shell2bf_D[s3];
                        auto n3 = D_bs[s3].size();

                        auto s4_begin = D_is_shelldiagonal ? s3 : 0;
                        auto s4_fence = D_is_shelldiagonal ? s3 + 1 : D_bs.size();

                        for (auto s4 = s4_begin; s4 != s4_fence; ++s4, ++s1234)
                        {
                            if (s1234 % nthreads != thread_id)
                                continue;

                            auto bf4_first = shell2bf_D[s4];
                            auto n4 = D_bs[s4].size();

                            // compute the permutational degeneracy (i.e. # of equivalents) of
                            // the given shell set
                            auto s12_deg = (s1 == s2) ? 1.0 : 2.0;

                            if (s3 >= s4)
                            {
                                auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
                                auto s1234_deg = s12_deg * s34_deg;
                                // auto s1234_deg = s12_deg;
                                engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                                    obs[s1], obs[s2], D_bs[s3], D_bs[s4]);
                                const auto *buf_1234 = buf[0];
                                if (buf_1234 != nullptr)
                                {
                                    for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1)
                                    {
                                        const auto bf1 = f1 + bf1_first;
                                        for (auto f2 = 0; f2 != n2; ++f2)
                                        {
                                            const auto bf2 = f2 + bf2_first;
                                            for (auto f3 = 0; f3 != n3; ++f3)
                                            {
                                                const auto bf3 = f3 + bf3_first;
                                                for (auto f4 = 0; f4 != n4; ++f4, ++f1234)
                                                {
                                                    const auto bf4 = f4 + bf4_first;

                                                    const auto value = buf_1234[f1234];
                                                    const auto value_scal_by_deg = value * s1234_deg;
                                                    g(bf1, bf2) += 2.0 * D(bf3, bf4) * value_scal_by_deg;
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                                obs[s1], D_bs[s3], obs[s2], D_bs[s4]);
                            const auto *buf_1324 = buf[0];
                            if (buf_1324 == nullptr)
                                continue; // if all integrals screened out, skip to next quartet

                            for (auto f1 = 0, f1324 = 0; f1 != n1; ++f1)
                            {
                                const auto bf1 = f1 + bf1_first;
                                for (auto f3 = 0; f3 != n3; ++f3)
                                {
                                    const auto bf3 = f3 + bf3_first;
                                    for (auto f2 = 0; f2 != n2; ++f2)
                                    {
                                        const auto bf2 = f2 + bf2_first;
                                        for (auto f4 = 0; f4 != n4; ++f4, ++f1324)
                                        {
                                            const auto bf4 = f4 + bf4_first;

                                            const auto value = buf_1324[f1324];
                                            const auto value_scal_by_deg = value * s12_deg;
                                            g(bf1, bf2) -= D(bf3, bf4) * value_scal_by_deg;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }; // thread lambda

        craso::parallel::parallel_do(lambda);

        // accumulate contributions from all threads
        for (size_t i = 1; i != nthreads; ++i)
        {
            G[0] += G[i];
        }

        // symmetrize the result and return
        return 0.5 * (G[0] + G[0].transpose());
    }

} // namespace craso::ints