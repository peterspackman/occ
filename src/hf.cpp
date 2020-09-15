#include "hf.h"
#include "parallel.h"
#include <Eigen/Dense>
#include <libint2/chemistry/sto3g_atomic_density.h>

namespace craso::hf
{

    HartreeFock::HartreeFock(const std::vector<libint2::Atom> &atoms, const BasisSet &basis) : m_atoms(atoms), m_basis(basis)
    {
        if (!libint2::initialized())
            libint2::initialize();
        std::tie(m_shellpair_list, m_shellpair_data) = compute_shellpairs(m_basis);
        for (const auto &a : m_atoms)
        {
            m_num_e += a.atomic_number;
        }
        m_num_e -= m_charge;
    }

    std::tuple<shellpair_list_t, shellpair_data_t>
    compute_shellpairs(const BasisSet &bs1, const BasisSet &_bs2, const double threshold)
    {
        using libint2::Operator;
        const BasisSet &bs2 = (_bs2.empty() ? bs1 : _bs2);
        const auto nsh1 = bs1.size();
        const auto nsh2 = bs2.size();
        const auto bs1_equiv_bs2 = (&bs1 == &bs2);

        using craso::parallel::nthreads;

        // construct the 2-electron repulsion integrals engine
        using libint2::Engine;
        std::vector<Engine> engines;
        engines.reserve(nthreads);
        engines.emplace_back(Operator::overlap,
                             std::max(bs1.max_nprim(), bs2.max_nprim()),
                             std::max(bs1.max_l(), bs2.max_l()), 0);
        for (size_t i = 1; i != nthreads; ++i)
        {
            engines.push_back(engines[0]);
        }

        std::cout << "computing non-negligible shell-pair list ... ";

        libint2::Timers<1> timer;
        timer.set_now_overhead(25);
        timer.start(0);

        shellpair_list_t splist;

        std::mutex mx;

        auto compute = [&](int thread_id) {
            auto &engine = engines[thread_id];
            const auto &buf = engine.results();

            // loop over permutationally-unique set of shells
            for (auto s1 = 0l, s12 = 0l; s1 != nsh1; ++s1)
            {
                mx.lock();
                if (splist.find(s1) == splist.end())
                    splist.insert(std::make_pair(s1, std::vector<size_t>()));
                mx.unlock();

                auto n1 = bs1[s1].size(); // number of basis functions in this shell

                auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
                for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12)
                {
                    if (s12 % nthreads != thread_id)
                        continue;

                    auto on_same_center = (bs1[s1].O == bs2[s2].O);
                    bool significant = on_same_center;
                    if (not on_same_center)
                    {
                        auto n2 = bs2[s2].size();
                        engines[thread_id].compute(bs1[s1], bs2[s2]);
                        Eigen::Map<const RowMajorMatrix> buf_mat(buf[0], n1, n2);
                        auto norm = buf_mat.norm();
                        significant = (norm >= threshold);
                    }

                    if (significant)
                    {
                        mx.lock();
                        splist[s1].emplace_back(s2);
                        mx.unlock();
                    }
                }
            }
        }; // end of compute

        craso::parallel::parallel_do(compute);

        // resort shell list in increasing order, i.e. splist[s][s1] < splist[s][s2] if s1 < s2
        // N.B. only parallelized over 1 shell index
        auto sort = [&](int thread_id) {
            for (auto s1 = 0l; s1 != nsh1; ++s1)
            {
                if (s1 % nthreads == thread_id)
                {
                    auto &list = splist[s1];
                    std::sort(list.begin(), list.end());
                }
            }
        }; // end of sort

        craso::parallel::parallel_do(sort);

        // compute shellpair data assuming that we are computing to default_epsilon
        // N.B. only parallelized over 1 shell index
        const auto ln_max_engine_precision = std::log(max_engine_precision);
        shellpair_data_t spdata(splist.size());
        auto make_spdata = [&](int thread_id) {
            for (auto s1 = 0l; s1 != nsh1; ++s1)
            {
                if (s1 % nthreads == thread_id)
                {
                    for (const auto &s2 : splist[s1])
                    {
                        spdata[s1].emplace_back(std::make_shared<libint2::ShellPair>(bs1[s1], bs2[s2], ln_max_engine_precision));
                    }
                }
            }
        }; // end of make_spdata

        craso::parallel::parallel_do(make_spdata);

        timer.stop(0);
        std::cout << "done (" << timer.read(0) << " s)" << std::endl;
        return std::make_tuple(splist, spdata);
    }

    double HartreeFock::nuclear_repulsion_energy() const
    {
        double enuc = 0.0;
        for (auto i = 0; i < m_atoms.size(); i++)
            for (auto j = i + 1; j < m_atoms.size(); j++)
            {
                auto xij = m_atoms[i].x - m_atoms[j].x;
                auto yij = m_atoms[i].y - m_atoms[j].y;
                auto zij = m_atoms[i].z - m_atoms[j].z;
                auto r2 = xij * xij + yij * yij + zij * zij;
                auto r = sqrt(r2);
                enuc += m_atoms[i].atomic_number * m_atoms[j].atomic_number / r;
            }
        return enuc;
    }

    RowMajorMatrix HartreeFock::compute_soad() const
    {
        // computes Superposition-Of-Atomic-Densities guess for the molecular density
        // matrix
        // in minimal basis; occupies subshells by smearing electrons evenly over the
        // orbitals
        // compute number of atomic orbitals
        size_t nao = 0;
        for (const auto &atom : m_atoms)
        {
            const auto Z = atom.atomic_number;
            nao += libint2::sto3g_num_ao(Z);
        }

        // compute the minimal basis density
        RowMajorMatrix D = RowMajorMatrix::Zero(nao, nao);
        size_t ao_offset = 0; // first AO of this atom
        for (const auto &atom : m_atoms)
        {
            const auto Z = atom.atomic_number;
            const auto &occvec = libint2::sto3g_ao_occupation_vector(Z);
            for (const auto &occ : occvec)
            {
                D(ao_offset, ao_offset) = occ;
                ++ao_offset;
            }
        }

        return D * 0.5; // we use densities normalized to # of electrons/2
    }

    RowMajorMatrix HartreeFock::compute_shellblock_norm(const RowMajorMatrix &A) const
    {
        const auto nsh = m_basis.size();
        RowMajorMatrix Ash(nsh, nsh);

        auto shell2bf = m_basis.shell2bf();
        for (size_t s1 = 0; s1 != nsh; ++s1)
        {
            const auto &s1_first = shell2bf[s1];
            const auto &s1_size = m_basis.at(s1).size();
            for (size_t s2 = 0; s2 != nsh; ++s2)
            {
                const auto &s2_first = shell2bf[s2];
                const auto &s2_size = m_basis.at(s2).size();

                Ash(s1, s2) = A.block(s1_first, s2_first, s1_size, s2_size)
                                  .lpNorm<Eigen::Infinity>();
            }
        }

        return Ash;
    }

    RowMajorMatrix HartreeFock::compute_2body_fock(double precision, const RowMajorMatrix &Schwarz) const
    {
        const auto n = m_basis.nbf();
        const auto nshells = m_basis.size();
        using craso::parallel::nthreads;
        std::vector<RowMajorMatrix> G(nthreads, RowMajorMatrix::Zero(n, n));

        const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
        RowMajorMatrix D_shblk_norm = compute_shellblock_norm(m_density); // RowMajorMatrix of infty-norms of shell blocks

        auto fock_precision = precision;
        // engine precision controls primitive truncation, assume worst-case scenario
        // (all primitive combinations add up constructively)
        auto max_nprim = m_basis.max_nprim();
        auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
        auto engine_precision = std::min(fock_precision / D_shblk_norm.maxCoeff(),
                                         std::numeric_limits<double>::epsilon()) /
                                max_nprim4;
        assert(engine_precision > max_engine_precision &&
               "using precomputed shell pair data limits the max engine precision"
               " ... make max_engine_precision smalle and recompile");

        // construct the 2-electron repulsion integrals engine pool
        using libint2::Engine;
        using libint2::Operator;
        using libint2::BraKet;
        std::vector<Engine> engines(nthreads);
        engines[0] = Engine(Operator::coulomb, m_basis.max_nprim(), m_basis.max_l(), 0);
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

        auto shell2bf = m_basis.shell2bf();

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
                auto n1 = m_basis[s1].size();      // number of basis functions in this shell

                auto sp12_iter = m_shellpair_data.at(s1).begin();

                for (const auto &s2 : m_shellpair_list.at(s1))
                {
                    auto bf2_first = shell2bf[s2];
                    auto n2 = m_basis[s2].size();

                    const auto *sp12 = sp12_iter->get();
                    ++sp12_iter;

                    const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

                    for (auto s3 = 0; s3 <= s1; ++s3)
                    {
                        auto bf3_first = shell2bf[s3];
                        auto n3 = m_basis[s3].size();

                        const auto Dnorm123 =
                            do_schwarz_screen
                                ? std::max(D_shblk_norm(s1, s3),
                                           std::max(D_shblk_norm(s2, s3), Dnorm12))
                                : 0.;

                        auto sp34_iter = m_shellpair_data.at(s3).begin();

                        const auto s4_max = (s1 == s3) ? s2 : s3;
                        for (const auto &s4 : m_shellpair_list.at(s3))
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
                            auto n4 = m_basis[s4].size();

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
                                m_basis[s1], m_basis[s2], m_basis[s3], m_basis[s4], sp12, sp34);
                            const auto *buf_1234 = buf[0];
                            if (buf_1234 == nullptr)
                                continue; // if all integrals screened out, skip to next quartet

#if defined(REPORT_INTEGRAL_TIMINGS)
                            timer.stop(0);
#endif

                            // 1) each shell set of integrals contributes up to 6 shell sets of
                            // the Fock RowMajorMatrix:
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

                                            g(bf1, bf2) += m_density(bf3, bf4) * value_scal_by_deg;
                                            g(bf3, bf4) += m_density(bf1, bf2) * value_scal_by_deg;
                                            g(bf1, bf3) -= 0.25 * m_density(bf2, bf4) * value_scal_by_deg;
                                            g(bf2, bf4) -= 0.25 * m_density(bf1, bf3) * value_scal_by_deg;
                                            g(bf1, bf4) -= 0.25 * m_density(bf2, bf3) * value_scal_by_deg;
                                            g(bf2, bf3) -= 0.25 * m_density(bf1, bf4) * value_scal_by_deg;
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
} // namespace craso::hf