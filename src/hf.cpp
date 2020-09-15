#include "hf.h"
#include "parallel.h"
#include <Eigen/Dense>
#include <libint2/chemistry/sto3g_atomic_density.h>

namespace craso::hf
{

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
        std::cout << "overlap condition number = " << S_condition_number;
        if (obs_nbf_omitted > 0)
            std::cout << " (dropped " << obs_nbf_omitted << " "
                      << (obs_nbf_omitted > 1 ? "fns" : "fn") << " to reduce to "
                      << XtX_condition_number << ")";
        std::cout << std::endl;

        if (obs_nbf_omitted > 0)
        {
            RowMajorMatrix should_be_I = X.transpose() * S * X;
            RowMajorMatrix I = RowMajorMatrix::Identity(should_be_I.rows(), should_be_I.cols());
            std::cout << "||X^t * S * X - I||_2 = " << (should_be_I - I).norm()
                      << " (should be 0)" << std::endl;
        }

        return std::make_tuple(X, Xinv, XtX_condition_number);
    }

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
        return craso::ints::compute_shellblock_norm(m_basis, A);
    }

    RowMajorMatrix HartreeFock::compute_2body_fock(double precision, const RowMajorMatrix &Schwarz) const
    {
        return craso::ints::compute_2body_fock(m_basis, m_shellpair_list, m_shellpair_data, m_density, precision, Schwarz);
    }

    void HartreeFock::compute_initial_guess()
    {
        int ndocc = m_num_e / 2;
        const auto tstart = std::chrono::high_resolution_clock::now();
        auto S = compute_overlap_integrals();
        auto T = compute_kinetic_energy_integrals();
        auto V = compute_nuclear_attraction_integrals();
        RowMajorMatrix H = T + V;
        RowMajorMatrix C;
        RowMajorMatrix C_occ;
        auto D_minbs = compute_soad(); // compute guess in minimal basis
        BasisSet minbs("STO-3G", m_atoms);
        if (minbs == m_basis)
            m_density = D_minbs;
        else
        {
            // compute orthogonalizer X such that X.transpose() . S . X = I
            RowMajorMatrix X, Xinv;
            double XtX_condition_number; // condition number of "re-conditioned"
                                         // overlap obtained as Xinv.transpose() . Xinv
            // one should think of columns of Xinv as the conditioned basis
            // Re: name ... cond # (Xinv.transpose() . Xinv) = cond # (X.transpose() .
            // X)
            // by default assume can manage to compute with condition number of S <=
            // 1/eps
            // this is probably too optimistic, but in well-behaved cases even 10^11 is
            // OK
            double S_condition_number_threshold =
                1.0 / std::numeric_limits<double>::epsilon();
            std::tie(X, Xinv, XtX_condition_number) =
                conditioning_orthogonalizer(S, S_condition_number_threshold);
            // if basis != minimal basis, map non-representable SOAD guess
            // into the AO basis
            // by diagonalizing a Fock matrix
            std::cout << "projecting SOAD into AO basis ... ";
            auto F = H;
            F += craso::ints::compute_2body_fock_general(m_basis, D_minbs, minbs, true, std::numeric_limits<double>::epsilon());

            // solve F C = e S C by (conditioned) transformation to F' C' = e C',
            // where
            // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
            Eigen::SelfAdjointEigenSolver<RowMajorMatrix> eig_solver(X.transpose() * F * X);
            C = X * eig_solver.eigenvectors();

            // compute density, D = C(occ) . C(occ)T
            C_occ = C.leftCols(ndocc);
            m_density = C_occ * C_occ.transpose();

            const auto tstop = std::chrono::high_resolution_clock::now();
            const std::chrono::duration<double> time_elapsed = tstop - tstart;
            std::cout << "done (" << time_elapsed.count() << " s)" << std::endl;
        }
    }
} // namespace craso::hf