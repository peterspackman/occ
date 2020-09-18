#include "hf.h"
#include "parallel.h"
#include <fmt/core.h>
#include <libint2/chemistry/sto3g_atomic_density.h>

namespace craso::hf
{

    HartreeFock::HartreeFock(const std::vector<libint2::Atom> &atoms, const BasisSet &basis) : m_atoms(atoms), m_basis(basis)
    {
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

        fmt::print("Computing non-negligible shell-pair list ");

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
                        Eigen::Map<const MatRM> buf_mat(buf[0], n1, n2);
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
        fmt::print(" {:.6f} s\n", timer.read(0));
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

    MatRM HartreeFock::compute_soad() const
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
        MatRM D = MatRM::Zero(nao, nao);
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

    MatRM HartreeFock::compute_shellblock_norm(const MatRM &A) const
    {
        return craso::ints::compute_shellblock_norm(m_basis, A);
    }

    MatRM HartreeFock::compute_2body_fock(const MatRM& D,
            double precision, const MatRM &Schwarz) const
    {
        return craso::ints::compute_2body_fock(m_basis, m_shellpair_list, m_shellpair_data, D, precision, Schwarz);
    }
    
} // namespace craso::hf
