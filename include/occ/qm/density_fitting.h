#pragma once
#include <libint2.hpp>
#include <occ/qm/ints.h>
#include <occ/qm/mo.h>

namespace occ::df {
using occ::qm::BasisSet;
using occ::ints::shellpair_data_t;
using occ::ints::shellpair_list_t;
using occ::qm::MolecularOrbitals;

struct DFFockEngine {

    BasisSet obs;
    BasisSet dfbs;
<<<<<<< HEAD
    Eigen::LLT<Mat> V_LLt, Vsqrt_LLt;
=======
    Eigen::LDLT<Mat> V_LLt, Vsqrt_LLt;
>>>>>>> c84ebb7437d82825e1a462094e2e155c1cad9ae5
    const size_t nbf{0}, ndf{0};

    DFFockEngine(const BasisSet &_obs, const BasisSet &_dfbs);

    std::vector<Mat> ints;

    // a DF-based builder, using coefficients of occupied MOs
    Mat compute_J(const MolecularOrbitals&);
    Mat compute_J_direct(const MolecularOrbitals&) const;
    Mat compute_K(const MolecularOrbitals&);
    Mat compute_K_direct(const MolecularOrbitals&) const;
    std::pair<Mat, Mat> compute_JK(const MolecularOrbitals&);
    std::pair<Mat, Mat> compute_JK_direct(const MolecularOrbitals&) const;
    Mat compute_fock(const MolecularOrbitals&);
    Mat compute_fock_direct(const MolecularOrbitals&) const;

    size_t num_rows() const {
        size_t n = 0;
        for(size_t s1 = 0; s1 < obs.size(); s1++) {
            size_t s1_size = obs[s1].size();
            size_t pairs_size = 0;
            for(const auto& s2 : m_shellpair_list.at(s1)) {
                pairs_size += obs[s2].size();
            }
            n += s1 * pairs_size;
        }
        return n;
    }

    size_t integral_storage_max_size() const {
        return ndf * num_rows();
    }

    const auto &shellpair_list() const { return m_shellpair_list; }
    const auto &shellpair_data() const { return m_shellpair_data; }


  private:

    void populate_integrals();
    Mat m_ints;
    bool m_have_integrals{false};
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

            for (auto s1 = 0l; s1 != nshells_df; ++s1) {
                if (s1 % nthreads != thread_id)
                    continue;
                auto bf1_first = shell2bf_df[s1];
                auto n1 = dfbs[s1].size();

                for (auto s2 = 0l; s2 != nshells; s2++) {
                    auto n2 = obs[s2].size();
                    auto bf2_first = shell2bf[s2];

                    for (auto s3 : m_shellpair_list.at(s2)) {
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
