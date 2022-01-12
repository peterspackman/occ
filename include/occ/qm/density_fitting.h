#pragma once
#include <libint2.hpp>
#include <occ/qm/ints.h>
#include <occ/qm/mo.h>
#include <occ/core/timings.h>

namespace occ::df {
using occ::qm::BasisSet;
using occ::ints::shellpair_data_t;
using occ::ints::shellpair_list_t;
using occ::qm::MolecularOrbitals;

struct DFFockEngine {

    enum Policy {
        Choose,
        Direct,
        Stored
    };

    constexpr static double default_prec = std::numeric_limits<double>::epsilon();

    size_t memory_limit{200 * 1024 * 1024}; // 200 MiB
    BasisSet obs;
    BasisSet dfbs;
    Eigen::LLT<Mat> V_LLt, Vsqrt_LLt;
    const size_t nbf{0}, ndf{0};

    DFFockEngine(const BasisSet &_obs, const BasisSet &_dfbs);

    std::vector<Mat> ints;

    // a DF-based builder, using coefficients of occupied MOs
    //
        
    Mat compute_J(const MolecularOrbitals&, double precision = default_prec, const Mat &Schwarz = Mat(), Policy policy = Policy::Choose);
    Mat compute_K(const MolecularOrbitals&, double precision = default_prec, const Mat &Schwarz = Mat(), Policy policy = Policy::Choose);
    std::pair<Mat, Mat> compute_JK(const MolecularOrbitals&, double precision = default_prec, const Mat &Schwarz = Mat(), Policy policy = Policy::Choose);
    Mat compute_fock(const MolecularOrbitals&, double precision = default_prec, const Mat &Schwarz = Mat(), Policy policy = Policy::Choose);

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
    Mat compute_J_stored(const MolecularOrbitals&);
    Mat compute_K_stored(const MolecularOrbitals&);
    std::pair<Mat, Mat> compute_JK_stored(const MolecularOrbitals&);
    Mat compute_fock_stored(const MolecularOrbitals&);
    Mat compute_J_direct(const MolecularOrbitals&, double, const Mat&);
    Mat compute_K_direct(const MolecularOrbitals&, double, const Mat&);
    std::pair<Mat, Mat> compute_JK_direct(const MolecularOrbitals&, double, const Mat&);
    Mat compute_fock_direct(const MolecularOrbitals&, double, const Mat&);


    void populate_integrals();
    Mat m_ints;
    bool m_have_integrals{false};
    shellpair_list_t m_shellpair_list{}; // shellpair list for OBS
    shellpair_data_t m_shellpair_data{}; // shellpair data for OBS

    mutable std::vector<libint2::Engine> m_engines;
    template <typename T> void three_center_integral_helper(T &func, const Mat &D, double precision = default_prec, const Mat &Schwarz = Mat()) const {
        using occ::parallel::nthreads;

        const auto nshells = obs.size();
        const auto nshells_df = dfbs.size();
        const auto &unitshell = libint2::Shell::unit();
        Mat D_shblk_norm = occ::ints::compute_shellblock_norm(obs, D);
        const bool do_schwarz_screen =
            Schwarz.cols() != 0 && Schwarz.rows() != 0;



        // construct the 2-electron 3-center repulsion integrals engine
        // since the code assumes (xx|xs) braket, and Engine/libint only
        // produces (xs|xx), use 4-center engine

        auto shell2bf = obs.shell2bf();
        auto shell2bf_df = dfbs.shell2bf();
        size_t num_skipped = 0;

        auto lambda = [&](int thread_id) {
            auto &engine = m_engines[thread_id];
            const auto &results = engine.results();

            for (auto s1 = 0l; s1 != nshells_df; ++s1) {
                if (s1 % nthreads != thread_id)
                    continue;
                auto bf1_first = shell2bf_df[s1];
                auto n1 = dfbs[s1].size();

                for (auto s2 = 0l; s2 != nshells; s2++) {
		    auto sp23_iter = m_shellpair_data.at(s2).begin();
                    auto n2 = obs[s2].size();
                    auto bf2_first = shell2bf[s2];


                    for (auto s3 : m_shellpair_list.at(s2)) {
			const auto* sp23 = sp23_iter->get();
			++sp23_iter;
                        const auto Dnorm23 =
                            do_schwarz_screen ? D_shblk_norm(s2, s3) : 0.;
                        if(do_schwarz_screen && (Schwarz(s2, s3) < precision)) {
                            num_skipped++;
                            continue;
                        }

                        auto n3 = obs[s3].size();
                        auto bf3_first = shell2bf[s3];

                        engine.compute2<libint2::Operator::coulomb,
                                        libint2::BraKet::xs_xx, 0>(
                            dfbs[s1], unitshell, obs[s2], obs[s3], nullptr, sp23);
                        const auto *buf = results[0];
                        if (buf == nullptr)
                            continue;

                        func(thread_id, bf1_first, n1, bf2_first, n2, bf3_first,
                             n3, buf);
                    }
                }
            }
        }; // lambda

	occ::timing::start(occ::timing::category::df);
        occ::parallel::parallel_do(lambda);
	occ::timing::stop(occ::timing::category::df);
    }
};

} // namespace occ::df
