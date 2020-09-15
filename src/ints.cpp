#include "ints.h"
#include "parallel.h"

namespace craso::ints
{
    template <Operator obtype, typename OperatorParams = typename libint2::operator_traits<obtype>::oper_params_type>
    std::array<RowMajorMatrix, libint2::operator_traits<obtype>::nopers>
    compute_1body_ints(const BasisSet &obs, const shellpair_list_t& shellpair_list, OperatorParams oparams)
    {
        const auto n = obs.nbf();
        const auto nshells = obs.size();
        using craso::parallel::nthreads;
        typedef std::array<RowMajorMatrix, libint2::operator_traits<obtype>::nopers>
            result_type;
        const unsigned int nopers = libint2::operator_traits<obtype>::nopers;
        result_type result;
        for (auto &r : result)
            r = RowMajorMatrix::Zero(n, n);

        // construct the 1-body integrals engine
        std::vector<libint2::Engine> engines(nthreads);
        engines[0] = libint2::Engine(obtype, obs.max_nprim(), obs.max_l(), 0);
        // pass operator params to the engine, e.g.
        // nuclear attraction ints engine needs to know where the charges sit ...
        // the nuclei are charges in this case; in QM/MM there will also be classical
        // charges
        engines[0].set_params(oparams);
        for (size_t i = 1; i != nthreads; ++i)
        {
            engines[i] = engines[0];
        }

        auto shell2bf = obs.shell2bf();

        auto compute = [&](int thread_id) {
            const auto &buf = engines[thread_id].results();

            // loop over unique shell pairs, {s1,s2} such that s1 >= s2
            // this is due to the permutational symmetry of the real integrals over
            // Hermitian operators: (1|2) = (2|1)
            for (auto s1 = 0l, s12 = 0l; s1 != nshells; ++s1)
            {
                auto bf1 = shell2bf[s1]; // first basis function in this shell
                auto n1 = obs[s1].size();

                auto s1_offset = s1 * (s1 + 1) / 2;
                for (auto s2 : shellpair_list.at(s1))
                {
                    auto s12 = s1_offset + s2;
                    if (s12 % nthreads != thread_id)
                        continue;

                    auto bf2 = shell2bf[s2];
                    auto n2 = obs[s2].size();

                    auto n12 = n1 * n2;

                    // compute shell pair; return is the pointer to the buffer
                    engines[thread_id].compute(obs[s1], obs[s2]);

                    for (unsigned int op = 0; op != nopers; ++op)
                    {
                        // "map" buffer to a const Eigen RowMajorMatrix, and copy it to the
                        // corresponding blocks of the result
                        Eigen::Map<const RowMajorMatrix> buf_mat(buf[op], n1, n2);
                        result[op].block(bf1, bf2, n1, n2) = buf_mat;
                        if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding
                                      // {s2,s1} block, note the transpose!
                            result[op].block(bf2, bf1, n2, n1) = buf_mat.transpose();
                    }
                }
            }
        }; // compute lambda
        craso::parallel::parallel_do(compute);
        return result;
    }
} // namespace craso::ints