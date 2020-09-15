#include "ints.h"
#include "parallel.h"

namespace craso::ints
{
    template <Operator obtype, typename OperatorParams = typename libint2::operator_traits<obtype>::oper_params_type>
    std::array<RowMajorMatrix, libint2::operator_traits<obtype>::nopers>
    compute_1body_ints(const BasisSet &obs, const shellpair_list_t &shellpair_list, OperatorParams oparams)
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

    template <Operator obtype>
    std::vector<RowMajorMatrix> compute_1body_ints_deriv(unsigned deriv_order,
                                                         const BasisSet &obs,
                                                         const shellpair_list_t &shellpair_list,
                                                         const std::vector<libint2::Atom> &atoms)
    {
        using craso::parallel::nthreads;
        const auto n = obs.nbf();
        const auto nshells = obs.size();
        constexpr auto nopers = libint2::operator_traits<obtype>::nopers;
        const auto nresults =
            nopers * libint2::num_geometrical_derivatives(atoms.size(), deriv_order);
        typedef std::vector<RowMajorMatrix> result_type;
        result_type result(nresults);
        for (auto &r : result)
            r = RowMajorMatrix::Zero(n, n);

        // construct the 1-body integrals engine
        std::vector<libint2::Engine> engines(nthreads);
        engines[0] =
            libint2::Engine(obtype, obs.max_nprim(), obs.max_l(), deriv_order);
        // nuclear attraction ints engine needs to know where the charges sit ...
        // the nuclei are charges in this case; in QM/MM there will also be classical
        // charges
        if (obtype == Operator::nuclear)
        {
            std::vector<std::pair<double, std::array<double, 3>>> q;
            for (const auto &atom : atoms)
            {
                q.push_back({static_cast<double>(atom.atomic_number),
                             {{atom.x, atom.y, atom.z}}});
            }
            engines[0].set_params(q);
        }
        for (size_t i = 1; i != nthreads; ++i)
        {
            engines[i] = engines[0];
        }

        auto shell2bf = obs.shell2bf();
        auto shell2atom = obs.shell2atom(atoms);

        const auto natoms = atoms.size();
        const auto two_times_ncoords = 6 * natoms;
        const auto nderivcenters_shset =
            2 + ((obtype == Operator::nuclear) ? natoms : 0);

        auto compute = [&](int thread_id) {
            const auto &buf = engines[thread_id].results();

            // loop over unique shell pairs, {s1,s2} such that s1 >= s2
            // this is due to the permutational symmetry of the real integrals over
            // Hermitian operators: (1|2) = (2|1)
            for (auto s1 = 0l, s12 = 0l; s1 != nshells; ++s1)
            {
                auto bf1 = shell2bf[s1]; // first basis function in this shell
                auto n1 = obs[s1].size();
                auto atom1 = shell2atom[s1];
                assert(atom1 != -1);

                auto s1_offset = s1 * (s1 + 1) / 2;
                for (auto s2 : shellpair_list.at(s1))
                {
                    auto s12 = s1_offset + s2;
                    if (s12 % nthreads != thread_id)
                        continue;

                    auto bf2 = shell2bf[s2];
                    auto n2 = obs[s2].size();
                    auto atom2 = shell2atom[s2];

                    auto n12 = n1 * n2;

                    // compute shell pair; return is the pointer to the buffer
                    engines[thread_id].compute(obs[s1], obs[s2]);

                    // "copy" lambda copies shell set \c idx to the operator matrix with
                    // index \c op
                    auto add_shellset_to_dest = [&](std::size_t op, std::size_t idx,
                                                    double scale = 1.0) {
                        // "map" buffer to a const Eigen Matrix, and copy it to the
                        // corresponding blocks of the result
                        Eigen::Map<const RowMajorMatrix> buf_mat(buf[idx], n1, n2);
                        if (scale == 1.0)
                            result[op].block(bf1, bf2, n1, n2) += buf_mat;
                        else
                            result[op].block(bf1, bf2, n1, n2) += scale * buf_mat;
                        if (s1 != s2)
                        { // if s1 >= s2, copy {s1,s2} to the corresponding
                            // {s2,s1} block, note the transpose!
                            if (scale == 1.0)
                                result[op].block(bf2, bf1, n2, n1) += buf_mat.transpose();
                            else
                                result[op].block(bf2, bf1, n2, n1) += scale * buf_mat.transpose();
                        }
                    };

                    switch (deriv_order)
                    {
                    case 0:
                        for (std::size_t op = 0; op != nopers; ++op)
                        {
                            add_shellset_to_dest(op, op);
                        }
                        break;

                        // map deriv quanta for this shell pair to the overall deriv quanta
                        //
                        // easiest to explain with example:
                        // in sto-3g water shells 0 1 2 sit on atom 0, shells 3 and 4 on atoms
                        // 1 and 2 respectively
                        // each call to engine::compute for nuclear ints will return
                        // derivatives
                        // with respect to 15 coordinates, obtained as 3 (x,y,z) times 2 + 3 =
                        // 5 centers
                        // (2 centers on which shells sit + 3 nuclear charges)
                        // (for overlap, kinetic, and emultipole ints we there are only 6
                        // coordinates
                        //  since the operator is coordinate-independent, or derivatives with
                        //  respect to
                        //  the operator coordinates are not computed)
                        //

                    case 1:
                    {
                        std::size_t shellset_idx = 0;
                        for (auto c = 0; c != nderivcenters_shset; ++c)
                        {
                            auto atom = (c == 0) ? atom1 : ((c == 1) ? atom2 : c - 2);
                            auto op_start = 3 * atom * nopers;
                            auto op_fence = op_start + nopers;
                            for (auto xyz = 0; xyz != 3;
                                 ++xyz, op_start += nopers, op_fence += nopers)
                            {
                                for (unsigned int op = op_start; op != op_fence;
                                     ++op, ++shellset_idx)
                                {
                                    add_shellset_to_dest(op, shellset_idx);
                                }
                            }
                        }
                    }
                    break;

                    case 2:
                    {
                        //
                        // must pay attention to symmetry when computing 2nd and higher-order derivs
                        // e.g. d2 (s1|s2) / dX dY involves several cases:
                        // 1. only s1 (or only s2) depends on X AND Y (i.e. X and Y refer to same atom) =>
                        //    d2 (s1|s2) / dX dY = (d2 s1 / dX dY | s2)
                        // 2. s1 depends on X only, s2 depends on Y only (or vice versa) =>
                        //    d2 (s1|s2) / dX dY = (d s1 / dX | d s2 / dY)
                        // 3. s1 AND s2 depend on X AND Y (i.e. X and Y refer to same atom) =>
                        //    case A: X != Y
                        //    d2 (s1|s2) / dX dY = (d2 s1 / dX dY | s2) + (d s1 / dX | d s2 / dY)
                        //      + (d s1 / dY | d s2 / dX) + (s1| d2 s2 / dX dY )
                        //    case B: X == Y
                        //    d2 (s1|s2) / dX2 = (d2 s1 / dX2 | s2) + 2 (d s1 / dX | d s2 / dX)
                        //      + (s1| d2 s2 / dX2 )

                        // computes upper triangle index
                        // n2 = matrix size times 2
                        // i,j = (unordered) indices
#define upper_triangle_index(n2, i, j)                             \
    (std::min((i), (j))) * ((n2) - (std::min((i), (j))) - 1) / 2 + \
        (std::max((i), (j)))

                        // look over shellsets in the order in which they appear
                        std::size_t shellset_idx = 0;
                        for (auto c1 = 0; c1 != nderivcenters_shset; ++c1)
                        {
                            auto a1 = (c1 == 0) ? atom1 : ((c1 == 1) ? atom2 : c1 - 2);
                            auto coord1 = 3 * a1;
                            for (auto xyz1 = 0; xyz1 != 3; ++xyz1, ++coord1)
                            {

                                for (auto c2 = c1; c2 != nderivcenters_shset; ++c2)
                                {
                                    auto a2 = (c2 == 0) ? atom1 : ((c2 == 1) ? atom2 : c2 - 2);
                                    auto xyz2_start = (c1 == c2) ? xyz1 : 0;
                                    auto coord2 = 3 * a2 + xyz2_start;
                                    for (auto xyz2 = xyz2_start; xyz2 != 3;
                                         ++xyz2, ++coord2)
                                    {

                                        double scale = (coord1 == coord2 && c1 != c2) ? 2.0 : 1.0;

                                        const auto coord12 =
                                            upper_triangle_index(two_times_ncoords, coord1, coord2);
                                        auto op_start = coord12 * nopers;
                                        auto op_fence = op_start + nopers;
                                        for (auto op = op_start; op != op_fence;
                                             ++op, ++shellset_idx)
                                        {
                                            add_shellset_to_dest(op, shellset_idx, scale);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    break;
#undef upper_triangle_index

                    default:
                    {
                        assert(false && "not yet implemented");

                        using ShellSetDerivIterator =
                            libint2::FixedOrderedIntegerPartitionIterator<
                                std::vector<unsigned int>>;
                        ShellSetDerivIterator shellset_diter(deriv_order,
                                                             nderivcenters_shset);
                        while (shellset_diter)
                        {
                            const auto &deriv = *shellset_diter;
                        }
                    }
                    } // copy shell block switch

                } // s2 <= s1
            }     // s1
        };        // compute lambda

        craso::parallel::parallel_do(compute);

        return result;
    }

    template <libint2::Operator Kernel>
    RowMajorMatrix compute_schwarz_ints(
        const BasisSet &bs1, const BasisSet &_bs2, bool use_2norm,
        typename libint2::operator_traits<Kernel>::oper_params_type params)
    {
        const BasisSet &bs2 = (_bs2.empty() ? bs1 : _bs2);
        const auto nsh1 = bs1.size();
        const auto nsh2 = bs2.size();
        const auto bs1_equiv_bs2 = (&bs1 == &bs2);

        RowMajorMatrix K = RowMajorMatrix::Zero(nsh1, nsh2);

        // construct the 2-electron repulsion integrals engine
        using craso::parallel::nthreads;
        using libint2::Engine;
        using libint2::BraKet;
        
        std::vector<Engine> engines(nthreads);

        // !!! very important: cannot screen primitives in Schwarz computation !!!
        auto epsilon = 0.;
        engines[0] = Engine(Kernel, std::max(bs1.max_nprim(), bs2.max_nprim()),
                            std::max(bs1.max_l(), bs2.max_l()), 0, epsilon, params);
        for (size_t i = 1; i != nthreads; ++i)
        {
            engines[i] = engines[0];
        }

        auto compute = [&](int thread_id) {
            const auto &buf = engines[thread_id].results();

            // loop over permutationally-unique set of shells
            for (auto s1 = 0l, s12 = 0l; s1 != nsh1; ++s1)
            {
                auto n1 = bs1[s1].size(); // number of basis functions in this shell

                auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
                for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12)
                {
                    if (s12 % nthreads != thread_id)
                        continue;

                    auto n2 = bs2[s2].size();
                    auto n12 = n1 * n2;

                    engines[thread_id].compute2<Kernel, BraKet::xx_xx, 0>(bs1[s1], bs2[s2],
                                                                          bs1[s1], bs2[s2]);
                    assert(buf[0] != nullptr &&
                           "to compute Schwarz ints turn off primitive screening");

                    // to apply Schwarz inequality to individual integrals must use the diagonal elements
                    // to apply it to sets of functions (e.g. shells) use the whole shell-set of ints here
                    Eigen::Map<const RowMajorMatrix> buf_mat(buf[0], n12, n12);
                    auto norm2 = use_2norm ? buf_mat.norm()
                                           : buf_mat.lpNorm<Eigen::Infinity>();
                    K(s1, s2) = std::sqrt(norm2);
                    if (bs1_equiv_bs2)
                        K(s2, s1) = K(s1, s2);
                }
            }
        }; // thread lambda

        craso::parallel::parallel_do(compute);

        return K;
    }

} // namespace craso::ints