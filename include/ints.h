#pragma once
#include "linear_algebra.h"
#include "parallel.h"
#include <array>
#include <libint2/basis.h>
#include <libint2/engine.h>
#include <libint2/shell.h>
#include <unordered_map>
#include <vector>

namespace craso::ints {
using craso::MatRM;
using libint2::BasisSet;
using libint2::BraKet;
using libint2::Operator;
using libint2::Shell;

using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
using shellpair_data_t = std::vector<std::vector<
    std::shared_ptr<libint2::ShellPair>>>; // in same order as shellpair_list_t
const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;

template <Operator obtype,
          typename OperatorParams =
              typename libint2::operator_traits<obtype>::oper_params_type>
std::array<MatRM, libint2::operator_traits<obtype>::nopers>
compute_1body_ints(const BasisSet &obs, const shellpair_list_t &shellpair_list,
                   OperatorParams oparams = OperatorParams()) {
  const auto n = obs.nbf();
  const auto nshells = obs.size();
  using craso::parallel::nthreads;
  typedef std::array<MatRM, libint2::operator_traits<obtype>::nopers>
      result_type;
  const unsigned int nopers = libint2::operator_traits<obtype>::nopers;
  result_type result;
  for (auto &r : result)
    r = MatRM::Zero(n, n);

  // construct the 1-body integrals engine
  std::vector<libint2::Engine> engines(nthreads);
  engines[0] = libint2::Engine(obtype, obs.max_nprim(), obs.max_l(), 0);
  // pass operator params to the engine, e.g.
  // nuclear attraction ints engine needs to know where the charges sit ...
  // the nuclei are charges in this case; in QM/MM there will also be classical
  // charges
  engines[0].set_params(oparams);
  for (size_t i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }

  auto shell2bf = obs.shell2bf();

  auto compute = [&](int thread_id) {
    const auto &buf = engines[thread_id].results();

    // loop over unique shell pairs, {s1,s2} such that s1 >= s2
    // this is due to the permutational symmetry of the real integrals over
    // Hermitian operators: (1|2) = (2|1)
    for (auto s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
      auto bf1 = shell2bf[s1]; // first basis function in this shell
      auto n1 = obs[s1].size();

      auto s1_offset = s1 * (s1 + 1) / 2;
      for (auto s2 : shellpair_list.at(s1)) {
        auto s12 = s1_offset + s2;
        if (s12 % nthreads != thread_id)
          continue;

        auto bf2 = shell2bf[s2];
        auto n2 = obs[s2].size();

        auto n12 = n1 * n2;

        // compute shell pair; return is the pointer to the buffer
        engines[thread_id].compute(obs[s1], obs[s2]);

        for (unsigned int op = 0; op != nopers; ++op) {
          // "map" buffer to a const Eigen MatRM, and copy it to the
          // corresponding blocks of the result
          Eigen::Map<const MatRM> buf_mat(buf[op], n1, n2);
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
std::vector<MatRM>
compute_1body_ints_deriv(unsigned deriv_order, const BasisSet &obs,
                         const shellpair_list_t &shellpair_list,
                         const std::vector<libint2::Atom> &atoms) {
  using craso::parallel::nthreads;
  const auto n = obs.nbf();
  const auto nshells = obs.size();
  constexpr auto nopers = libint2::operator_traits<obtype>::nopers;
  const auto nresults =
      nopers * libint2::num_geometrical_derivatives(atoms.size(), deriv_order);
  typedef std::vector<MatRM> result_type;
  result_type result(nresults);
  for (auto &r : result)
    r = MatRM::Zero(n, n);

  // construct the 1-body integrals engine
  std::vector<libint2::Engine> engines(nthreads);
  engines[0] =
      libint2::Engine(obtype, obs.max_nprim(), obs.max_l(), deriv_order);
  // nuclear attraction ints engine needs to know where the charges sit ...
  // the nuclei are charges in this case; in QM/MM there will also be classical
  // charges
  if (obtype == Operator::nuclear) {
    std::vector<std::pair<double, std::array<double, 3>>> q;
    for (const auto &atom : atoms) {
      q.push_back({static_cast<double>(atom.atomic_number),
                   {{atom.x, atom.y, atom.z}}});
    }
    engines[0].set_params(q);
  }
  for (size_t i = 1; i != nthreads; ++i) {
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
    for (auto s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
      auto bf1 = shell2bf[s1]; // first basis function in this shell
      auto n1 = obs[s1].size();
      auto atom1 = shell2atom[s1];
      assert(atom1 != -1);

      auto s1_offset = s1 * (s1 + 1) / 2;
      for (auto s2 : shellpair_list.at(s1)) {
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
          Eigen::Map<const MatRM> buf_mat(buf[idx], n1, n2);
          if (scale == 1.0)
            result[op].block(bf1, bf2, n1, n2) += buf_mat;
          else
            result[op].block(bf1, bf2, n1, n2) += scale * buf_mat;
          if (s1 != s2) { // if s1 >= s2, copy {s1,s2} to the corresponding
            // {s2,s1} block, note the transpose!
            if (scale == 1.0)
              result[op].block(bf2, bf1, n2, n1) += buf_mat.transpose();
            else
              result[op].block(bf2, bf1, n2, n1) += scale * buf_mat.transpose();
          }
        };

        switch (deriv_order) {
        case 0:
          for (std::size_t op = 0; op != nopers; ++op) {
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

        case 1: {
          std::size_t shellset_idx = 0;
          for (auto c = 0; c != nderivcenters_shset; ++c) {
            auto atom = (c == 0) ? atom1 : ((c == 1) ? atom2 : c - 2);
            auto op_start = 3 * atom * nopers;
            auto op_fence = op_start + nopers;
            for (auto xyz = 0; xyz != 3;
                 ++xyz, op_start += nopers, op_fence += nopers) {
              for (unsigned int op = op_start; op != op_fence;
                   ++op, ++shellset_idx) {
                add_shellset_to_dest(op, shellset_idx);
              }
            }
          }
        } break;

        case 2: {
          //
          // must pay attention to symmetry when computing 2nd and higher-order
          // derivs e.g. d2 (s1|s2) / dX dY involves several cases:
          // 1. only s1 (or only s2) depends on X AND Y (i.e. X and Y refer to
          // same atom) =>
          //    d2 (s1|s2) / dX dY = (d2 s1 / dX dY | s2)
          // 2. s1 depends on X only, s2 depends on Y only (or vice versa) =>
          //    d2 (s1|s2) / dX dY = (d s1 / dX | d s2 / dY)
          // 3. s1 AND s2 depend on X AND Y (i.e. X and Y refer to same atom) =>
          //    case A: X != Y
          //    d2 (s1|s2) / dX dY = (d2 s1 / dX dY | s2) + (d s1 / dX | d s2 /
          //    dY)
          //      + (d s1 / dY | d s2 / dX) + (s1| d2 s2 / dX dY )
          //    case B: X == Y
          //    d2 (s1|s2) / dX2 = (d2 s1 / dX2 | s2) + 2 (d s1 / dX | d s2 /
          //    dX)
          //      + (s1| d2 s2 / dX2 )

          // computes upper triangle index
          // n2 = matrix size times 2
          // i,j = (unordered) indices
#define upper_triangle_index(n2, i, j)                                         \
  (std::min((i), (j))) * ((n2) - (std::min((i), (j))) - 1) / 2 +               \
      (std::max((i), (j)))

          // look over shellsets in the order in which they appear
          std::size_t shellset_idx = 0;
          for (auto c1 = 0; c1 != nderivcenters_shset; ++c1) {
            auto a1 = (c1 == 0) ? atom1 : ((c1 == 1) ? atom2 : c1 - 2);
            auto coord1 = 3 * a1;
            for (auto xyz1 = 0; xyz1 != 3; ++xyz1, ++coord1) {

              for (auto c2 = c1; c2 != nderivcenters_shset; ++c2) {
                auto a2 = (c2 == 0) ? atom1 : ((c2 == 1) ? atom2 : c2 - 2);
                auto xyz2_start = (c1 == c2) ? xyz1 : 0;
                auto coord2 = 3 * a2 + xyz2_start;
                for (auto xyz2 = xyz2_start; xyz2 != 3; ++xyz2, ++coord2) {

                  double scale = (coord1 == coord2 && c1 != c2) ? 2.0 : 1.0;

                  const auto coord12 =
                      upper_triangle_index(two_times_ncoords, coord1, coord2);
                  auto op_start = coord12 * nopers;
                  auto op_fence = op_start + nopers;
                  for (auto op = op_start; op != op_fence;
                       ++op, ++shellset_idx) {
                    add_shellset_to_dest(op, shellset_idx, scale);
                  }
                }
              }
            }
          }
        } break;
#undef upper_triangle_index

        default: {
          assert(false && "not yet implemented");

          using ShellSetDerivIterator =
              libint2::FixedOrderedIntegerPartitionIterator<
                  std::vector<unsigned int>>;
          ShellSetDerivIterator shellset_diter(deriv_order,
                                               nderivcenters_shset);
          while (shellset_diter) {
            const auto &deriv = *shellset_diter;
          }
        }
        } // copy shell block switch

      } // s2 <= s1
    }   // s1
  };    // compute lambda

  craso::parallel::parallel_do(compute);

  return result;
}

template <libint2::Operator Kernel = libint2::Operator::coulomb>
MatRM compute_schwarz_ints(
    const BasisSet &bs1, const BasisSet &_bs2 = BasisSet(),
    bool use_2norm = false, // use infty norm by default
    typename libint2::operator_traits<Kernel>::oper_params_type params =
        libint2::operator_traits<Kernel>::default_params())

{
  const BasisSet &bs2 = (_bs2.empty() ? bs1 : _bs2);
  const auto nsh1 = bs1.size();
  const auto nsh2 = bs2.size();
  const auto bs1_equiv_bs2 = (&bs1 == &bs2);

  MatRM K = MatRM::Zero(nsh1, nsh2);

  // construct the 2-electron repulsion integrals engine
  using craso::parallel::nthreads;
  using libint2::BraKet;
  using libint2::Engine;

  std::vector<Engine> engines(nthreads);

  // !!! very important: cannot screen primitives in Schwarz computation !!!
  auto epsilon = 0.;
  engines[0] = Engine(Kernel, std::max(bs1.max_nprim(), bs2.max_nprim()),
                      std::max(bs1.max_l(), bs2.max_l()), 0, epsilon, params);
  for (size_t i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }

  auto compute = [&](int thread_id) {
    const auto &buf = engines[thread_id].results();

    // loop over permutationally-unique set of shells
    for (auto s1 = 0l, s12 = 0l; s1 != nsh1; ++s1) {
      auto n1 = bs1[s1].size(); // number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
      for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12) {
        if (s12 % nthreads != thread_id)
          continue;

        auto n2 = bs2[s2].size();
        auto n12 = n1 * n2;

        engines[thread_id].compute2<Kernel, BraKet::xx_xx, 0>(bs1[s1], bs2[s2],
                                                              bs1[s1], bs2[s2]);
        assert(buf[0] != nullptr &&
               "to compute Schwarz ints turn off primitive screening");

        // to apply Schwarz inequality to individual integrals must use the
        // diagonal elements to apply it to sets of functions (e.g. shells) use
        // the whole shell-set of ints here
        Eigen::Map<const MatRM> buf_mat(buf[0], n12, n12);
        auto norm2 =
            use_2norm ? buf_mat.norm() : buf_mat.lpNorm<Eigen::Infinity>();
        K(s1, s2) = std::sqrt(norm2);
        if (bs1_equiv_bs2)
          K(s2, s1) = K(s1, s2);
      }
    }
  }; // thread lambda

  craso::parallel::parallel_do(compute);

  return K;
}

MatRM compute_shellblock_norm(const BasisSet &obs, const MatRM &A);
MatRM compute_2body_2index_ints(const BasisSet &);
MatRM compute_2body_fock(
    const BasisSet &obs, const shellpair_list_t &shellpair_list,
    const shellpair_data_t &shellpair_data, const MatRM &D,
    double precision = std::numeric_limits<
        double>::epsilon(),        // discard contributions smaller than this
    const MatRM &Schwarz = MatRM() // K_ij = sqrt(||(ij|ij)||_\infty); if
                                   // empty, do not Schwarz screen
);

std::pair<MatRM, MatRM>
compute_JK(const BasisSet &obs, const shellpair_list_t &shellpair_list,
           const shellpair_data_t &shellpair_data, const MatRM &D,
           double precision = std::numeric_limits<double>::epsilon(),
           const MatRM &Schwarz = MatRM());

std::pair<MatRM, MatRM> compute_2body_fock_unrestricted(
    const BasisSet &obs, const shellpair_list_t &shellpair_list,
    const shellpair_data_t &shellpair_data, const MatRM &Da, const MatRM &Db,
    double precision = std::numeric_limits<
        double>::epsilon(),        // discard contributions smaller than this
    const MatRM &Schwarz = MatRM() // K_ij = sqrt(||(ij|ij)||_\infty); if
                                   // empty, do not Schwarz screen
);

std::tuple<MatRM, MatRM, MatRM, MatRM> compute_JK_unrestricted(
    const BasisSet &obs, const shellpair_list_t &shellpair_list,
    const shellpair_data_t &shellpair_data, const MatRM &Da, const MatRM &Db,
    double precision = std::numeric_limits<double>::epsilon(),
    const MatRM &Schwarz = MatRM());

MatRM compute_2body_fock_general(
    const BasisSet &obs, const MatRM &D, const BasisSet &D_bs,
    bool D_is_shelldiagonal,
    double precision = std::numeric_limits<double>::epsilon());

template <unsigned deriv_order>
std::vector<MatRM> compute_2body_fock_deriv(
    const BasisSet &obs, const shellpair_list_t &shellpair_list,
    const shellpair_data_t &shellpair_data,
    const std::vector<libint2::Atom> &atoms, const MatRM &D, double precision,
    const MatRM &Schwarz) {
  const auto n = obs.nbf();
  const auto nshells = obs.size();
  const auto nderiv_shellset = libint2::num_geometrical_derivatives(
      4, deriv_order); // # of derivs for each shell quartet
  const auto nderiv = libint2::num_geometrical_derivatives(
      atoms.size(), deriv_order); // total # of derivs
  const auto ncoords_times_two = (atoms.size() * 3) * 2;
  using craso::parallel::nthreads;
  std::vector<MatRM> G(nthreads * nderiv, MatRM::Zero(n, n));

  const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
  MatRM D_shblk_norm =
      compute_shellblock_norm(obs, D); // matrix of infty-norms of shell blocks

  auto fock_precision = precision;
  // engine precision controls primitive truncation, assume worst-case scenario
  // (all primitive combinations add up constructively)
  auto max_nprim = obs.max_nprim();
  auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
  auto engine_precision = std::min(fock_precision / D_shblk_norm.maxCoeff(),
                                   std::numeric_limits<double>::epsilon()) /
                          max_nprim4;

  // construct the 2-electron repulsion integrals engine pool
  using libint2::Engine;
  std::vector<Engine> engines(nthreads);
  engines[0] =
      Engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), deriv_order);
  engines[0].set_precision(engine_precision); // shellset-dependent precision
                                              // control will likely break
                                              // positive definiteness
                                              // stick with this simple recipe
  std::cout << "compute_2body_fock:precision = " << precision << std::endl;
  std::cout << "Engine::precision = " << engines[0].precision() << std::endl;
  for (size_t i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }
  std::atomic<size_t> num_ints_computed{0};

#if defined(REPORT_INTEGRAL_TIMINGS)
  std::vector<libint2::Timers<1>> timers(nthreads);
#endif

  auto shell2bf = obs.shell2bf();
  auto shell2atom = obs.shell2atom(atoms);

  auto lambda = [&](int thread_id) {
    auto &engine = engines[thread_id];
    const auto &buf = engine.results();

#if defined(REPORT_INTEGRAL_TIMINGS)
    auto &timer = timers[thread_id];
    timer.clear();
    timer.set_now_overhead(25);
#endif

    size_t shell_atoms[4];

    // loop over permutationally-unique set of shells
    for (auto s1 = 0l, s1234 = 0l; s1 != nshells; ++s1) {
      auto bf1_first = shell2bf[s1]; // first basis function in this shell
      auto n1 = obs[s1].size();      // number of basis functions in this shell
      shell_atoms[0] = shell2atom[s1];

      for (const auto &s2 : shellpair_list.at(s1)) {
        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();
        shell_atoms[1] = shell2atom[s2];

        const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

        for (auto s3 = 0; s3 <= s1; ++s3) {
          auto bf3_first = shell2bf[s3];
          auto n3 = obs[s3].size();
          shell_atoms[2] = shell2atom[s3];

          const auto Dnorm123 =
              do_schwarz_screen
                  ? std::max(D_shblk_norm(s1, s3),
                             std::max(D_shblk_norm(s2, s3), Dnorm12))
                  : 0.;

          const auto s4_max = (s1 == s3) ? s2 : s3;
          for (const auto &s4 : shellpair_list.at(s3)) {
            if (s4 > s4_max)
              break; // for each s3, s4 are stored in monotonically increasing
                     // order

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
                Dnorm1234 * Schwarz(s1, s2) * Schwarz(s3, s4) < fock_precision)
              continue;

            auto bf4_first = shell2bf[s4];
            auto n4 = obs[s4].size();
            shell_atoms[3] = shell2atom[s4];

            const auto n1234 = n1 * n2 * n3 * n4;

            // compute the permutational degeneracy (i.e. # of equivalents) of
            // the given shell set
            auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
            auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
            auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
            auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

            // computes contribution from shell set \c idx to the operator
            // matrix with index \c op
            auto add_shellset_to_dest = [&](std::size_t op, std::size_t idx,
                                            int coord1, int coord2,
                                            double scale = 1.0) {
              auto &g = G[op];
              auto shset = buf[idx];
              const auto weight = scale * s1234_deg;

              for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                const auto bf1 = f1 + bf1_first;
                for (auto f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (auto f3 = 0; f3 != n3; ++f3) {
                    const auto bf3 = f3 + bf3_first;
                    for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                      const auto bf4 = f4 + bf4_first;

                      const auto value = shset[f1234];
                      const auto wvalue = value * weight;

                      g(bf1, bf2) += D(bf3, bf4) * wvalue;
                      g(bf3, bf4) += D(bf1, bf2) * wvalue;
                      g(bf1, bf3) -= 0.25 * D(bf2, bf4) * wvalue;
                      g(bf2, bf4) -= 0.25 * D(bf1, bf3) * wvalue;
                      g(bf1, bf4) -= 0.25 * D(bf2, bf3) * wvalue;
                      g(bf2, bf3) -= 0.25 * D(bf1, bf4) * wvalue;
                    }
                  }
                }
              }
            };

#if defined(REPORT_INTEGRAL_TIMINGS)
            timer.start(0);
#endif

            engine.compute2<Operator::coulomb, BraKet::xx_xx, deriv_order>(
                obs[s1], obs[s2], obs[s3], obs[s4]);
            if (buf[0] == nullptr)
              continue; // if all integrals screened out, skip to next quartet
            num_ints_computed += nderiv_shellset * n1234;

#if defined(REPORT_INTEGRAL_TIMINGS)
            timer.stop(0);
#endif

            switch (deriv_order) {
            case 0: {
              int coord1 = 0, coord2 = 0;
              add_shellset_to_dest(thread_id, 0, coord1, coord2);
            } break;

            case 1: {
              for (auto d = 0; d != 12; ++d) {
                const int a = d / 3;
                const int xyz = d % 3;

                auto coord = shell_atoms[a] * 3 + xyz;
                auto &g = G[thread_id * nderiv + coord];

                int coord1 = 0, coord2 = 0;

                add_shellset_to_dest(thread_id * nderiv + coord, d, coord1,
                                     coord2);

              } // d \in [0,12)
            } break;

            case 2: {
// computes upper triangle index
// n2 = matrix size times 2
// i,j = (unordered) indices
#define upper_triangle_index(n2, i, j)                                         \
  (std::min((i), (j))) * ((n2) - (std::min((i), (j))) - 1) / 2 +               \
      (std::max((i), (j)))
              // look over shellsets in the order in which they appear
              std::size_t shellset_idx = 0;
              for (auto c1 = 0; c1 != 4; ++c1) {
                auto a1 = shell_atoms[c1];
                auto coord1 = 3 * a1;
                for (auto xyz1 = 0; xyz1 != 3; ++xyz1, ++coord1) {
                  for (auto c2 = c1; c2 != 4; ++c2) {
                    auto a2 = shell_atoms[c2];
                    auto xyz2_start = (c1 == c2) ? xyz1 : 0;
                    auto coord2 = 3 * a2 + xyz2_start;
                    for (auto xyz2 = xyz2_start; xyz2 != 3; ++xyz2, ++coord2) {
                      double scale = (coord1 == coord2 && c1 != c2) ? 2.0 : 1.0;

                      const auto coord12 = upper_triangle_index(
                          ncoords_times_two, coord1, coord2);
                      auto op = thread_id * nderiv + coord12;
                      add_shellset_to_dest(op, shellset_idx, coord1, coord2,
                                           scale);
                      ++shellset_idx;
                    }
                  }
                }
              }
            } break;
#undef upper_triangle_index

            default:
              assert(deriv_order <= 2 &&
                     "support for 3rd and higher derivatives of the Fock "
                     "matrix not yet implemented");
            }
          }
        }
      }
    }
  }; // end of lambda

  craso::parallel::parallel_do(lambda);

  // accumulate contributions from all threads
  for (size_t t = 1; t != nthreads; ++t) {
    for (auto d = 0; d != nderiv; ++d) {
      G[d] += G[t * nderiv + d];
    }
  }

#if defined(REPORT_INTEGRAL_TIMINGS)
  double time_for_ints = 0.0;
  for (auto &t : timers) {
    time_for_ints += t.read(0);
  }
  std::cout << "time for integrals = " << time_for_ints << std::endl;
  for (int t = 0; t != nthreads; ++t)
    engines[t].print_timers();
#endif

  std::vector<MatRM> GG(nderiv);
  for (auto d = 0; d != nderiv; ++d) {
    GG[d] = 0.5 * (G[d] + G[d].transpose());
  }

  std::cout << "# of integrals = " << num_ints_computed << std::endl;

  // symmetrize the result and return
  return GG;
}
} // namespace craso::ints
