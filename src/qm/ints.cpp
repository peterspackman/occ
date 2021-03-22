#include <tonto/core/logger.h>
#include <tonto/qm/ints.h>
#include <tonto/qm/spinorbital.h>

namespace tonto::ints {

MatRM compute_2body_2index_ints(const BasisSet &bs) {
    tonto::timing::start(tonto::timing::category::ints2e);
  using tonto::parallel::nthreads;
  using libint2::BraKet;
  using libint2::Engine;
  using libint2::Shell;

  const auto n = bs.nbf();
  const auto nshells = bs.size();
  MatRM result = MatRM::Zero(n, n);

  // build engines for each thread

  std::vector<Engine> engines(nthreads);
  engines[0] =
      Engine(libint2::Operator::coulomb, bs.max_nprim(), bs.max_l(), 0);
  engines[0].set(BraKet::xs_xs);
  for (size_t i = 1; i != nthreads; ++i) {
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
    for (size_t s1 = 0, s12 = 0; s1 != nshells; ++s1) {
      auto bf1 = shell2bf[s1]; // first basis function in this shell
      auto n1 = bs[s1].size();

      for (size_t s2 = 0; s2 <= s1; ++s2, ++s12) {
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
        Eigen::Map<const MatRM> buf_mat(buf[0], n1, n2);
        result.block(bf1, bf2, n1, n2) = buf_mat;
        if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding {s2,s1}
                      // block, note the transpose!
          result.block(bf2, bf1, n2, n1) = buf_mat.transpose();
      }
    }
  }; // compute lambda

  tonto::parallel::parallel_do(compute);
  tonto::timing::stop(tonto::timing::category::ints2e);

  return result;
}

std::tuple<shellpair_list_t, shellpair_data_t>
compute_shellpairs(const BasisSet &bs1, const BasisSet &_bs2,
                   const double threshold) {
    tonto::timing::start(tonto::timing::category::ints1e);
  using libint2::Operator;
  const BasisSet &bs2 = (_bs2.empty() ? bs1 : _bs2);
  const auto nsh1 = bs1.size();
  const auto nsh2 = bs2.size();
  const auto bs1_equiv_bs2 = (&bs1 == &bs2);

  using tonto::parallel::nthreads;

  // construct the 2-electron repulsion integrals engine
  using libint2::Engine;
  std::vector<Engine> engines;
  engines.reserve(nthreads);
  engines.emplace_back(Operator::overlap,
                       std::max(bs1.max_nprim(), bs2.max_nprim()),
                       std::max(bs1.max_l(), bs2.max_l()), 0);
  for (size_t i = 1; i != nthreads; ++i) {
    engines.push_back(engines[0]);
  }

  tonto::log::debug("start computing non-negligible shell-pair list");

  libint2::Timers<1> timer;
  timer.set_now_overhead(25);
  timer.start(0);

  shellpair_list_t splist;

  std::mutex mx;

  auto compute = [&](int thread_id) {
    auto &engine = engines[thread_id];
    const auto &buf = engine.results();

    // loop over permutationally-unique set of shells
    for (size_t s1 = 0, s12 = 0; s1 != nsh1; ++s1) {
      mx.lock();
      if (splist.find(s1) == splist.end())
        splist.insert(std::make_pair(s1, std::vector<size_t>()));
      mx.unlock();

      auto n1 = bs1[s1].size(); // number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
      for (size_t s2 = 0; s2 <= s2_max; ++s2, ++s12) {
        if (s12 % nthreads != thread_id)
          continue;

        auto on_same_center = (bs1[s1].O == bs2[s2].O);
        bool significant = on_same_center;
        if (not on_same_center) {
          auto n2 = bs2[s2].size();
          engines[thread_id].compute(bs1[s1], bs2[s2]);
          Eigen::Map<const MatRM> buf_mat(buf[0], n1, n2);
          auto norm = buf_mat.norm();
          significant = (norm >= threshold);
        }

        if (significant) {
          mx.lock();
          splist[s1].emplace_back(s2);
          mx.unlock();
        }
      }
    }
  }; // end of compute

  tonto::parallel::parallel_do(compute);

  // resort shell list in increasing order, i.e. splist[s][s1] < splist[s][s2]
  // if s1 < s2 N.B. only parallelized over 1 shell index
  auto sort = [&](int thread_id) {
    for (auto s1 = 0l; s1 != nsh1; ++s1) {
      if (s1 % nthreads == thread_id) {
        auto &list = splist[s1];
        std::sort(list.begin(), list.end());
      }
    }
  }; // end of sort

  tonto::parallel::parallel_do(sort);

  // compute shellpair data assuming that we are computing to default_epsilon
  // N.B. only parallelized over 1 shell index
  const auto ln_max_engine_precision = std::log(max_engine_precision);
  shellpair_data_t spdata(splist.size());
  auto make_spdata = [&](int thread_id) {
    for (auto s1 = 0l; s1 != nsh1; ++s1) {
      if (s1 % nthreads == thread_id) {
        for (const auto &s2 : splist[s1]) {
          spdata[s1].emplace_back(std::make_shared<libint2::ShellPair>(
              bs1[s1], bs2[s2], ln_max_engine_precision));
        }
      }
    }
  }; // end of make_spdata

  tonto::parallel::parallel_do(make_spdata);

  timer.stop(0);
  tonto::log::debug("computed non-negligible shell-pair list in {:.6f} s", timer.read(0));
  tonto::timing::stop(tonto::timing::category::ints1e);

  return std::make_tuple(splist, spdata);
}

MatRM compute_shellblock_norm(const BasisSet &obs, const MatRM &A) {
  const auto nsh = obs.size();
  MatRM Ash(nsh, nsh);

  auto shell2bf = obs.shell2bf();
  for (size_t s1 = 0; s1 != nsh; ++s1) {
    const auto &s1_first = shell2bf[s1];
    const auto &s1_size = obs[s1].size();
    for (size_t s2 = 0; s2 != nsh; ++s2) {
      const auto &s2_first = shell2bf[s2];
      const auto &s2_size = obs[s2].size();

      Ash(s1, s2) = A.block(s1_first, s2_first, s1_size, s2_size)
                        .lpNorm<Eigen::Infinity>();
    }
  }

  return Ash;
}

MatRM compute_2body_fock_mixed_basis(const BasisSet &obs, const MatRM &D,
                                 const BasisSet &D_bs, bool D_is_shelldiagonal,
                                 double precision) {
    tonto::timing::start(tonto::timing::category::ints2e);

  const auto n = obs.nbf();
  const auto nshells = obs.size();
  const auto n_D = D_bs.nbf();
  assert(D.cols() == D.rows() && D.cols() == n_D);

  using tonto::parallel::nthreads;
  std::vector<MatRM> G(nthreads, MatRM::Zero(n, n));

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
  for (size_t i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }
  auto shell2bf = obs.shell2bf();
  auto shell2bf_D = D_bs.shell2bf();

  auto lambda = [&](int thread_id) {
    auto &engine = engines[thread_id];
    auto &g = G[thread_id];
    const auto &buf = engine.results();

    // loop over permutationally-unique set of shells
    for (auto s1 = 0l, s1234 = 0l; s1 != nshells; ++s1) {
      auto bf1_first = shell2bf[s1]; // first basis function in this shell
      auto n1 = obs[s1].size();      // number of basis functions in this shell

      for (auto s2 = 0; s2 <= s1; ++s2) {
        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        for (auto s3 = 0; s3 < D_bs.size(); ++s3) {
          auto bf3_first = shell2bf_D[s3];
          auto n3 = D_bs[s3].size();

          auto s4_begin = D_is_shelldiagonal ? s3 : 0;
          auto s4_fence = D_is_shelldiagonal ? s3 + 1 : D_bs.size();

          for (auto s4 = s4_begin; s4 != s4_fence; ++s4, ++s1234) {
            if (s1234 % nthreads != thread_id)
              continue;

            auto bf4_first = shell2bf_D[s4];
            auto n4 = D_bs[s4].size();

            // compute the permutational degeneracy (i.e. # of equivalents) of
            // the given shell set
            auto s12_deg = (s1 == s2) ? 1.0 : 2.0;

            if (s3 >= s4) {
              auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
              auto s1234_deg = s12_deg * s34_deg;
              // auto s1234_deg = s12_deg;
              engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                  obs[s1], obs[s2], D_bs[s3], D_bs[s4]);
              const auto *buf_1234 = buf[0];
              if (buf_1234 != nullptr) {
                for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                  const auto bf1 = f1 + bf1_first;
                  for (auto f2 = 0; f2 != n2; ++f2) {
                    const auto bf2 = f2 + bf2_first;
                    for (auto f3 = 0; f3 != n3; ++f3) {
                      const auto bf3 = f3 + bf3_first;
                      for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
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

            for (auto f1 = 0, f1324 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (auto f3 = 0; f3 != n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for (auto f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (auto f4 = 0; f4 != n4; ++f4, ++f1324) {
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

  tonto::parallel::parallel_do(lambda);

  // accumulate contributions from all threads
  for (size_t i = 1; i != nthreads; ++i) {
    G[0] += G[i];
  }
  tonto::timing::stop(tonto::timing::category::ints2e);

  // symmetrize the result and return
  return 0.5 * (G[0] + G[0].transpose());
}

tonto::Vec compute_electric_potential(const tonto::MatRM &D, const BasisSet &obs,
                                      const shellpair_list_t &shellpair_list, const tonto::Mat3N &positions)
{
    tonto::timing::start(tonto::timing::category::ints1e);

    using tonto::qm::expectation;
    const auto n = obs.nbf();
    const bool unrestricted = D.rows() > n;
    const auto nshells = obs.size();
    using tonto::parallel::nthreads;
    typedef std::array<MatRM, libint2::operator_traits<Operator::nuclear>::nopers>
            result_type;
    const unsigned int nopers = libint2::operator_traits<Operator::nuclear>::nopers;

    tonto::Vec result(positions.cols());
    std::vector<libint2::Engine> engines(nthreads);
    engines[0] = libint2::Engine(Operator::nuclear, obs.max_nprim(), obs.max_l(), 0);
    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }
    std::vector<MatRM> opmats(nthreads, MatRM::Zero(n, n));
    auto shell2bf = obs.shell2bf();

    auto compute = [&](int thread_id) {
        for(size_t pt = 0; pt < positions.cols(); pt++) {
            if (pt % nthreads != thread_id) continue;
            opmats[thread_id].setZero();
            std::vector<std::pair<double, std::array<double, 3>>> chgs {
                {1, {positions(0, pt), positions(1, pt), positions(2, pt)}}
            };
            engines[thread_id].set_params(chgs);

            const auto &buf = engines[thread_id].results();

            // loop over unique shell pairs, {s1,s2} such that s1 >= s2
            // this is due to the permutational symmetry of the real integrals over
            // Hermitian operators: (1|2) = (2|1)
            for (size_t s1 = 0l, s12 = 0l; s1 != nshells; ++s1) {
                size_t bf1 = shell2bf[s1]; // first basis function in this shell
                size_t n1 = obs[s1].size();

                size_t s1_offset = s1 * (s1 + 1) / 2;
                for (size_t s2 : shellpair_list.at(s1)) {
                    size_t s12 = s1_offset + s2;
                    size_t bf2 = shell2bf[s2];
                    size_t n2 = obs[s2].size();

                    // compute shell pair; return is the pointer to the buffer
                    engines[thread_id].compute(obs[s1], obs[s2]);

                    Eigen::Map<const MatRM> buf_mat(buf[0], n1, n2);
                    opmats[thread_id].block(bf1, bf2, n1, n2) = buf_mat;
                    if (s1 != s2) // if s1 >= s2, copy {s1,s2} to the corresponding
                        // {s2,s1} block, note the transpose!
                        opmats[thread_id].block(bf2, bf1, n2, n1) = buf_mat.transpose();
                }
            }
            if(unrestricted) result(pt) = 2 * expectation<tonto::qm::SpinorbitalKind::Unrestricted>(D, opmats[thread_id]);
            else result(pt) = 2 * expectation<tonto::qm::SpinorbitalKind::Restricted>(D, opmats[thread_id]);
        }
    }; // compute lambda
    tonto::parallel::parallel_do(compute);
    tonto::timing::stop(tonto::timing::category::ints1e);
    return result;
}

tonto::Mat3N compute_electric_field(const tonto::MatRM &D, const BasisSet &obs,
                                    const shellpair_list_t &shellpair_list, const tonto::Mat3N &positions)
{
    tonto::timing::start(tonto::timing::category::ints1e);
    using tonto::parallel::nthreads;
    using tonto::qm::expectation;
    // seems to be correct for very small molecules but has issues -- use finite differences for now

    const auto n = obs.nbf();
    const auto nshells = obs.size();
    const auto nresults = libint2::num_geometrical_derivatives(1, 1);
    const bool unrestricted = D.rows() > n;
    tonto::Mat3N result(3, positions.cols());

    std::vector<MatRM> xmats(nthreads, MatRM::Zero(n, n));
    std::vector<MatRM> ymats(nthreads, MatRM::Zero(n, n));
    std::vector<MatRM> zmats(nthreads, MatRM::Zero(n, n));

    // construct the 1-body integrals engine
    std::vector<libint2::Engine> engines(nthreads);
    engines[0] = libint2::Engine(Operator::nuclear, obs.max_nprim(), obs.max_l(), 1);

    for (size_t i = 1; i != nthreads; ++i) {
        engines[i] = engines[0];
    }
    auto shell2bf = obs.shell2bf();

    auto compute = [&](int thread_id) {
        for(size_t pt = 0; pt < positions.cols(); pt++) {
            if (pt % nthreads != thread_id) continue;
            std::vector<std::pair<double, std::array<double, 3>>> chgs {
                {1, {positions(0, pt), positions(1, pt), positions(2, pt)}}
            };
            engines[thread_id].set_params(chgs);
            const auto &buf = engines[thread_id].results();
            for (size_t s1 = 0l, s12 = 0l; s1 != nshells; ++s1)
            {
                size_t bf1 = shell2bf[s1]; // first basis function in this shell
                size_t n1 = obs[s1].size();
                size_t s1_offset = s1 * (s1 + 1) / 2;
                for (size_t s2 : shellpair_list.at(s1))
                {
                    size_t s12 = s1_offset + s2;
                    size_t bf2 = shell2bf[s2];
                    size_t n2 = obs[s2].size();
                    engines[thread_id].compute(obs[s1], obs[s2]);

                    Eigen::Map<const MatRM> buf_mat_x(buf[0], n1, n2), buf_mat_y(buf[1], n1, n2), buf_mat_z(buf[2], n1, n2);
                    Eigen::Map<const MatRM> buf_mat_x2(buf[3], n1, n2), buf_mat_y2(buf[4], n1, n2), buf_mat_z2(buf[5], n1, n2);
                    xmats[thread_id].block(bf1, bf2, n1, n2) = - (buf_mat_x + buf_mat_x2);
                    ymats[thread_id].block(bf1, bf2, n1, n2) = - (buf_mat_y + buf_mat_y2);
                    zmats[thread_id].block(bf1, bf2, n1, n2) = - (buf_mat_z + buf_mat_z2);
                    if (s1 != s2) {
                        xmats[thread_id].block(bf2, bf1, n2, n1) = - (buf_mat_x + buf_mat_x2).transpose();
                        ymats[thread_id].block(bf2, bf1, n2, n1) = - (buf_mat_y + buf_mat_y2).transpose();
                        zmats[thread_id].block(bf2, bf1, n2, n1) = - (buf_mat_z + buf_mat_z2).transpose();
                    }
                }

            }
            if (unrestricted) {
                result(0, pt) = - 2 * expectation<tonto::qm::SpinorbitalKind::Unrestricted>(D, xmats[thread_id]);
                result(1, pt) = - 2 * expectation<tonto::qm::SpinorbitalKind::Unrestricted>(D, ymats[thread_id]);
                result(2, pt) = - 2 * expectation<tonto::qm::SpinorbitalKind::Unrestricted>(D, zmats[thread_id]);
            }
            else {
                result(0, pt) = - 2 * expectation<tonto::qm::SpinorbitalKind::Restricted>(D, xmats[thread_id]);
                result(1, pt) = - 2 * expectation<tonto::qm::SpinorbitalKind::Restricted>(D, ymats[thread_id]);
                result(2, pt) = - 2 * expectation<tonto::qm::SpinorbitalKind::Restricted>(D, zmats[thread_id]);
            }
        }
    };
    tonto::parallel::parallel_do(compute);
    tonto::timing::stop(tonto::timing::category::ints1e);
    return result;
}

} // namespace tonto::ints
