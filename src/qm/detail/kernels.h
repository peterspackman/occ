#pragma once
#include "kernel_traits.h"
#include <occ/core/timings.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm::detail {

using ShellList = std::vector<Shell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = Shell::Kind;
using Op = cint::Operator;

template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center(Lambda &f, cint::IntegralEnvironment &env,
                         const AOBasis &basis, int thread_id = 0) {
  using Result = IntegralEngine::IntegralResult<2>;
  occ::qm::cint::Optimizer opt(env, op, 2);
  auto nthreads = occ::parallel::get_num_threads();
  auto bufsize = env.buffer_size_1e(op);
  const auto nsh = basis.size();

  auto buffer = std::make_unique<double[]>(bufsize);
  const auto &first_bf = basis.first_bf();
  for (int p = 0, pq = 0; p < nsh; p++) {
    int bf1 = first_bf[p];
    for (int q = 0; q <= p; q++) {
      if (pq++ % nthreads != thread_id)
        continue;
      int bf2 = first_bf[q];
      std::array<int, 2> idxs{p, q};
      Result args{thread_id,
                  idxs,
                  {bf1, bf2},
                  env.two_center_helper<op, kind>(idxs, opt.optimizer_ptr(),
                                                  buffer.get(), nullptr),
                  buffer.get()};
      if (args.dims[0] > -1)
        f(args);
    }
  }
}

template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center_with_shellpairs(Lambda &f,
                                         cint::IntegralEnvironment &env,
                                         const AOBasis &basis,
                                         const ShellPairList &shellpairs,
                                         int thread_id = 0) {
  using Result = IntegralEngine::IntegralResult<2>;
  occ::qm::cint::Optimizer opt(env, op, 2);
  auto nthreads = occ::parallel::get_num_threads();
  auto bufsize = env.buffer_size_1e(op);

  auto buffer = std::make_unique<double[]>(bufsize);
  const auto &first_bf = basis.first_bf();
  for (int p = 0, pq = 0; p < basis.size(); p++) {
    int bf1 = first_bf[p];
    for (const auto &q : shellpairs[p]) {
      if (pq++ % nthreads != thread_id)
        continue;
      int bf2 = first_bf[q];
      std::array<int, 2> idxs{p, static_cast<int>(q)};
      Result args{thread_id,
                  idxs,
                  {bf1, bf2},
                  env.two_center_helper<op, kind>(idxs, opt.optimizer_ptr(),
                                                  buffer.get(), nullptr),
                  buffer.get()};
      if (args.dims[0] > -1)
        f(args);
    }
  }
}

template <Op op, ShellKind kind = ShellKind::Cartesian>
Mat one_electron_operator_kernel(const AOBasis &basis,
                                 cint::IntegralEnvironment &env,
                                 const ShellPairList &shellpairs) {
  using Result = IntegralEngine::IntegralResult<2>;
  auto nthreads = occ::parallel::get_num_threads();
  const auto nbf = basis.nbf();
  Mat result = Mat::Zero(nbf, nbf);
  std::vector<Mat> results;
  results.emplace_back(Mat::Zero(nbf, nbf));
  for (size_t i = 1; i < nthreads; i++) {
    results.push_back(results[0]);
  }
  auto f = [&results](const Result &args) {
    auto &result = results[args.thread];
    Eigen::Map<const occ::Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
    result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmp;
    if (args.shell[0] != args.shell[1]) {
      result.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) =
          tmp.transpose();
    }
  };

  auto lambda = [&](int thread_id) {
    if (shellpairs.size() > 0) {
      evaluate_two_center_with_shellpairs<op, kind>(f, env, basis, shellpairs,
                                                    thread_id);
    } else {
      evaluate_two_center<op, kind>(f, env, basis, thread_id);
    }
  };
  occ::parallel::parallel_do(lambda);

  for (auto i = 1; i < nthreads; ++i) {
    results[0].noalias() += results[i];
  }
  return results[0];
}

template <Op op, ShellKind kind, typename Lambda>
void evaluate_four_center(Lambda &f, cint::IntegralEnvironment &env,
                          const AOBasis &basis, const ShellPairList &shellpairs,
                          const Mat &Dnorm = Mat(), const Mat &Schwarz = Mat(),
                          double precision = 1e-12, int thread_id = 0) {
  using Result = IntegralEngine::IntegralResult<4>;
  occ::timing::start(occ::timing::category::ints4c2e);
  auto nthreads = occ::parallel::get_num_threads();
  occ::qm::cint::Optimizer opt(env, Op::coulomb, 4);
  auto buffer = std::make_unique<double[]>(env.buffer_size_2e());
  std::array<int, 4> shell_idx;
  std::array<int, 4> bf;

  const auto &first_bf = basis.first_bf();
  const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
  // <pq|rs>
  for (size_t p = 0, pqrs = 0; p < basis.size(); p++) {
    bf[0] = first_bf[p];
    const auto &plist = shellpairs[p];
    for (const auto q : plist) {
      bf[1] = first_bf[q];

      // for Schwarz screening
      const double norm_pq = do_schwarz_screen ? Dnorm(p, q) : 0.0;

      for (size_t r = 0; r <= p; r++) {
        bf[2] = first_bf[r];
        // check if <pq|ps>, if so ensure s <= q else s <= r
        const auto s_max = (p == r) ? q : r;

        const double norm_pqr =
            do_schwarz_screen ? max_of(Dnorm(p, r), Dnorm(q, r), norm_pq) : 0.0;

        for (const auto s : shellpairs[r]) {
          if (s > s_max)
            break;
          if (pqrs++ % nthreads != thread_id)
            continue;
          const double norm_pqrs =
              do_schwarz_screen
                  ? max_of(Dnorm(p, s), Dnorm(q, s), Dnorm(r, s), norm_pqr)
                  : 0.0;
          if (do_schwarz_screen &&
              norm_pqrs * Schwarz(p, q) * Schwarz(r, s) < precision)
            continue;

          bf[3] = first_bf[s];
          shell_idx = {static_cast<int>(p), static_cast<int>(q),
                       static_cast<int>(r), static_cast<int>(s)};

          Result args{
              thread_id, shell_idx, bf,
              env.four_center_helper<Op::coulomb, kind>(
                  shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr),
              buffer.get()};
          if (args.dims[0] > -1)
            f(args);
        }
      }
    }
  }
  occ::timing::stop(occ::timing::category::ints4c2e);
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Mat fock_operator_kernel(cint::IntegralEnvironment &env, const AOBasis &basis,
                         const ShellPairList &shellpairs,
                         const MolecularOrbitals &mo, double precision = 1e-12,
                         const Mat &Schwarz = Mat()) {
  using Result = IntegralEngine::IntegralResult<4>;
  auto nthreads = occ::parallel::get_num_threads();
  constexpr Op op = Op::coulomb;
  std::vector<Mat> Fmats(nthreads, Mat::Zero(mo.D.rows(), mo.D.cols()));
  Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);

  const auto &D = mo.D;
  auto f = [&D, &Fmats](const Result &args) {
    auto &F = Fmats[args.thread];
    auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
    auto pr_qs_degree = (args.shell[0] == args.shell[2])
                            ? (args.shell[1] == args.shell[3] ? 1 : 2)
                            : 2;
    auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
    auto scale = pq_degree * rs_degree * pr_qs_degree;

    for (auto f3 = 0, f0123 = 0; f3 != args.dims[3]; ++f3) {
      const auto bf3 = f3 + args.bf[3];
      for (auto f2 = 0; f2 != args.dims[2]; ++f2) {
        const auto bf2 = f2 + args.bf[2];
        for (auto f1 = 0; f1 != args.dims[1]; ++f1) {
          const auto bf1 = f1 + args.bf[1];
          for (auto f0 = 0; f0 != args.dims[0]; ++f0, ++f0123) {
            const auto bf0 = f0 + args.bf[0];
            const auto value = args.buffer[f0123] * scale;
            detail::delegate_fock<sk>(D, F, bf0, bf1, bf2, bf3, value);
          }
        }
      }
    }
  };
  auto lambda = [&](int thread_id) {
    evaluate_four_center<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz,
                                   precision, thread_id);
  };
  occ::timing::start(occ::timing::category::fock);
  occ::parallel::parallel_do(lambda);
  occ::timing::stop(occ::timing::category::fock);

  Mat F = Mat::Zero(Fmats[0].rows(), Fmats[0].cols());

  for (const auto &part : Fmats) {
    detail::accumulate_operator_symmetric<sk>(part, F);
  }
  F *= 0.5;

  return F;
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Mat coulomb_kernel(cint::IntegralEnvironment &env, const AOBasis &basis,
                   const ShellPairList &shellpairs, const MolecularOrbitals &mo,
                   double precision = 1e-12, const Mat &Schwarz = Mat()) {
  using Result = IntegralEngine::IntegralResult<4>;
  auto nthreads = occ::parallel::get_num_threads();
  constexpr Op op = Op::coulomb;
  std::vector<Mat> Jmats(nthreads, Mat::Zero(mo.D.rows(), mo.D.cols()));
  Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);

  const auto &D = mo.D;
  auto f = [&D, &Jmats](const Result &args) {
    auto &J = Jmats[args.thread];
    auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
    auto pr_qs_degree = (args.shell[0] == args.shell[2])
                            ? (args.shell[1] == args.shell[3] ? 1 : 2)
                            : 2;
    auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
    auto scale = pq_degree * rs_degree * pr_qs_degree;

    for (auto f3 = 0, f0123 = 0; f3 != args.dims[3]; ++f3) {
      const auto bf3 = f3 + args.bf[3];
      for (auto f2 = 0; f2 != args.dims[2]; ++f2) {
        const auto bf2 = f2 + args.bf[2];
        for (auto f1 = 0; f1 != args.dims[1]; ++f1) {
          const auto bf1 = f1 + args.bf[1];
          for (auto f0 = 0; f0 != args.dims[0]; ++f0, ++f0123) {
            const auto bf0 = f0 + args.bf[0];
            const auto value = args.buffer[f0123] * scale;
            detail::delegate_j<sk>(D, J, bf0, bf1, bf2, bf3, value);
          }
        }
      }
    }
  };
  auto lambda = [&](int thread_id) {
    evaluate_four_center<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz,
                                   precision, thread_id);
  };
  occ::timing::start(occ::timing::category::fock);
  occ::parallel::parallel_do(lambda);
  occ::timing::stop(occ::timing::category::fock);

  Mat J = Mat::Zero(Jmats[0].rows(), Jmats[0].cols());

  for (size_t i = 0; i < nthreads; i++) {
    detail::accumulate_operator_symmetric<sk>(Jmats[i], J);
  }
  J *= 0.5;
  return J;
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
JKPair coulomb_and_exchange_kernel(cint::IntegralEnvironment &env,
                                   const AOBasis &basis,
                                   const ShellPairList &shellpairs,
                                   const MolecularOrbitals &mo,
                                   double precision = 1e-12,
                                   const Mat &Schwarz = Mat()) {
  using Result = IntegralEngine::IntegralResult<4>;
  auto nthreads = occ::parallel::get_num_threads();
  constexpr Op op = Op::coulomb;
  std::vector<Mat> Jmats(nthreads, Mat::Zero(mo.D.rows(), mo.D.cols()));
  std::vector<Mat> Kmats(nthreads, Mat::Zero(mo.D.rows(), mo.D.cols()));
  Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);

  const auto &D = mo.D;
  auto f = [&D, &Jmats, &Kmats](const Result &args) {
    auto &J = Jmats[args.thread];
    auto &K = Kmats[args.thread];
    auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
    auto pr_qs_degree = (args.shell[0] == args.shell[2])
                            ? (args.shell[1] == args.shell[3] ? 1 : 2)
                            : 2;
    auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
    auto scale = pq_degree * rs_degree * pr_qs_degree;

    for (auto f3 = 0, f0123 = 0; f3 != args.dims[3]; ++f3) {
      const auto bf3 = f3 + args.bf[3];
      for (auto f2 = 0; f2 != args.dims[2]; ++f2) {
        const auto bf2 = f2 + args.bf[2];
        for (auto f1 = 0; f1 != args.dims[1]; ++f1) {
          const auto bf1 = f1 + args.bf[1];
          for (auto f0 = 0; f0 != args.dims[0]; ++f0, ++f0123) {
            const auto bf0 = f0 + args.bf[0];
            const auto value = args.buffer[f0123] * scale;
            detail::delegate_jk<sk>(D, J, K, bf0, bf1, bf2, bf3, value);
          }
        }
      }
    }
  };
  auto lambda = [&](int thread_id) {
    evaluate_four_center<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz,
                                   precision, thread_id);
  };
  occ::timing::start(occ::timing::category::fock);
  occ::parallel::parallel_do(lambda);
  occ::timing::stop(occ::timing::category::fock);

  JKPair result{Mat::Zero(Jmats[0].rows(), Jmats[0].cols()),
                Mat::Zero(Kmats[0].rows(), Kmats[0].cols())};

  for (size_t i = 0; i < nthreads; i++) {
    detail::accumulate_operator_symmetric<sk>(Jmats[i], result.J);
    detail::accumulate_operator_symmetric<sk>(Kmats[i], result.K);
  }
  result.J *= 0.5;
  result.K *= 0.5;
  return result;
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
std::vector<JKPair> coulomb_and_exchange_kernel_list(
    cint::IntegralEnvironment &env, const AOBasis &basis,
    const ShellPairList &shellpairs, const std::vector<MolecularOrbitals> &mos,
    double precision = 1e-12, const Mat &Schwarz = Mat()) {

  using Result = IntegralEngine::IntegralResult<4>;
  auto nthreads = occ::parallel::get_num_threads();
  constexpr Op op = Op::coulomb;

  const int rows = mos[0].D.rows();
  const int cols = mos[0].D.cols();

  std::vector<std::vector<JKPair>> jkpairs(
      mos.size(), std::vector<JKPair>(nthreads, JKPair{Mat::Zero(rows, cols),
                                                       Mat::Zero(rows, cols)}));

  Mat Dnorm = shellblock_norm<sk, kind>(basis, mos[0].D);

  auto f = [&mos, &jkpairs](const Result &args) {
    auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
    auto pr_qs_degree = (args.shell[0] == args.shell[2])
                            ? (args.shell[1] == args.shell[3] ? 1 : 2)
                            : 2;
    auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
    auto scale = pq_degree * rs_degree * pr_qs_degree;

    for (int mo_index = 0; mo_index < mos.size(); mo_index++) {
      const auto &D = mos[mo_index].D;
      auto &J = jkpairs[mo_index][args.thread].J;
      auto &K = jkpairs[mo_index][args.thread].K;

      for (auto f3 = 0, f0123 = 0; f3 != args.dims[3]; ++f3) {
        const auto bf3 = f3 + args.bf[3];
        for (auto f2 = 0; f2 != args.dims[2]; ++f2) {
          const auto bf2 = f2 + args.bf[2];
          for (auto f1 = 0; f1 != args.dims[1]; ++f1) {
            const auto bf1 = f1 + args.bf[1];
            for (auto f0 = 0; f0 != args.dims[0]; ++f0, ++f0123) {
              const auto bf0 = f0 + args.bf[0];
              const auto value = args.buffer[f0123] * scale;
              detail::delegate_jk<sk>(D, J, K, bf0, bf1, bf2, bf3, value);
            }
          }
        }
      }
    }
  };
  auto lambda = [&](int thread_id) {
    detail::evaluate_four_center<op, kind>(f, env, basis, shellpairs, Dnorm,
                                           Schwarz, precision, thread_id);
  };
  occ::timing::start(occ::timing::category::fock);
  occ::parallel::parallel_do(lambda);
  occ::timing::stop(occ::timing::category::fock);

  std::vector<JKPair> results;
  for (size_t mo_index = 0; mo_index < mos.size(); mo_index++) {
    JKPair result{Mat::Zero(rows, cols), Mat::Zero(rows, cols)};
    const auto mo_jk = jkpairs[mo_index];
    for (size_t i = 0; i < nthreads; i++) {
      const auto &J = mo_jk[i].J;
      const auto &K = mo_jk[i].K;
      detail::accumulate_operator_symmetric<sk>(J, result.J);
      detail::accumulate_operator_symmetric<sk>(K, result.K);
    }
    result.J *= 0.5;
    result.K *= 0.5;
    results.push_back(result);
  }
  return results;
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
std::vector<Mat>
coulomb_kernel_list(cint::IntegralEnvironment &env, const AOBasis &basis,
                    const ShellPairList &shellpairs,
                    const std::vector<MolecularOrbitals> &mos,
                    double precision = 1e-12, const Mat &Schwarz = Mat()) {

  using Result = IntegralEngine::IntegralResult<4>;
  auto nthreads = occ::parallel::get_num_threads();
  constexpr Op op = Op::coulomb;

  const int rows = mos[0].D.rows();
  const int cols = mos[0].D.cols();

  std::vector<std::vector<Mat>> js(
      mos.size(), std::vector<Mat>(nthreads, Mat::Zero(rows, cols)));

  Mat Dnorm = shellblock_norm<sk, kind>(basis, mos[0].D);

  auto f = [&mos, &js](const Result &args) {
    auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
    auto pr_qs_degree = (args.shell[0] == args.shell[2])
                            ? (args.shell[1] == args.shell[3] ? 1 : 2)
                            : 2;
    auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
    auto scale = pq_degree * rs_degree * pr_qs_degree;

    for (int mo_index = 0; mo_index < mos.size(); mo_index++) {
      const auto &D = mos[mo_index].D;
      auto &J = js[mo_index][args.thread];

      for (auto f3 = 0, f0123 = 0; f3 != args.dims[3]; ++f3) {
        const auto bf3 = f3 + args.bf[3];
        for (auto f2 = 0; f2 != args.dims[2]; ++f2) {
          const auto bf2 = f2 + args.bf[2];
          for (auto f1 = 0; f1 != args.dims[1]; ++f1) {
            const auto bf1 = f1 + args.bf[1];
            for (auto f0 = 0; f0 != args.dims[0]; ++f0, ++f0123) {
              const auto bf0 = f0 + args.bf[0];
              const auto value = args.buffer[f0123] * scale;
              detail::delegate_j<sk>(D, J, bf0, bf1, bf2, bf3, value);
            }
          }
        }
      }
    }
  };
  auto lambda = [&](int thread_id) {
    detail::evaluate_four_center<op, kind>(f, env, basis, shellpairs, Dnorm,
                                           Schwarz, precision, thread_id);
  };
  occ::timing::start(occ::timing::category::fock);
  occ::parallel::parallel_do(lambda);
  occ::timing::stop(occ::timing::category::fock);

  std::vector<Mat> results;
  for (size_t mo_index = 0; mo_index < mos.size(); mo_index++) {
    Mat result = Mat::Zero(rows, cols);
    const auto mo_jk = js[mo_index];
    for (size_t i = 0; i < nthreads; i++) {
      const auto &J = mo_jk[i];
      detail::accumulate_operator_symmetric<sk>(J, result);
    }
    result *= 0.5;
    results.push_back(result);
  }
  return results;
}

} // namespace occ::qm::detail
