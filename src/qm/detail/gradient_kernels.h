#pragma once
#include "kernel_traits.h"
#include <occ/core/timings.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm::detail {

template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center_grad(Lambda &f, IntEnv &env, const AOBasis &basis,
                              int thread_id = 0) {
  using Result = IntegralEngine::IntegralResult<2>;
  occ::qm::cint::Optimizer opt(env, op, 2, 1);
  auto nthreads = occ::parallel::get_num_threads();
  auto bufsize = env.buffer_size_1e(op, 1);
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
                  env.two_center_helper_grad<op, kind>(
                      idxs, opt.optimizer_ptr(), buffer.get(), nullptr),
                  buffer.get()};
      if (args.dims[0] > -1)
        f(args);
    }
  }
}

template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center_with_shellpairs_grad(Lambda &f, IntEnv &env,
                                              const AOBasis &basis,
                                              const ShellPairList &shellpairs,
                                              int thread_id = 0) {
  using Result = IntegralEngine::IntegralResult<2>;
  occ::qm::cint::Optimizer opt(env, op, 2, 1);
  auto nthreads = occ::parallel::get_num_threads();
  auto bufsize = env.buffer_size_1e(op, 1);

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
                  env.two_center_helper_grad<op, kind>(
                      idxs, opt.optimizer_ptr(), buffer.get(), nullptr),
                  buffer.get()};
      if (args.dims[0] > -1)
        f(args);

      if (p != q) {
        std::array<int, 2> idxs2{static_cast<int>(q), p};
        Result args2{thread_id,
                     idxs2,
                     {bf2, bf1},
                     env.two_center_helper_grad<op, kind>(
                         idxs2, opt.optimizer_ptr(), buffer.get(), nullptr),
                     buffer.get()};
        if (args2.dims[0] > -1)
          f(args2);
      }
    }
  }
}

template <Op op, ShellKind kind, typename Lambda>
void evaluate_four_center_grad(Lambda &f, IntEnv &env, const AOBasis &basis,
                               const ShellPairList &shellpairs,
                               const Mat &Dnorm = Mat(),
                               const Mat &Schwarz = Mat(),
                               double precision = 1e-12, int thread_id = 0) {
  using Result = IntegralEngine::IntegralResult<4>;
  occ::timing::start(occ::timing::category::ints4c2e);
  auto nthreads = occ::parallel::get_num_threads();
  occ::qm::cint::Optimizer opt(env, Op::coulomb, 4, 1);
  auto buffer = std::make_unique<double[]>(env.buffer_size_2e(1));
  std::array<int, 4> shell_idx;
  std::array<int, 4> bf;

  const auto &first_bf = basis.first_bf();

  const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
  // <pq|rs>
  for (size_t p = 0, pqrs = 0; p < basis.size(); p++) {
    bf[0] = first_bf[p];
    for (const auto q : shellpairs[p]) {
      bf[1] = first_bf[q];
      const double norm_pq = do_schwarz_screen ? Dnorm(p, q) : 0.0;
      for (size_t r = 0; r < basis.size(); r++) {
        bf[2] = first_bf[r];
        for (const auto s : shellpairs[r]) {
          if (pqrs++ % nthreads != thread_id)
            continue;
          const double norm_pqr =
              do_schwarz_screen ? max_of(Dnorm(p, r), Dnorm(q, r), norm_pq)
                                : 0.0;
          bf[3] = first_bf[s];
          const double norm_pqrs =
              do_schwarz_screen
                  ? max_of(Dnorm(p, s), Dnorm(q, s), Dnorm(r, s), norm_pqr)
                  : 0.0;
          if (do_schwarz_screen &&
              norm_pqrs * Schwarz(p, q) * Schwarz(r, s) < precision)
            continue;
          shell_idx = {static_cast<int>(p), static_cast<int>(q),
                       static_cast<int>(r), static_cast<int>(s)};
          // PQ | RS and PQ | SR

          {
            Result args{
                thread_id, shell_idx, bf,
                env.four_center_helper_grad<Op::coulomb, kind>(
                    shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr),
                buffer.get()};
            if (args.dims[0] > -1) {
              f(args);
            }
          }
          // QP | RS and QP | SR
          if (p != q) {
            std::array<int, 4> shell_idx2 = {
                static_cast<int>(q), static_cast<int>(p), static_cast<int>(r),
                static_cast<int>(s)};
            std::array<int, 4> bf2{bf[1], bf[0], bf[2], bf[3]};
            Result args2{
                thread_id, shell_idx2, bf2,
                env.four_center_helper_grad<Op::coulomb, kind>(
                    shell_idx2, opt.optimizer_ptr(), buffer.get(), nullptr),
                buffer.get()};
            if (args2.dims[0] > -1) {
              f(args2);
            }
          }
        }
      }
    }
  }
  occ::timing::stop(occ::timing::category::ints4c2e);
}

template <class Func>
inline void
four_center_inner_loop(Func &store,
                       const IntegralEngine::IntegralResult<4> &args,
                       const Mat &D, MatTriple &dest) {
  auto scale = (args.shell[2] == args.shell[3]) ? 1 : 2;

  const auto num_elements =
      args.dims[0] * args.dims[1] * args.dims[2] * args.dims[3];

  for (auto f3 = 0, f0123 = 0; f3 < args.dims[3]; ++f3) {
    const auto bf3 = f3 + args.bf[3];
    for (auto f2 = 0; f2 < args.dims[2]; ++f2) {
      const auto bf2 = f2 + args.bf[2];
      for (auto f1 = 0; f1 < args.dims[1]; ++f1) {
        const auto bf1 = f1 + args.bf[1];
        for (auto f0 = 0; f0 < args.dims[0]; ++f0, ++f0123) {
          const auto bf0 = f0 + args.bf[0];
          // x
          store(D, dest.x, bf0, bf1, bf2, bf3, args.buffer[f0123] * scale);
          // y
          store(D, dest.y, bf0, bf1, bf2, bf3,
                args.buffer[f0123 + num_elements] * scale);
          // z

          store(D, dest.z, bf0, bf1, bf2, bf3,
                args.buffer[f0123 + 2 * num_elements] * scale);
        }
      }
    }
  }
}

} // namespace occ::qm::detail
