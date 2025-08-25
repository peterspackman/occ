#pragma once
#include "kernel_traits.h"
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm::detail {

using ShellList = std::vector<Shell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = Shell::Kind;
using Op = cint::Operator;

// Two-center integral evaluation
template <Op op, ShellKind kind = ShellKind::Cartesian>
Mat one_electron_operator_kernel(const AOBasis &basis,
                                 cint::IntegralEnvironment &env,
                                 const ShellPairList &shellpairs) {
  using Result = IntegralEngine::IntegralResult<2>;
  const auto nbf = basis.nbf();
  const auto nsh = basis.size();

  occ::parallel::thread_local_storage<Mat> results_local(Mat::Zero(nbf, nbf));
  occ::parallel::thread_local_storage<occ::qm::cint::Optimizer> opt_local(
      [&env]() { return occ::qm::cint::Optimizer(env, op, 2); });
  occ::parallel::thread_local_storage<std::unique_ptr<double[]>> buffer_local(
      [&env]() { return std::make_unique<double[]>(env.buffer_size_1e(op)); });

  auto f = [&results_local](const Result &args) {
    auto &result = results_local.local();
    Eigen::Map<const occ::Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
    result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmp;
    if (args.shell[0] != args.shell[1]) {
      result.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) =
          tmp.transpose();
    }
  };

  if (shellpairs.size() > 0) {
    // Build list of shell pairs to process in parallel
    std::vector<std::pair<int, int>> pairs_to_process;
    for (int p = 0; p < nsh; p++) {
      for (const auto &q : shellpairs[p]) {
        pairs_to_process.emplace_back(p, q);
      }
    }

    occ::parallel::parallel_for(
        size_t(0), pairs_to_process.size(), [&](size_t idx) {
          auto &opt = opt_local.local();
          auto &buffer = buffer_local.local();
          const auto &first_bf = basis.first_bf();

          int p = pairs_to_process[idx].first;
          int q = pairs_to_process[idx].second;
          int bf1 = first_bf[p];
          int bf2 = first_bf[q];

          std::array<int, 2> idxs{p, q};
          Result args{0,
                      idxs,
                      {bf1, bf2},
                      env.two_center_helper<op, kind>(idxs, opt.optimizer_ptr(),
                                                      buffer.get(), nullptr),
                      buffer.get()};
          if (args.dims[0] > -1)
            f(args);
        });
  } else {
    // Build list of unique shell pairs (p,q) with p >= q
    std::vector<std::pair<int, int>> pairs_to_process;
    for (int p = 0; p < nsh; p++) {
      for (int q = 0; q <= p; q++) {
        pairs_to_process.emplace_back(p, q);
      }
    }

    occ::parallel::parallel_for(
        size_t(0), pairs_to_process.size(), [&](size_t idx) {
          auto &opt = opt_local.local();
          auto &buffer = buffer_local.local();
          const auto &first_bf = basis.first_bf();

          int p = pairs_to_process[idx].first;
          int q = pairs_to_process[idx].second;
          int bf1 = first_bf[p];
          int bf2 = first_bf[q];

          std::array<int, 2> idxs{p, q};
          Result args{0,
                      idxs,
                      {bf1, bf2},
                      env.two_center_helper<op, kind>(idxs, opt.optimizer_ptr(),
                                                      buffer.get(), nullptr),
                      buffer.get()};
          if (args.dims[0] > -1)
            f(args);
        });
  }

  // Reduce results from all threads
  Mat result = Mat::Zero(nbf, nbf);
  for (const auto &local_result : results_local) {
    result.noalias() += local_result;
  }
  return result;
}

} // namespace occ::qm::detail