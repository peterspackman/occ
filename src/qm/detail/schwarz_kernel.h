#pragma once
#include <occ/core/parallel.h>

namespace occ::qm::detail {

template <ShellKind kind = ShellKind::Cartesian>
Mat schwarz_kernel(cint::IntegralEnvironment &env, const AOBasis &basis,
                   const ShellPairList &shellpairs) {
  constexpr auto op = Op::coulomb;
  using Result = IntegralEngine::IntegralResult<4>;
  constexpr bool use_euclidean_norm{false};
  const auto nsh = basis.size();
  const auto &first_bf = basis.first_bf();
  
  // Use TBB-based thread-local storage for results
  occ::parallel::thread_local_storage<Mat> results_local(Mat::Zero(nsh, nsh));
  occ::parallel::thread_local_storage<std::unique_ptr<double[]>> buffer_local(
    [&env]() { return std::make_unique<double[]>(env.buffer_size_2e()); }
  );

  auto f = [&results_local](const Result &args) {
    auto &result = results_local.local();
    auto N = args.dims[0] * args.dims[1];
    Eigen::Map<const occ::Mat> tmp(args.buffer, N, N);
    double sq_norm =
        use_euclidean_norm ? tmp.norm() : tmp.array().abs().maxCoeff();
    double norm = std::sqrt(sq_norm);
    result(args.shell[0], args.shell[1]) = norm;
    result(args.shell[1], args.shell[0]) = norm;
  };

  // Build list of shell pairs to process in parallel
  std::vector<std::pair<int, int>> pairs_to_process;
  for (int p = 0; p < nsh; p++) {
    for (const auto &q : shellpairs[p]) {
      pairs_to_process.emplace_back(p, q);
    }
  }
  
  occ::parallel::parallel_for(size_t(0), pairs_to_process.size(), [&](size_t idx) {
    auto &buffer = buffer_local.local();
    
    int p = pairs_to_process[idx].first;
    int q = pairs_to_process[idx].second;
    int bf1 = first_bf[p];
    int bf2 = first_bf[q];
    
    std::array<int, 4> idxs{p, q, p, q};
    Result args{0, idxs, {bf1, bf2, bf1, bf2},
                env.four_center_helper<op, kind>(idxs, nullptr,
                                                 buffer.get(), nullptr),
                buffer.get()};
    if (args.dims[0] > -1)
      f(args);
  });

  // Reduce results from all threads
  Mat result = Mat::Zero(nsh, nsh);
  for (const auto &local_result : results_local) {
    result.noalias() += local_result;
  }

  return result;
}
} // namespace occ::qm::detail
