#pragma once

namespace occ::qm::detail {

template <ShellKind kind = ShellKind::Cartesian>
Mat schwarz_kernel(cint::IntegralEnvironment &env, const AOBasis &basis,
                   const ShellPairList &shellpairs) {
  constexpr auto op = Op::coulomb;
  using Result = IntegralEngine::IntegralResult<4>;
  auto nthreads = occ::parallel::get_num_threads();
  constexpr bool use_euclidean_norm{false};
  const auto nsh = basis.size();
  const auto &first_bf = basis.first_bf();
  std::vector<Mat> results;
  results.emplace_back(Mat::Zero(nsh, nsh));
  for (size_t i = 1; i < nthreads; i++) {
    results.push_back(results[0]);
  }

  auto f = [&results](const Result &args) {
    auto &result = results[args.thread];
    auto N = args.dims[0] * args.dims[1];
    Eigen::Map<const occ::Mat> tmp(args.buffer, N, N);
    double sq_norm =
        use_euclidean_norm ? tmp.norm() : tmp.array().abs().maxCoeff();
    double norm = std::sqrt(sq_norm);
    result(args.shell[0], args.shell[1]) = norm;
    result(args.shell[1], args.shell[0]) = norm;
  };

  auto lambda = [&](int thread_id) {
    auto buffer = std::make_unique<double[]>(env.buffer_size_2e());
    for (int p = 0, pq = 0; p < nsh; p++) {
      int bf1 = first_bf[p];
      for (const auto &q : shellpairs[p]) {
        if (pq++ % nthreads != thread_id)
          continue;
        int bf2 = first_bf[q];
        std::array<int, 4> idxs{p, static_cast<int>(q), p, static_cast<int>(q)};
        Result args{thread_id,
                    idxs,
                    {bf1, bf2, bf1, bf2},
                    env.four_center_helper<op, kind>(idxs, nullptr,
                                                     buffer.get(), nullptr),
                    buffer.get()};
        if (args.dims[0] > -1)
          f(args);
      }
    }
  };
  occ::parallel::parallel_do(lambda);

  for (auto i = 1; i < nthreads; ++i) {
    results[0].noalias() += results[i];
  }

  return results[0];
}
} // namespace occ::qm::detail
