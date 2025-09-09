#pragma once
#include "kernel_traits.h"
#include <occ/core/parallel.h>

namespace occ::qm::detail {

template <int order, SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Vec multipole_kernel(const AOBasis &basis, cint::IntegralEnvironment &env,
                     const ShellPairList &shellpairs,
                     const MolecularOrbitals &mo, const Vec3 &origin) {
  using Result = IntegralEngine::IntegralResult<2>;
  constexpr std::array<Op, 5> ops{Op::overlap, Op::dipole, Op::quadrupole,
                                  Op::octapole, Op::hexadecapole};
  constexpr Op op = ops[order];

  size_t num_components = occ::core::num_multipole_components_tensor(order);
  env.set_common_origin({origin.x(), origin.y(), origin.z()});
  const auto &D = mo.D;
  
  // Use TBB-based thread-local storage for accumulation
  occ::parallel::thread_local_storage<Vec> results_local(Vec::Zero(num_components));
  occ::parallel::thread_local_storage<occ::qm::cint::Optimizer> opt_local(
    [&env]() { return occ::qm::cint::Optimizer(env, op, 2); }
  );
  occ::parallel::thread_local_storage<std::unique_ptr<double[]>> buffer_local(
    [&env]() { return std::make_unique<double[]>(env.buffer_size_1e(op)); }
  );
  
  /*
   * For symmetric matrices
   * the of a matrix product tr(D @ O) is equal to
   * the sum of the elementwise product with the transpose:
   * tr(D @ O) == sum(D * O^T)
   * since expectation is -2 tr(D @ O) we factor that into the
   * inner loop
   */
  auto f = [&D, &results_local, &num_components](const Result &args) {
    auto &result = results_local.local();
    size_t offset = 0;
    double scale = (args.shell[0] != args.shell[1]) ? 2.0 : 1.0;
    // TODO avoid redundant tensor calcs
    for (size_t n = 0; n < num_components; n++) {
      Eigen::Map<const occ::Mat> tmp(args.buffer + offset, args.dims[0],
                                     args.dims[1]);
      if constexpr (sk == SpinorbitalKind::Restricted) {
        result(n) +=
            scale * (D.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1])
                         .array() *
                     tmp.array())
                        .sum();
      } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
        const auto Da = qm::block::a(D);
        const auto Db = qm::block::b(D);
        result(n) += scale * (Da.block(args.bf[0], args.bf[1], args.dims[0],
                                       args.dims[1])
                                  .array() *
                              tmp.array())
                                 .sum();
        result(n) += scale * (Db.block(args.bf[0], args.bf[1], args.dims[0],
                                       args.dims[1])
                                  .array() *
                              tmp.array())
                                 .sum();
      } else if constexpr (sk == SpinorbitalKind::General) {
        const auto Daa = qm::block::aa(D);
        const auto Dab = qm::block::ab(D);
        const auto Dba = qm::block::ba(D);
        const auto Dbb = qm::block::bb(D);
        result(n) += scale * (Daa.block(args.bf[0], args.bf[1], args.dims[0],
                                        args.dims[1])
                                  .array() *
                              tmp.array())
                                 .sum();
        result(n) += scale * (Dab.block(args.bf[0], args.bf[1], args.dims[0],
                                        args.dims[1])
                                  .array() *
                              tmp.array())
                                 .sum();
        result(n) += scale * (Dba.block(args.bf[0], args.bf[1], args.dims[0],
                                        args.dims[1])
                                  .array() *
                              tmp.array())
                                 .sum();
        result(n) += scale * (Dbb.block(args.bf[0], args.bf[1], args.dims[0],
                                        args.dims[1])
                                  .array() *
                              tmp.array())
                                 .sum();
      }
      offset += tmp.size();
    }
  };

  const auto nsh = basis.size();
  const auto &first_bf = basis.first_bf();

  if (shellpairs.size() > 0) {
    // Build list of shell pairs to process in parallel
    std::vector<std::pair<int, int>> pairs_to_process;
    for (int p = 0; p < nsh; p++) {
      for (const auto &q : shellpairs[p]) {
        pairs_to_process.emplace_back(p, q);
      }
    }
    
    occ::parallel::parallel_for(size_t(0), pairs_to_process.size(), [&](size_t idx) {
      auto &opt = opt_local.local();
      auto &buffer = buffer_local.local();
      
      int p = pairs_to_process[idx].first;
      int q = pairs_to_process[idx].second;
      int bf1 = first_bf[p];
      int bf2 = first_bf[q];
      
      std::array<int, 2> idxs{p, q};
      Result args{0, idxs, {bf1, bf2},
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
    
    occ::parallel::parallel_for(size_t(0), pairs_to_process.size(), [&](size_t idx) {
      auto &opt = opt_local.local();
      auto &buffer = buffer_local.local();
      
      int p = pairs_to_process[idx].first;
      int q = pairs_to_process[idx].second;
      int bf1 = first_bf[p];
      int bf2 = first_bf[q];
      
      std::array<int, 2> idxs{p, q};
      Result args{0, idxs, {bf1, bf2},
                  env.two_center_helper<op, kind>(idxs, opt.optimizer_ptr(),
                                                  buffer.get(), nullptr),
                  buffer.get()};
      if (args.dims[0] > -1)
        f(args);
    });
  }

  // Reduce results from all threads
  Vec total_result = Vec::Zero(num_components);
  for (const auto &local_result : results_local) {
    total_result.noalias() += local_result;
  }

  total_result *= -2;
  // TODO refactor this
  Vec unique(occ::core::num_unique_multipole_components(order));
  if constexpr (order <= 1) {
    return total_result;
  } else if constexpr (order == 2) {
    const auto &Q = total_result;
    unique(0) = Q(0); // xx
    unique(1) = Q(1); // xy
    unique(2) = Q(2); // xz
    unique(3) = Q(4); // yy
    unique(4) = Q(5); // yz
    unique(5) = Q(8); // zz
  } else if constexpr (order == 3) {
    const auto &O = total_result;
    unique(0) = O(0);  // xxx
    unique(1) = O(1);  // xxy
    unique(2) = O(2);  // xxz
    unique(3) = O(4);  // xyy
    unique(4) = O(5);  // xyz
    unique(5) = O(8);  // xzz
    unique(6) = O(13); // yyy
    unique(7) = O(14); // yyz
    unique(8) = O(17); // yzz
    unique(9) = O(26); // zzz
  } else if constexpr (order == 4) {
    const auto &H = total_result;
    unique(0) = H(0);   // xxxx
    unique(1) = H(1);   // xxxy
    unique(2) = H(2);   // xxxz
    unique(3) = H(4);   // xxyy
    unique(4) = H(5);   // xxyz
    unique(5) = H(8);   // xxzz
    unique(6) = H(13);  // xyyy
    unique(7) = H(14);  // xyyz
    unique(8) = H(17);  // xyzz
    unique(9) = H(26);  // xzzz
    unique(10) = H(40); // yyyy
    unique(11) = H(41); // yyyz
    unique(12) = H(44); // yyzz
    unique(13) = H(53); // yzzz
    unique(14) = H(80); // zzzz
  }
  return unique;
}
} // namespace occ::qm::detail
