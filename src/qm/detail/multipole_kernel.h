#pragma once
#include "kernel_traits.h"

namespace occ::qm::detail {

template <int order, SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Vec multipole_kernel(const AOBasis &basis, cint::IntegralEnvironment &env,
                     const ShellPairList &shellpairs,
                     const MolecularOrbitals &mo, const Vec3 &origin) {
  using Result = IntegralEngine::IntegralResult<2>;
  constexpr std::array<Op, 5> ops{Op::overlap, Op::dipole, Op::quadrupole,
                                  Op::octapole, Op::hexadecapole};
  constexpr Op op = ops[order];

  auto nthreads = occ::parallel::get_num_threads();
  size_t num_components = occ::core::num_multipole_components_tensor(order);
  env.set_common_origin({origin.x(), origin.y(), origin.z()});
  std::vector<Vec> results(nthreads, Vec::Zero(num_components));
  const auto &D = mo.D;
  /*
   * For symmetric matrices
   * the of a matrix product tr(D @ O) is equal to
   * the sum of the elementwise product with the transpose:
   * tr(D @ O) == sum(D * O^T)
   * since expectation is -2 tr(D @ O) we factor that into the
   * inner loop
   */
  auto f = [&D, &results, &num_components](const Result &args) {
    auto &result = results[args.thread];
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

  auto lambda = [&](int thread_id) {
    detail::evaluate_two_center_with_shellpairs<op, kind>(
        f, env, basis, shellpairs, thread_id);
  };
  occ::parallel::parallel_do(lambda);

  for (auto i = 1; i < nthreads; ++i) {
    results[0].noalias() += results[i];
  }

  results[0] *= -2;
  // TODO refactor this
  Vec unique(occ::core::num_unique_multipole_components(order));
  if constexpr (order <= 1) {
    return results[0];
  } else if constexpr (order == 2) {
    const auto &Q = results[0];
    unique(0) = Q(0); // xx
    unique(1) = Q(1); // xy
    unique(2) = Q(2); // xz
    unique(3) = Q(4); // yy
    unique(4) = Q(5); // yz
    unique(5) = Q(8); // zz
  } else if constexpr (order == 3) {
    const auto &O = results[0];
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
    const auto &H = results[0];
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
