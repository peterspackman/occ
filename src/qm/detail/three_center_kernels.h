#pragma once
#include "kernel_traits.h"
#include <occ/core/timings.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm::detail {

template <ShellKind kind, typename Lambda>
void three_center_aux_kernel(Lambda &f, qm::cint::IntegralEnvironment &env,
                             const qm::AOBasis &aobasis,
                             const qm::AOBasis &auxbasis,
                             const ShellPairList &shellpairs,
                             int thread_id = 0) noexcept {
  using Result = IntegralEngine::IntegralResult<3>;
  occ::timing::start(occ::timing::category::ints3c2e);
  auto nthreads = occ::parallel::get_num_threads();
  occ::qm::cint::Optimizer opt(env, Op::coulomb, 3);
  size_t bufsize = aobasis.max_shell_size() * aobasis.max_shell_size() *
                   auxbasis.max_shell_size();
  auto buffer = std::make_unique<double[]>(bufsize);
  Result args;
  args.thread = thread_id;
  args.buffer = buffer.get();
  std::array<int, 3> shell_idx;
  const auto &first_bf_ao = aobasis.first_bf();
  const auto &first_bf_aux = auxbasis.first_bf();
  for (int auxP = 0; auxP < auxbasis.size(); auxP++) {
    if (auxP % nthreads != thread_id)
      continue;
    args.bf[2] = first_bf_aux[auxP];
    args.shell[2] = auxP;
    for (int p = 0; p < aobasis.size(); p++) {
      args.bf[0] = first_bf_ao[p];
      args.shell[0] = p;
      const auto &plist = shellpairs[p];
      for (const auto &q : plist) {
        args.bf[1] = first_bf_ao[q];
        args.shell[1] = q;
        shell_idx = {p, static_cast<int>(q),
                     auxP + static_cast<int>(aobasis.size())};
        args.dims = env.three_center_helper<Op::coulomb, kind>(
            shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr);
        if (args.dims[0] > -1) {
          f(args);
        }
      }
    }
  }
  occ::timing::stop(occ::timing::category::ints3c2e);
}

template <ShellKind kind = ShellKind::Cartesian>
Mat point_charge_potential_kernel(cint::IntegralEnvironment &env,
                                  const AOBasis &aobasis,
                                  const AOBasis &auxbasis,
                                  const ShellPairList &shellpairs) {
  using Result = IntegralEngine::IntegralResult<3>;
  auto nthreads = occ::parallel::get_num_threads();
  const auto nbf = aobasis.nbf();
  std::vector<Mat> results(nthreads, Mat::Zero(nbf, nbf));
  auto f = [&results](const Result &args) {
    auto &result = results[args.thread];
    Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
    result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += tmp;
    if (args.shell[0] != args.shell[1]) {
      result.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
          tmp.transpose();
    }
  };

  auto lambda = [&](int thread_id) {
    three_center_aux_kernel<kind>(f, env, aobasis, auxbasis, shellpairs,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda);

  for (auto i = 1; i < nthreads; i++) {
    results[0] += results[i];
  }
  return results[0];
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Vec electric_potential_kernel(cint::IntegralEnvironment &env,
                              const AOBasis &aobasis, const AOBasis &auxbasis,
                              const ShellPairList &shellpairs,
                              const MolecularOrbitals &mo) {
  using Result = IntegralEngine::IntegralResult<3>;
  auto nthreads = occ::parallel::get_num_threads();
  size_t npts = auxbasis.size();
  std::vector<Vec> results(nthreads, Vec::Zero(npts));

  const auto &D = mo.D;
  auto f = [&D, &results](const Result &args) {
    auto &v = results[args.thread];
    auto scale = (args.shell[0] == args.shell[1]) ? 1 : 2;
    Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
    if constexpr (sk == SpinorbitalKind::Restricted) {
      v(args.shell[2]) +=
          (D.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]).array() *
           tmp.array())
              .sum() *
          scale;
    } else if constexpr (sk == SpinorbitalKind::Unrestricted) {
      const auto alpha = occ::qm::block::a(D).block(args.bf[0], args.bf[1],
                                                    args.dims[0], args.dims[1]);
      const auto beta = occ::qm::block::b(D).block(args.bf[0], args.bf[1],
                                                   args.dims[0], args.dims[1]);
      v(args.shell[2]) +=
          ((alpha.array() + beta.array()) * tmp.array()).sum() * scale;
    } else if constexpr (sk == SpinorbitalKind::General) {
      const auto aa = occ::qm::block::aa(D).block(args.bf[0], args.bf[1],
                                                  args.dims[0], args.dims[1]);
      const auto ab = occ::qm::block::ab(D).block(args.bf[0], args.bf[1],
                                                  args.dims[0], args.dims[1]);
      const auto ba = occ::qm::block::ba(D).block(args.bf[0], args.bf[1],
                                                  args.dims[0], args.dims[1]);
      const auto bb = occ::qm::block::bb(D).block(args.bf[0], args.bf[1],
                                                  args.dims[0], args.dims[1]);
      v(args.shell[2]) +=
          ((aa.array() + ab.array() + ba.array() + bb.array()) * tmp.array())
              .sum() *
          scale;
    }
  };

  auto lambda = [&](int thread_id) {
    three_center_aux_kernel<kind>(f, env, aobasis, auxbasis, shellpairs,
                                  thread_id);
  };
  occ::parallel::parallel_do(lambda);

  for (auto i = 1; i < nthreads; i++) {
    results[0] += results[i];
  }
  return 2 * results[0];
}

template <ShellKind kind, typename Lambda>
void three_center_screened_kernel(Lambda &f, qm::cint::IntegralEnvironment &env,
                                  const qm::AOBasis &aobasis,
                                  const qm::AOBasis &auxbasis,
                                  const ShellPairList &shellpairs,
                                  int unit_shell_index,
                                  int thread_id = 0) noexcept {
  using Result = IntegralEngine::IntegralResult<3>;
  occ::timing::start(occ::timing::category::ints3c2e);
  auto nthreads = occ::parallel::get_num_threads();
  occ::qm::cint::Optimizer opt(env, Op::coulomb, 4);
  size_t bufsize = aobasis.max_shell_size() * aobasis.max_shell_size() *
                   auxbasis.max_shell_size() * auxbasis.max_shell_size();
  auto buffer = std::make_unique<double[]>(bufsize);
  Result args;
  args.thread = thread_id;
  args.buffer = buffer.get();
  std::array<int, 3> shell_idx;
  unit_shell_index += static_cast<int>(aobasis.size());
  const auto &first_bf_ao = aobasis.first_bf();
  const auto &first_bf_aux = auxbasis.first_bf();
  for (int auxP = 0; auxP < auxbasis.size() - 1; auxP++) {
    if (auxP % nthreads != thread_id)
      continue;
    args.bf[2] = first_bf_aux[auxP];
    args.shell[2] = auxP;
    for (int p = 0; p < aobasis.size(); p++) {
      args.bf[0] = first_bf_ao[p];
      args.shell[0] = p;
      const auto &plist = shellpairs[p];
      for (const auto &q : plist) {
        args.bf[1] = first_bf_ao[q];
        args.shell[1] = q;
        shell_idx = {p, static_cast<int>(q),
                     auxP + static_cast<int>(aobasis.size())};
        args.dims = env.three_center_rs_helper<kind>(
            shell_idx, unit_shell_index, opt.optimizer_ptr(), buffer.get(),
            nullptr);
        if (args.dims[0] > -1) {
          f(args);
        }
      }
    }
  }
  occ::timing::stop(occ::timing::category::ints3c2e);
}

template <ShellKind kind = ShellKind::Cartesian>
Mat point_charge_potential_screened_kernel(cint::IntegralEnvironment &env,
                                           const AOBasis &aobasis,
                                           const AOBasis &auxbasis,
                                           const ShellPairList &shellpairs,
                                           int unit_shell_index) {
  using Result = IntegralEngine::IntegralResult<3>;
  auto nthreads = occ::parallel::get_num_threads();
  const auto nbf = aobasis.nbf();
  const auto nsh = aobasis.size();
  std::vector<Mat> results(nthreads, Mat::Zero(nbf, nbf));
  auto f = [&results](const Result &args) {
    auto &result = results[args.thread];
    Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
    result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += tmp;
    if (args.shell[0] != args.shell[1]) {
      result.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
          tmp.transpose();
    }
  };

  auto lambda = [&, unit_shell_index](int thread_id) {
    three_center_screened_kernel<kind>(f, env, aobasis, auxbasis, shellpairs,
                                       unit_shell_index, thread_id);
  };
  occ::parallel::parallel_do(lambda);

  for (auto i = 1; i < nthreads; i++) {
    results[0] += results[i];
  }
  return results[0];
}

} // namespace occ::qm::detail
