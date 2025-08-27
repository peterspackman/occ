#pragma once
#include "kernel_traits.h"
#include <occ/core/timings.h>
#include <occ/core/parallel.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm::detail {

template <ShellKind kind, typename Lambda>
void three_center_aux_kernel(Lambda &f, qm::cint::IntegralEnvironment &env,
                             const qm::AOBasis &aobasis,
                             const qm::AOBasis &auxbasis,
                             const ShellPairList &shellpairs) noexcept {
  using Result = IntegralEngine::IntegralResult<3>;
  occ::timing::start(occ::timing::category::ints3c2e);

  // Use TBB for dynamic load balancing
  occ::parallel::thread_local_storage<std::unique_ptr<double[]>> buffer_local(
      [&]() {
        size_t bufsize = aobasis.max_shell_size() * aobasis.max_shell_size() *
                         auxbasis.max_shell_size();
        return std::make_unique<double[]>(bufsize);
      });

  occ::parallel::thread_local_storage<occ::qm::cint::Optimizer> opt_local(
      [&]() { return occ::qm::cint::Optimizer(env, Op::coulomb, 3); });

  const auto &first_bf_ao = aobasis.first_bf();
  const auto &first_bf_aux = auxbasis.first_bf();

  occ::parallel::parallel_for(0, static_cast<size_t>(auxbasis.size()), [&](size_t auxP_idx) {
    int auxP = static_cast<int>(auxP_idx);
    auto &buffer = buffer_local.local();
    auto &opt = opt_local.local();
    
    Result args;
    args.thread = 0; // Not used in TBB mode
    args.buffer = buffer.get();
    args.bf[2] = first_bf_aux[auxP];
    args.shell[2] = auxP;
    
    std::array<int, 3> shell_idx;
    
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
  });
  occ::timing::stop(occ::timing::category::ints3c2e);
}

template <ShellKind kind = ShellKind::Cartesian>
Mat point_charge_potential_kernel(cint::IntegralEnvironment &env,
                                  const AOBasis &aobasis,
                                  const AOBasis &auxbasis,
                                  const ShellPairList &shellpairs) {
  using Result = IntegralEngine::IntegralResult<3>;
  const auto nbf = aobasis.nbf();
  
  // Use TBB thread-local storage for proper accumulation
  occ::parallel::thread_local_storage<Mat> results_local(
    [nbf]() { return Mat::Zero(nbf, nbf); }
  );
  
  auto f = [&results_local](const Result &args) {
    auto &result = results_local.local();
    Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
    result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += tmp;
    if (args.shell[0] != args.shell[1]) {
      result.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
          tmp.transpose();
    }
  };

  // Use the TBB implementation in three_center_aux_kernel directly
  three_center_aux_kernel<kind>(f, env, aobasis, auxbasis, shellpairs);

  // Combine results from all threads
  Mat result = Mat::Zero(nbf, nbf);
  for (const auto &local : results_local) {
    result += local;
  }
  return result;
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Vec electric_potential_kernel(cint::IntegralEnvironment &env,
                              const AOBasis &aobasis, const AOBasis &auxbasis,
                              const ShellPairList &shellpairs,
                              const MolecularOrbitals &mo) {
  using Result = IntegralEngine::IntegralResult<3>;
  size_t npts = auxbasis.size();

  // Use TBB thread-local storage for better scalability
  occ::parallel::thread_local_storage<Vec> results_local(
      [npts]() { return Vec::Zero(npts); });

  const auto &D = mo.D;
  auto f = [&D, &results_local](const Result &args) {
    auto &v = results_local.local();
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

  // Use the TBB implementation in three_center_aux_kernel directly
  three_center_aux_kernel<kind>(f, env, aobasis, auxbasis, shellpairs);

  // Combine results from all threads
  Vec result = Vec::Zero(npts);
  for (const auto &local : results_local) {
    result += local;
  }
  return 2 * result;
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
  
  // Use TBB thread-local storage instead of thread-indexed arrays
  occ::parallel::thread_local_storage<Mat> tl_results(Mat::Zero(nbf, nbf));
  
  auto f = [&tl_results](const Result &args) {
    auto &result = tl_results.local();
    Eigen::Map<const Mat> tmp(args.buffer, args.dims[0], args.dims[1]);
    result.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += tmp;
    if (args.shell[0] != args.shell[1]) {
      result.block(args.bf[1], args.bf[0], args.dims[1], args.dims[0]) +=
          tmp.transpose();
    }
  };

  // Use TBB parallel_for instead of deprecated parallel_do
  occ::parallel::parallel_for(size_t(0), size_t(nthreads), [&, unit_shell_index](size_t thread_id) {
    three_center_screened_kernel<kind>(f, env, aobasis, auxbasis, shellpairs,
                                       unit_shell_index, int(thread_id));
  });

  // Reduce thread-local results
  Mat result = Mat::Zero(nbf, nbf);
  for (const auto &local_result : tl_results) {
    result += local_result;
  }
  return result;
}

} // namespace occ::qm::detail
