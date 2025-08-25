#pragma once
#include "kernel_traits.h"
#include <mutex>
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

// Legacy function - now unused, kept for compatibility
template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center(Lambda &f, cint::IntegralEnvironment &env,
                         const AOBasis &basis, int thread_id = 0) {
  // This function is deprecated - use TBB-based parallel_for in one_electron_operator_kernel instead
}

// Legacy function - now unused, kept for compatibility
template <Op op, ShellKind kind, typename Lambda>
void evaluate_two_center_with_shellpairs(Lambda &f,
                                         cint::IntegralEnvironment &env,
                                         const AOBasis &basis,
                                         const ShellPairList &shellpairs,
                                         int thread_id = 0) {
  // This function is deprecated - use TBB-based parallel_for in one_electron_operator_kernel instead
}

template <Op op, ShellKind kind = ShellKind::Cartesian>
Mat one_electron_operator_kernel(const AOBasis &basis,
                                 cint::IntegralEnvironment &env,
                                 const ShellPairList &shellpairs) {
  using Result = IntegralEngine::IntegralResult<2>;
  const auto nbf = basis.nbf();
  const auto nsh = basis.size();
  
  occ::parallel::thread_local_storage<Mat> results_local(Mat::Zero(nbf, nbf));
  occ::parallel::thread_local_storage<occ::qm::cint::Optimizer> opt_local(
    [&env]() { return occ::qm::cint::Optimizer(env, op, 2); }
  );
  occ::parallel::thread_local_storage<std::unique_ptr<double[]>> buffer_local(
    [&env]() { return std::make_unique<double[]>(env.buffer_size_1e(op)); }
  );
  
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
    
    occ::parallel::parallel_for(size_t(0), pairs_to_process.size(), [&](size_t idx) {
      auto &opt = opt_local.local();
      auto &buffer = buffer_local.local();
      const auto &first_bf = basis.first_bf();
      
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
      const auto &first_bf = basis.first_bf();
      
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
  Mat result = Mat::Zero(nbf, nbf);
  for (const auto &local_result : results_local) {
    result.noalias() += local_result;
  }
  return result;
}

// Legacy function - now unused, kept for compatibility
template <Op op, ShellKind kind, typename Lambda>
void evaluate_four_center(Lambda &f, cint::IntegralEnvironment &env,
                          const AOBasis &basis, const ShellPairList &shellpairs,
                          const Mat &Dnorm = Mat(), const Mat &Schwarz = Mat(),
                          double precision = 1e-12, int thread_id = 0) {
  // This function is deprecated - use evaluate_four_center_tbb instead
}

// TBB-optimized version with better load balancing
template <Op op, ShellKind kind, typename Lambda>
void evaluate_four_center_tbb(Lambda &f, cint::IntegralEnvironment &env,
                              const AOBasis &basis, const ShellPairList &shellpairs,
                              const Mat &Dnorm = Mat(), const Mat &Schwarz = Mat(),
                              double precision = 1e-12) {
  using Result = IntegralEngine::IntegralResult<4>;
  
  // Build a list of shell quartets to process
  struct ShellQuartet {
    size_t p, q, r, s;
    double norm_pqrs;
    double schwarz_pq_rs;
  };
  
  std::vector<ShellQuartet> quartets;
  const auto &first_bf = basis.first_bf();
  const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
  
  // First pass: collect all shell quartets that pass screening
  for (size_t p = 0; p < basis.size(); p++) {
    const auto &plist = shellpairs[p];
    for (const auto q : plist) {
      const double norm_pq = do_schwarz_screen ? Dnorm(p, q) : 0.0;
      
      for (size_t r = 0; r <= p; r++) {
        const auto s_max = (p == r) ? q : r;
        const double norm_pqr = do_schwarz_screen 
            ? max_of(Dnorm(p, r), Dnorm(q, r), norm_pq) : 0.0;
        
        for (const auto s : shellpairs[r]) {
          if (s > s_max) break;
          
          const double norm_pqrs = do_schwarz_screen
              ? max_of(Dnorm(p, s), Dnorm(q, s), Dnorm(r, s), norm_pqr)
              : 0.0;
          const double schwarz_pq_rs = do_schwarz_screen 
              ? Schwarz(p, q) * Schwarz(r, s) : 1.0;
          
          if (do_schwarz_screen && norm_pqrs * schwarz_pq_rs < precision)
            continue;
          
          quartets.push_back({p, q, r, s, norm_pqrs, schwarz_pq_rs});
        }
      }
    }
  }
  
  // Thread-local storage for optimizer and buffer
  occ::parallel::thread_local_storage<occ::qm::cint::Optimizer> opt_local(
    [&env]() { return occ::qm::cint::Optimizer(env, Op::coulomb, 4); }
  );
  
  occ::parallel::thread_local_storage<std::unique_ptr<double[]>> buffer_local(
    [&env]() { return std::make_unique<double[]>(env.buffer_size_2e()); }
  );
  
  // Process quartets in parallel with dynamic load balancing
  occ::parallel::parallel_for(0, quartets.size(),
    [&](size_t idx) {
      auto &opt = opt_local.local();
      auto &buffer = buffer_local.local();
      
      const auto &quartet = quartets[idx];
      std::array<int, 4> shell_idx = {
        static_cast<int>(quartet.p), static_cast<int>(quartet.q),
        static_cast<int>(quartet.r), static_cast<int>(quartet.s)
      };
      std::array<int, 4> bf = {
        first_bf[quartet.p], first_bf[quartet.q],
        first_bf[quartet.r], first_bf[quartet.s]
      };
      
      Result args{
        0, shell_idx, bf,
        env.four_center_helper<op, kind>(
            shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr),
        buffer.get()
      };
      
      if (args.dims[0] > -1)
        f(args);
    });
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Mat fock_operator_kernel(cint::IntegralEnvironment &env, const AOBasis &basis,
                         const ShellPairList &shellpairs,
                         const MolecularOrbitals &mo, double precision = 1e-12,
                         const Mat &Schwarz = Mat()) {
  using Result = IntegralEngine::IntegralResult<4>;
  auto nthreads = occ::parallel::get_num_threads();
  constexpr Op op = Op::coulomb;
  Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);
  const auto &D = mo.D;
  
  // Use TBB with proper thread-local storage and final reduction
  occ::timing::start(occ::timing::category::ints4c2e);
  
  // Thread-local storage for Fock matrices
  occ::parallel::thread_local_storage<Mat> F_local(
    [&mo]() { return Mat::Zero(mo.D.rows(), mo.D.cols()); }
  );
  
  auto f = [&D, &F_local](const Result &args) {
    auto& F = F_local.local();
    
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
  
  evaluate_four_center_tbb<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz, precision);
  
  // Combine all thread-local matrices
  Mat F = Mat::Zero(mo.D.rows(), mo.D.cols());
  for (const auto& local_F : F_local) {
    detail::accumulate_operator_symmetric<sk>(local_F, F);
  }
  
  occ::timing::stop(occ::timing::category::ints4c2e);
  F *= 0.5;
  return F;
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
Mat coulomb_kernel(cint::IntegralEnvironment &env, const AOBasis &basis,
                   const ShellPairList &shellpairs, const MolecularOrbitals &mo,
                   double precision = 1e-12, const Mat &Schwarz = Mat()) {
  using Result = IntegralEngine::IntegralResult<4>;
  constexpr Op op = Op::coulomb;
  Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);
  
  occ::timing::start(occ::timing::category::ints4c2e);
  
  // Thread-local storage for Coulomb matrices
  occ::parallel::thread_local_storage<Mat> J_local(
    [&mo]() { return Mat::Zero(mo.D.rows(), mo.D.cols()); }
  );
  
  const auto &D = mo.D;
  auto f = [&D, &J_local](const Result &args) {
    auto &J = J_local.local();
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
  
  evaluate_four_center_tbb<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz, precision);
  
  // Combine all thread-local matrices
  Mat J = Mat::Zero(mo.D.rows(), mo.D.cols());
  for (const auto& local_J : J_local) {
    detail::accumulate_operator_symmetric<sk>(local_J, J);
  }
  
  occ::timing::stop(occ::timing::category::ints4c2e);
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
  constexpr Op op = Op::coulomb;
  Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);
  
  occ::timing::start(occ::timing::category::ints4c2e);
  
  // Thread-local storage for J and K matrices
  occ::parallel::thread_local_storage<JKPair> JK_local(
    [&mo]() { return JKPair{Mat::Zero(mo.D.rows(), mo.D.cols()),
                            Mat::Zero(mo.D.rows(), mo.D.cols())}; }
  );

  const auto &D = mo.D;
  auto f = [&D, &JK_local](const Result &args) {
    auto &jk = JK_local.local();
    auto &J = jk.J;
    auto &K = jk.K;
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
  
  evaluate_four_center_tbb<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz, precision);

  // Combine all thread-local matrices
  JKPair result{Mat::Zero(mo.D.rows(), mo.D.cols()),
                Mat::Zero(mo.D.rows(), mo.D.cols())};
  for (const auto& local_jk : JK_local) {
    detail::accumulate_operator_symmetric<sk>(local_jk.J, result.J);
    detail::accumulate_operator_symmetric<sk>(local_jk.K, result.K);
  }
  
  occ::timing::stop(occ::timing::category::ints4c2e);
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

  Mat Dnorm = shellblock_norm<sk, kind>(basis, mos[0].D);
  
  // Thread-local storage for multiple MO JK pairs
  occ::parallel::thread_local_storage<std::vector<JKPair>> jkpairs_local(
    [&mos, rows, cols]() {
      return std::vector<JKPair>(mos.size(), 
        JKPair{Mat::Zero(rows, cols), Mat::Zero(rows, cols)});
    }
  );

  auto f = [&mos, &jkpairs_local](const Result &args) {
    auto &local_jkpairs = jkpairs_local.local();
    auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
    auto pr_qs_degree = (args.shell[0] == args.shell[2])
                            ? (args.shell[1] == args.shell[3] ? 1 : 2)
                            : 2;
    auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
    auto scale = pq_degree * rs_degree * pr_qs_degree;

    for (int mo_index = 0; mo_index < mos.size(); mo_index++) {
      const auto &D = mos[mo_index].D;
      auto &J = local_jkpairs[mo_index].J;
      auto &K = local_jkpairs[mo_index].K;

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
  
  occ::timing::start(occ::timing::category::fock);
  evaluate_four_center_tbb<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz, precision);
  occ::timing::stop(occ::timing::category::fock);

  // Combine results from all threads
  std::vector<JKPair> results;
  for (size_t mo_index = 0; mo_index < mos.size(); mo_index++) {
    JKPair result{Mat::Zero(rows, cols), Mat::Zero(rows, cols)};
    for (const auto &local_jkpairs : jkpairs_local) {
      const auto &J = local_jkpairs[mo_index].J;
      const auto &K = local_jkpairs[mo_index].K;
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

  Mat Dnorm = shellblock_norm<sk, kind>(basis, mos[0].D);
  
  // Thread-local storage for multiple MO J matrices
  occ::parallel::thread_local_storage<std::vector<Mat>> js_local(
    [&mos, rows, cols]() {
      return std::vector<Mat>(mos.size(), Mat::Zero(rows, cols));
    }
  );

  auto f = [&mos, &js_local](const Result &args) {
    auto &local_js = js_local.local();
    auto pq_degree = (args.shell[0] == args.shell[1]) ? 1 : 2;
    auto pr_qs_degree = (args.shell[0] == args.shell[2])
                            ? (args.shell[1] == args.shell[3] ? 1 : 2)
                            : 2;
    auto rs_degree = (args.shell[2] == args.shell[3]) ? 1 : 2;
    auto scale = pq_degree * rs_degree * pr_qs_degree;

    for (int mo_index = 0; mo_index < mos.size(); mo_index++) {
      const auto &D = mos[mo_index].D;
      auto &J = local_js[mo_index];

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
  
  occ::timing::start(occ::timing::category::fock);
  evaluate_four_center_tbb<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz, precision);
  occ::timing::stop(occ::timing::category::fock);

  // Combine results from all threads
  std::vector<Mat> results;
  for (size_t mo_index = 0; mo_index < mos.size(); mo_index++) {
    Mat result = Mat::Zero(rows, cols);
    for (const auto &local_js : js_local) {
      const auto &J = local_js[mo_index];
      detail::accumulate_operator_symmetric<sk>(J, result);
    }
    result *= 0.5;
    results.push_back(result);
  }
  return results;
}

} // namespace occ::qm::detail
