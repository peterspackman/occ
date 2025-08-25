#pragma once
#include "kernel_traits.h"

#include "jk.h"
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

// TBB-optimized version with shell pair screening
template <Op op, ShellKind kind, typename Lambda>
void evaluate_four_center_tbb_with_shellpairs(
    Lambda &f, cint::IntegralEnvironment &env, const AOBasis &basis,
    const ShellPairList &shellpairs, const Mat &Dnorm = Mat(),
    const Mat &Schwarz = Mat(), double precision = 1e-12) {
  using Result = IntegralEngine::IntegralResult<4>;

  const auto &first_bf = basis.first_bf();
  const auto do_schwarz_screen = (Schwarz.cols() != 0 && Schwarz.rows() != 0) &&
                                 (Dnorm.cols() != 0 && Dnorm.rows() != 0);

  // Thread-local storage for optimizer and buffer
  occ::parallel::thread_local_storage<occ::qm::cint::Optimizer> opt_local(
      [&env]() { return occ::qm::cint::Optimizer(env, Op::coulomb, 4); });

  occ::parallel::thread_local_storage<std::unique_ptr<double[]>> buffer_local(
      [&env]() { return std::make_unique<double[]>(env.buffer_size_2e()); });

  const size_t nsh = basis.size();
  const size_t num_pq_pairs = (nsh * (nsh + 1)) / 2;

  occ::parallel::parallel_for(size_t(0), num_pq_pairs, [&](size_t pq_idx) {
    auto &opt = opt_local.local();
    auto &buffer = buffer_local.local();

    // Convert linear index to triangular (p,q) with q ≤ p
    size_t p = static_cast<size_t>((std::sqrt(8.0 * pq_idx + 1.0) - 1.0) / 2.0);
    size_t q = pq_idx - p * (p + 1) / 2;

    // Check if (p,q) is in shell pair list
    if (p >= shellpairs.size() ||
        !std::binary_search(shellpairs[p].begin(), shellpairs[p].end(), q)) {
      return; // Not a significant shell pair
    }

    const double norm_pq = do_schwarz_screen ? Dnorm(p, q) : 0.0;

    // Loop over (r,s) pairs with r ≤ p and appropriate s constraints
    for (size_t r = 0; r <= p; r++) {
      const auto s_max = (p == r) ? q : r;
      const double norm_pqr =
          do_schwarz_screen ? max_of(Dnorm(p, r), Dnorm(q, r), norm_pq) : 0.0;

      if (r >= shellpairs.size())
        continue;

      const auto &rlist = shellpairs[r];
      for (const auto s : rlist) {
        if (s > s_max)
          break;

        // Apply Schwarz screening
        if (do_schwarz_screen) {
          const double norm_pqrs =
              max_of(Dnorm(p, s), Dnorm(q, s), Dnorm(r, s), norm_pqr);
          const double schwarz_pq_rs = Schwarz(p, q) * Schwarz(r, s);
          if (norm_pqrs * schwarz_pq_rs < precision) {
            continue;
          }
        }

        std::array<int, 4> shell_idx = {
            static_cast<int>(p), static_cast<int>(q), static_cast<int>(r),
            static_cast<int>(s)};
        std::array<int, 4> bf = {first_bf[p], first_bf[q], first_bf[r],
                                 first_bf[s]};

        Result args{0, shell_idx, bf,
                    env.four_center_helper<op, kind>(
                        shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr),
                    buffer.get()};

        if (args.dims[0] > -1) {
          f(args);
        }
      }
    }
  });
}

// TBB-optimized version without shell pair screening
template <Op op, ShellKind kind, typename Lambda>
void evaluate_four_center_tbb_no_shellpairs(Lambda &f,
                                            cint::IntegralEnvironment &env,
                                            const AOBasis &basis,
                                            const Mat &Dnorm = Mat(),
                                            const Mat &Schwarz = Mat(),
                                            double precision = 1e-12) {
  using Result = IntegralEngine::IntegralResult<4>;

  const auto &first_bf = basis.first_bf();
  const auto do_schwarz_screen = (Schwarz.cols() != 0 && Schwarz.rows() != 0) &&
                                 (Dnorm.cols() != 0 && Dnorm.rows() != 0);

  // Thread-local storage for optimizer and buffer
  occ::parallel::thread_local_storage<occ::qm::cint::Optimizer> opt_local(
      [&env]() { return occ::qm::cint::Optimizer(env, Op::coulomb, 4); });

  occ::parallel::thread_local_storage<std::unique_ptr<double[]>> buffer_local(
      [&env]() { return std::make_unique<double[]>(env.buffer_size_2e()); });

  const size_t nsh = basis.size();
  const size_t num_pq_pairs = (nsh * (nsh + 1)) / 2;

  occ::parallel::parallel_for(size_t(0), num_pq_pairs, [&](size_t pq_idx) {
    auto &opt = opt_local.local();
    auto &buffer = buffer_local.local();

    // Convert linear index to triangular (p,q) with q ≤ p
    size_t p = static_cast<size_t>((std::sqrt(8.0 * pq_idx + 1.0) - 1.0) / 2.0);
    size_t q = pq_idx - p * (p + 1) / 2;

    const double norm_pq = do_schwarz_screen ? Dnorm(p, q) : 0.0;

    // Loop over (r,s) pairs with r ≤ p and appropriate s constraints
    for (size_t r = 0; r <= p; r++) {
      const auto s_max = (p == r) ? q : r;
      const double norm_pqr =
          do_schwarz_screen ? max_of(Dnorm(p, r), Dnorm(q, r), norm_pq) : 0.0;

      // No shell pair screening, iterate over all s ≤ s_max
      for (size_t s = 0; s <= s_max; ++s) {
        // Apply Schwarz screening
        if (do_schwarz_screen) {
          const double norm_pqrs =
              max_of(Dnorm(p, s), Dnorm(q, s), Dnorm(r, s), norm_pqr);
          const double schwarz_pq_rs = Schwarz(p, q) * Schwarz(r, s);
          if (norm_pqrs * schwarz_pq_rs < precision) {
            continue;
          }
        }

        std::array<int, 4> shell_idx = {
            static_cast<int>(p), static_cast<int>(q), static_cast<int>(r),
            static_cast<int>(s)};
        std::array<int, 4> bf = {first_bf[p], first_bf[q], first_bf[r],
                                 first_bf[s]};

        Result args{0, shell_idx, bf,
                    env.four_center_helper<op, kind>(
                        shell_idx, opt.optimizer_ptr(), buffer.get(), nullptr),
                    buffer.get()};

        if (args.dims[0] > -1) {
          f(args);
        }
      }
    }
  });
}

// Dispatch function that chooses the right implementation
template <Op op, ShellKind kind, typename Lambda>
void evaluate_four_center_tbb(Lambda &f, cint::IntegralEnvironment &env,
                              const AOBasis &basis,
                              const ShellPairList &shellpairs,
                              const Mat &Dnorm = Mat(),
                              const Mat &Schwarz = Mat(),
                              double precision = 1e-12) {
  // Check if we have valid shell pairs data
  bool has_valid_shellpairs =
      (shellpairs.size() > 0 && shellpairs.size() >= basis.size());

  if (has_valid_shellpairs) {
    // Additional validation: check that shell pairs are not empty
    bool has_nonempty_pairs = false;
    for (size_t i = 0; i < std::min(shellpairs.size(), size_t(5)); ++i) {
      if (!shellpairs[i].empty()) {
        has_nonempty_pairs = true;
        break;
      }
    }

    if (has_nonempty_pairs) {
      evaluate_four_center_tbb_with_shellpairs<op, kind>(
          f, env, basis, shellpairs, Dnorm, Schwarz, precision);
    } else {
      evaluate_four_center_tbb_no_shellpairs<op, kind>(f, env, basis, Dnorm,
                                                       Schwarz, precision);
    }
  } else {
    evaluate_four_center_tbb_no_shellpairs<op, kind>(f, env, basis, Dnorm,
                                                     Schwarz, precision);
  }
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
      [&mo]() { return Mat::Zero(mo.D.rows(), mo.D.cols()); });

  auto f = [&D, &F_local](const Result &args) {
    auto &F = F_local.local();

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

  evaluate_four_center_tbb<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz,
                                     precision);

  // Combine all thread-local matrices
  Mat F = Mat::Zero(mo.D.rows(), mo.D.cols());
  for (const auto &local_F : F_local) {
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
      [&mo]() { return Mat::Zero(mo.D.rows(), mo.D.cols()); });

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

  evaluate_four_center_tbb<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz,
                                     precision);

  // Combine all thread-local matrices
  Mat J = Mat::Zero(mo.D.rows(), mo.D.cols());
  for (const auto &local_J : J_local) {
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
  occ::parallel::thread_local_storage<JKPair> JK_local([&mo]() {
    return JKPair{Mat::Zero(mo.D.rows(), mo.D.cols()),
                  Mat::Zero(mo.D.rows(), mo.D.cols())};
  });

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

  evaluate_four_center_tbb<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz,
                                     precision);

  // Combine all thread-local matrices
  JKPair result{Mat::Zero(mo.D.rows(), mo.D.cols()),
                Mat::Zero(mo.D.rows(), mo.D.cols())};
  for (const auto &local_jk : JK_local) {
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
        return std::vector<JKPair>(
            mos.size(), JKPair{Mat::Zero(rows, cols), Mat::Zero(rows, cols)});
      });

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
  evaluate_four_center_tbb<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz,
                                     precision);
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
      });

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
  evaluate_four_center_tbb<op, kind>(f, env, basis, shellpairs, Dnorm, Schwarz,
                                     precision);
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
