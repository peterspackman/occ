#include "detail/four_center_kernels.h"


namespace occ::qm {

using ShellList = std::vector<Shell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = Shell::Kind;
using Op = cint::Operator;
using occ::core::PointCharge;

Mat IntegralEngine::fock_operator(SpinorbitalKind sk,
                                  const MolecularOrbitals &mo,
                                  const Mat &Schwarz) const {
  constexpr auto R = SpinorbitalKind::Restricted;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  constexpr auto Sph = ShellKind::Spherical;
  constexpr auto Cart = ShellKind::Cartesian;
  bool spherical = is_spherical();
  switch (sk) {
  default:
  case R:
    if (spherical) {
      return detail::fock_operator_kernel<R, Sph>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    } else {
      return detail::fock_operator_kernel<R, Cart>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    }
    break;
  case U:
    if (spherical) {
      return detail::fock_operator_kernel<U, Sph>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    } else {
      return detail::fock_operator_kernel<U, Cart>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    }

  case G:
    if (spherical) {
      return detail::fock_operator_kernel<G, Sph>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    } else {
      return detail::fock_operator_kernel<G, Cart>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    }
  }
}

Mat IntegralEngine::coulomb(SpinorbitalKind sk, const MolecularOrbitals &mo,
                            const Mat &Schwarz) const {
  constexpr auto R = SpinorbitalKind::Restricted;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  constexpr auto Sph = ShellKind::Spherical;
  constexpr auto Cart = ShellKind::Cartesian;
  bool spherical = is_spherical();
  switch (sk) {
  default:
  case R:
    if (spherical) {
      return detail::coulomb_kernel<R, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                            m_precision, Schwarz);
    } else {
      return detail::coulomb_kernel<R, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                             m_precision, Schwarz);
    }
    break;
  case U:
    if (spherical) {
      return detail::coulomb_kernel<U, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                            m_precision, Schwarz);
    } else {
      return detail::coulomb_kernel<U, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                             m_precision, Schwarz);
    }

  case G:
    if (spherical) {
      return detail::coulomb_kernel<G, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                            m_precision, Schwarz);
    } else {
      return detail::coulomb_kernel<G, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                             m_precision, Schwarz);
    }
  }
}

JKPair IntegralEngine::coulomb_and_exchange(SpinorbitalKind sk,
                                            const MolecularOrbitals &mo,
                                            const Mat &Schwarz) const {
  constexpr auto R = SpinorbitalKind::Restricted;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  constexpr auto Sph = ShellKind::Spherical;
  constexpr auto Cart = ShellKind::Cartesian;
  bool spherical = is_spherical();
  switch (sk) {
  default:
  case R:
    if (spherical) {
      return detail::coulomb_and_exchange_kernel<R, Sph>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    } else {
      return detail::coulomb_and_exchange_kernel<R, Cart>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    }
    break;
  case U:
    if (spherical) {
      return detail::coulomb_and_exchange_kernel<U, Sph>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    } else {
      return detail::coulomb_and_exchange_kernel<U, Cart>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    }

  case G:
    if (spherical) {
      return detail::coulomb_and_exchange_kernel<G, Sph>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    } else {
      return detail::coulomb_and_exchange_kernel<G, Cart>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    }
  }
}

std::vector<Mat>
IntegralEngine::coulomb_list(SpinorbitalKind sk,
                             const std::vector<MolecularOrbitals> &mos,
                             const Mat &Schwarz) const {
  constexpr auto R = SpinorbitalKind::Restricted;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  constexpr auto Sph = ShellKind::Spherical;
  constexpr auto Cart = ShellKind::Cartesian;
  bool spherical = is_spherical();
  switch (sk) {
  default:
  case R:
    if (spherical) {
      return detail::coulomb_kernel_list<R, Sph>(m_env, m_aobasis, m_shellpairs,
                                                 mos, m_precision, Schwarz);
    } else {
      return detail::coulomb_kernel_list<R, Cart>(
          m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
    }
    break;
  case U:
    if (spherical) {
      return detail::coulomb_kernel_list<U, Sph>(m_env, m_aobasis, m_shellpairs,
                                                 mos, m_precision, Schwarz);
    } else {
      return detail::coulomb_kernel_list<U, Cart>(
          m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
    }

  case G:
    if (spherical) {
      return detail::coulomb_kernel_list<G, Sph>(m_env, m_aobasis, m_shellpairs,
                                                 mos, m_precision, Schwarz);
    } else {
      return detail::coulomb_kernel_list<G, Cart>(
          m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
    }
  }
}

std::vector<JKPair> IntegralEngine::coulomb_and_exchange_list(
    SpinorbitalKind sk, const std::vector<MolecularOrbitals> &mos,
    const Mat &Schwarz) const {
  constexpr auto R = SpinorbitalKind::Restricted;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  constexpr auto Sph = ShellKind::Spherical;
  constexpr auto Cart = ShellKind::Cartesian;
  bool spherical = is_spherical();
  switch (sk) {
  default:
  case R:
    if (spherical) {
      return detail::coulomb_and_exchange_kernel_list<R, Sph>(
          m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
    } else {
      return detail::coulomb_and_exchange_kernel_list<R, Cart>(
          m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
    }
    break;
  case U:
    if (spherical) {
      return detail::coulomb_and_exchange_kernel_list<U, Sph>(
          m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
    } else {
      return detail::coulomb_and_exchange_kernel_list<U, Cart>(
          m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
    }

  case G:
    if (spherical) {
      return detail::coulomb_and_exchange_kernel_list<G, Sph>(
          m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
    } else {
      return detail::coulomb_and_exchange_kernel_list<G, Cart>(
          m_env, m_aobasis, m_shellpairs, mos, m_precision, Schwarz);
    }
  }
}

Mat IntegralEngine::fock_operator_mixed_basis(const Mat &D, const AOBasis &D_bs,
                                              bool is_shell_diagonal) {
  set_auxiliary_basis(D_bs.shells(), false);
  constexpr Op op = Op::coulomb;

  constexpr auto Sph = ShellKind::Spherical;
  constexpr auto Cart = ShellKind::Cartesian;
  bool spherical = is_spherical();
  const int nbf = m_aobasis.nbf();
  const int nsh = m_aobasis.size();
  const int nbf_aux = m_auxbasis.nbf();
  const int nsh_aux = m_auxbasis.size();
  assert(D.cols() == D.rows() && D.cols() == nbf_aux);

  // Thread-local storage for Fock matrices
  occ::parallel::thread_local_storage<Mat> F_local(Mat::Zero(nbf, nbf));

  // Thread-local storage for optimizer and buffer
  occ::parallel::thread_local_storage<occ::qm::cint::Optimizer> opt_local(
      [this]() { return occ::qm::cint::Optimizer(m_env, Op::coulomb, 4); });

  occ::parallel::thread_local_storage<std::unique_ptr<double[]>> buffer_local(
      [this]() { return std::make_unique<double[]>(m_env.buffer_size_2e()); });

  auto shell2bf = m_aobasis.first_bf();
  auto shell2bf_D = m_auxbasis.first_bf();

  // Parallelize over unique shell pairs (s1,s2) with s2 ≤ s1
  size_t num_pairs = (size_t(nsh) * (nsh + 1)) / 2;

  occ::parallel::parallel_for(size_t(0), num_pairs, [&](size_t pair_idx) {
    auto &F = F_local.local();
    auto &opt = opt_local.local();
    auto &buffer = buffer_local.local();

    // Convert linear index to triangular (s1,s2) with s2 ≤ s1
    // pair_idx = s1*(s1+1)/2 + s2, so solve quadratic for s1
    size_t s1 =
        static_cast<size_t>((std::sqrt(8.0 * pair_idx + 1.0) - 1.0) / 2.0);
    size_t s2 = pair_idx - s1 * (s1 + 1) / 2;

    int bf1_first = shell2bf[s1];
    int bf2_first = shell2bf[s2];
    int n1 = m_aobasis[s1].size();
    int n2 = m_aobasis[s2].size();
    double s12_deg = (s1 == s2) ? 1.0 : 2.0;

    for (int s3 = 0; s3 < nsh_aux; ++s3) {
      int bf3_first = shell2bf_D[s3];
      int n3 = D_bs[s3].size();

      int s4_begin = is_shell_diagonal ? s3 : 0;
      int s4_fence = is_shell_diagonal ? s3 + 1 : nsh_aux;

      for (int s4 = s4_begin; s4 != s4_fence; ++s4) {
        int bf4_first = shell2bf_D[s4];
        int n4 = D_bs[s4].size();
        double s34_deg = (s3 == s4) ? 1.0 : 2.0;

        std::array<int, 4> dims;

        // First integral: (s1 s2 | s3 s4)
        if (s3 >= s4) {
          double s1234_deg = s12_deg * s34_deg;
          std::array<int, 4> idxs{static_cast<int>(s1), static_cast<int>(s2),
                                  s3 + nsh, s4 + nsh};
          if (spherical) {
            dims = m_env.four_center_helper<op, Sph>(idxs, opt.optimizer_ptr(),
                                                     buffer.get(), nullptr);
          } else {
            dims = m_env.four_center_helper<op, Cart>(idxs, opt.optimizer_ptr(),
                                                      buffer.get(), nullptr);
          }

          if (dims[0] >= 0) {
            const auto *buf_1234 = buffer.get();
            for (auto f4 = 0, f1234 = 0; f4 != n4; ++f4) {
              const auto bf4 = f4 + bf4_first;
              for (auto f3 = 0; f3 != n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for (auto f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (auto f1 = 0; f1 != n1; ++f1, ++f1234) {
                    const auto bf1 = f1 + bf1_first;
                    const auto value = buf_1234[f1234];
                    const auto value_scal_by_deg = value * s1234_deg;
                    F(bf1, bf2) += 2.0 * D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }
            }
          }
        }

        // Second integral: (s1 s3 | s2 s4)
        std::array<int, 4> idxs{static_cast<int>(s1), s3 + nsh,
                                static_cast<int>(s2), s4 + nsh};
        if (spherical) {
          dims = m_env.four_center_helper<op, Sph>(idxs, opt.optimizer_ptr(),
                                                   buffer.get(), nullptr);
        } else {
          dims = m_env.four_center_helper<op, Cart>(idxs, opt.optimizer_ptr(),
                                                    buffer.get(), nullptr);
        }

        if (dims[0] >= 0) {
          const auto *buf_1324 = buffer.get();
          for (auto f4 = 0, f1324 = 0; f4 != n4; ++f4) {
            const auto bf4 = f4 + bf4_first;
            for (auto f2 = 0; f2 != n2; ++f2) {
              const auto bf2 = f2 + bf2_first;
              for (auto f3 = 0; f3 != n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for (auto f1 = 0; f1 != n1; ++f1, ++f1324) {
                  const auto bf1 = f1 + bf1_first;
                  const auto value = buf_1324[f1324];
                  const auto value_scal_by_deg = value * s12_deg;
                  F(bf1, bf2) -= D(bf3, bf4) * value_scal_by_deg;
                }
              }
            }
          }
        }
      }
    }
  });

  // Reduce results from all threads
  Mat F_result = Mat::Zero(nbf, nbf);
  for (const auto &F_thread : F_local) {
    F_result += F_thread;
  }

  clear_auxiliary_basis();
  // symmetrize the result and return
  return 0.5 * (F_result + F_result.transpose());
}

Eigen::Tensor<double, 4>
IntegralEngine::four_center_integrals_tensor(const Mat &Schwarz) const {
  using Result = IntegralEngine::IntegralResult<4>;
  const size_t n_ao = nbf();
  constexpr auto op = cint::Operator::coulomb;
  auto nthreads = occ::parallel::get_num_threads();

  occ::log::info(
      "Computing AO integrals using parallel dense tensor with {} threads",
      nthreads);
  occ::log::info("AO basis size: {} functions", n_ao);
  occ::log::info(
      "Using 8-fold symmetry storage - storing only unique integrals");
  if (Schwarz.size() > 0) {
    occ::log::info("Using Schwarz screening for shell pair screening");
  }

  // Create the result tensor directly
  Eigen::Tensor<double, 4> result(n_ao, n_ao, n_ao, n_ao);
  result.setZero();

  // Lambda function to process shell quartets and store only unique integrals
  auto f = [&result, n_ao](const Result &args) {
    // Extract integrals from buffer and store only canonical form
    for (auto f3 = 0, f0123 = 0; f3 != args.dims[3]; ++f3) {
      const auto bf3 = f3 + args.bf[3];
      for (auto f2 = 0; f2 != args.dims[2]; ++f2) {
        const auto bf2 = f2 + args.bf[2];
        for (auto f1 = 0; f1 != args.dims[1]; ++f1) {
          const auto bf1 = f1 + args.bf[1];
          for (auto f0 = 0; f0 != args.dims[0]; ++f0, ++f0123) {
            const auto bf0 = f0 + args.bf[0];
            const auto value = args.buffer[f0123];

            // Store only if significant and in canonical form
            if (std::abs(value) > 1e-12) {
              // Determine canonical ordering for 8-fold symmetry
              // Store in form where: μ <= ν and ρ <= σ and (μν) <= (ρσ)
              size_t mu = std::min(bf0, bf1);
              size_t nu = std::max(bf0, bf1);
              size_t rho = std::min(bf2, bf3);
              size_t sigma = std::max(bf2, bf3);

              // Ensure (μν) <= (ρσ) by comparing composite indices
              size_t munu = mu * n_ao + nu;
              size_t rhosigma = rho * n_ao + sigma;

              if (munu <= rhosigma) {
                result(mu, nu, rho, sigma) = value;
              } else {
                result(rho, sigma, mu, nu) = value;
              }
            }
          }
        }
      }
    }
  };

  // Execute parallel integral computation using TBB
  occ::timing::start(occ::timing::category::ints4c2e);

  // Compute density norm matrix for screening (empty for now)
  Mat Dnorm;

  if (is_spherical()) {
    detail::evaluate_four_center_tbb<op, Shell::Kind::Spherical>(
        f, m_env, m_aobasis, m_shellpairs, Dnorm, Schwarz, m_precision);
  } else {
    detail::evaluate_four_center_tbb<op, Shell::Kind::Cartesian>(
        f, m_env, m_aobasis, m_shellpairs, Dnorm, Schwarz, m_precision);
  }

  occ::timing::stop(occ::timing::category::ints4c2e);

  occ::log::info("AO integral tensor computation completed");

  return result;
}

double IntegralEngine::get_integral_8fold_symmetry(
    const Eigen::Tensor<double, 4> &tensor, size_t i, size_t j, size_t k,
    size_t l, size_t n_ao) {
  // Map indices to canonical form using 8-fold symmetry
  // Canonical form: μ <= ν and ρ <= σ and (μν) <= (ρσ)
  size_t mu = std::min(i, j);
  size_t nu = std::max(i, j);
  size_t rho = std::min(k, l);
  size_t sigma = std::max(k, l);

  // Ensure (μν) <= (ρσ) by comparing composite indices
  size_t munu = mu * n_ao + nu;
  size_t rhosigma = rho * n_ao + sigma;

  if (munu <= rhosigma) {
    return tensor(mu, nu, rho, sigma);
  } else {
    return tensor(rho, sigma, mu, nu);
  }
}

} // namespace occ::qm
