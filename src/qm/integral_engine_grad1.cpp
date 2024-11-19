#include "detail/gradient_kernels.h"
#include "detail/jk_grad.h"
#include "detail/kernel_traits.h"
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/gto/gto.h>
#include <occ/qm/integral_engine.h>
#include <unsupported/Eigen/CXX11/Tensor>

namespace occ::qm {

using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = Shell::Kind;
using Op = cint::Operator;

template <Op op, ShellKind kind = ShellKind::Cartesian>
MatTriple one_electron_operator_grad_kernel(const AOBasis &basis, IntEnv &env,
                                            const ShellPairList &shellpairs) {
  using Result = IntegralEngine::IntegralResult<2>;
  auto nthreads = occ::parallel::get_num_threads();
  const auto nbf = basis.nbf();
  MatTriple result;
  result.x = Mat::Zero(nbf, nbf);
  result.y = Mat::Zero(nbf, nbf);
  result.z = Mat::Zero(nbf, nbf);

  std::vector<MatTriple> results;
  results.push_back(result);

  for (size_t i = 1; i < nthreads; i++) {
    results.push_back(results[0]);
  }

  auto f = [&results](const Result &args) {
    auto &result = results[args.thread];
    const auto num_elements = args.dims[0] * args.dims[1];
    Eigen::Map<const Mat> tmpx(args.buffer, args.dims[0], args.dims[1]),
        tmpy(args.buffer + num_elements, args.dims[0], args.dims[1]),
        tmpz(args.buffer + num_elements * 2, args.dims[0], args.dims[1]);

    result.x.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmpx;
    result.y.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmpy;
    result.z.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmpz;
  };

  auto lambda = [&](int thread_id) {
    if (shellpairs.size() > 0) {
      detail::evaluate_two_center_with_shellpairs_grad<op, kind>(
          f, env, basis, shellpairs, thread_id);
    } else {
      detail::evaluate_two_center_grad<op, kind>(f, env, basis, thread_id);
    }
  };
  occ::parallel::parallel_do(lambda);

  for (auto i = 1; i < nthreads; ++i) {
    results[0].x.noalias() += results[i].x;
    results[0].y.noalias() += results[i].y;
    results[0].z.noalias() += results[i].z;
  }
  return results[0];
}

MatTriple
IntegralEngine::one_electron_operator_grad(Op op,
                                           bool use_shellpair_list) const {
  bool spherical = is_spherical();
  constexpr auto Cart = ShellKind::Cartesian;
  constexpr auto Sph = ShellKind::Spherical;
  ShellPairList empty_shellpairs = {};
  const auto &shellpairs = use_shellpair_list ? m_shellpairs : empty_shellpairs;
  switch (op) {
  case Op::overlap: {
    if (spherical) {
      return one_electron_operator_grad_kernel<Op::overlap, Sph>(
          m_aobasis, m_env, shellpairs);
    } else {
      return one_electron_operator_grad_kernel<Op::overlap, Cart>(
          m_aobasis, m_env, shellpairs);
    }
    break;
  }
  case Op::nuclear: {
    if (spherical) {
      return one_electron_operator_grad_kernel<Op::nuclear, Sph>(
          m_aobasis, m_env, shellpairs);
    } else {
      return one_electron_operator_grad_kernel<Op::nuclear, Cart>(
          m_aobasis, m_env, shellpairs);
    }
    break;
  }
  case Op::kinetic: {
    if (spherical) {
      return one_electron_operator_grad_kernel<Op::kinetic, Sph>(
          m_aobasis, m_env, shellpairs);
    } else {
      return one_electron_operator_grad_kernel<Op::kinetic, Cart>(
          m_aobasis, m_env, shellpairs);
    }
    break;
  }
  case Op::coulomb: {
    if (spherical) {
      return one_electron_operator_grad_kernel<Op::coulomb, Sph>(
          m_aobasis, m_env, shellpairs);
    } else {
      return one_electron_operator_grad_kernel<Op::coulomb, Cart>(
          m_aobasis, m_env, shellpairs);
    }
    break;
  }
  default:
    throw std::runtime_error("Invalid operator for two-center integral");
    break;
  }
}

MatTriple
IntegralEngine::rinv_operator_grad_atom(size_t atom_index,
                                        bool use_shellpair_list) const {
  const auto &atoms = m_aobasis.atoms();
  if (atom_index > atoms.size())
    throw std::runtime_error("Invalid atom index for rinv operator");

  bool spherical = is_spherical();
  constexpr auto Cart = ShellKind::Cartesian;
  constexpr auto Sph = ShellKind::Spherical;
  ShellPairList empty_shellpairs = {};
  const auto &shellpairs = use_shellpair_list ? m_shellpairs : empty_shellpairs;

  std::array<double, 3> origin{atoms[atom_index].x, atoms[atom_index].y,
                               atoms[atom_index].z};
  m_env.set_rinv_origin(origin);
  MatTriple result;

  if (spherical) {
    result = one_electron_operator_grad_kernel<Op::rinv, Sph>(m_aobasis, m_env,
                                                              shellpairs);
  } else {
    result = one_electron_operator_grad_kernel<Op::rinv, Cart>(m_aobasis, m_env,
                                                               shellpairs);
  }
  m_env.set_rinv_origin({0.0, 0.0, 0.0});
  return result;
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
MatTriple
coulomb_kernel_grad(cint::IntegralEnvironment &env, const AOBasis &basis,
                    const ShellPairList &shellpairs,
                    const MolecularOrbitals &mo, double precision = 1e-12,
                    const Mat &Schwarz = Mat()) {
  using Result = IntegralEngine::IntegralResult<4>;
  auto nthreads = occ::parallel::get_num_threads();
  constexpr Op op = Op::coulomb;

  const auto nbf = basis.nbf();

  auto results = detail::initialize_result_matrices<sk>(nbf, nthreads);
  Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);

  const auto &D = mo.D;
  auto f = [&D, &results](const Result &args) {
    auto &dest = results[args.thread];
    detail::four_center_inner_loop(detail::delegate_j_grad<sk>, args, D, dest);
  };
  auto lambda = [&](int thread_id) {
    detail::evaluate_four_center_grad<op, kind>(
        f, env, basis, shellpairs, Dnorm, Schwarz, precision, thread_id);
  };
  occ::timing::start(occ::timing::category::fock);
  occ::parallel::parallel_do(lambda);
  occ::timing::stop(occ::timing::category::fock);

  for (size_t i = 1; i < nthreads; i++) {
    results[0].x.noalias() += results[i].x;
    results[0].y.noalias() += results[i].y;
    results[0].z.noalias() += results[i].z;
  }

  results[0].scale_by(-2.0);
  return results[0];
}

MatTriple IntegralEngine::coulomb_grad(SpinorbitalKind sk,
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
      return coulomb_kernel_grad<R, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                         m_precision, Schwarz);
    } else {
      return coulomb_kernel_grad<R, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                          m_precision, Schwarz);
    }
    break;
  case U:
    if (spherical) {
      return coulomb_kernel_grad<U, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                         m_precision, Schwarz);
    } else {
      return coulomb_kernel_grad<U, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                          m_precision, Schwarz);
    }

  case G:
    if (spherical) {
      return coulomb_kernel_grad<G, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                         m_precision, Schwarz);
    } else {
      return coulomb_kernel_grad<G, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                          m_precision, Schwarz);
    }
  }
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
JKTriple coulomb_exchange_kernel_grad(IntEnv &env, const AOBasis &basis,
                                      const ShellPairList &shellpairs,
                                      const MolecularOrbitals &mo,
                                      double precision = 1e-12,
                                      const Mat &Schwarz = Mat()) {
  using Result = IntegralEngine::IntegralResult<4>;
  auto nthreads = occ::parallel::get_num_threads();
  constexpr Op op = Op::coulomb;

  const auto nbf = basis.nbf();

  auto jmats = detail::initialize_result_matrices<sk>(nbf, nthreads);
  auto kmats = detail::initialize_result_matrices<sk>(nbf, nthreads);
  Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);

  const auto &D = mo.D;
  auto f = [&D, &jmats, &kmats](const Result &args) {
    auto &dest_j = jmats[args.thread];
    detail::four_center_inner_loop(detail::delegate_j_grad<sk>, args, D,
                                   dest_j);

    auto &dest_k = kmats[args.thread];
    detail::four_center_inner_loop(detail::delegate_k_grad<sk>, args, D,
                                   dest_k);
  };
  auto lambda = [&](int thread_id) {
    detail::evaluate_four_center_grad<op, kind>(
        f, env, basis, shellpairs, Dnorm, Schwarz, precision, thread_id);
  };
  occ::timing::start(occ::timing::category::fock);
  occ::parallel::parallel_do(lambda);
  occ::timing::stop(occ::timing::category::fock);

  for (size_t i = 1; i < nthreads; i++) {
    jmats[0].x.noalias() += jmats[i].x;
    jmats[0].y.noalias() += jmats[i].y;
    jmats[0].z.noalias() += jmats[i].z;
    kmats[0].x.noalias() += kmats[i].x;
    kmats[0].y.noalias() += kmats[i].y;
    kmats[0].z.noalias() += kmats[i].z;
  }

  jmats[0].scale_by(-2.0);
  kmats[0].scale_by(0.5);
  return {jmats[0], kmats[0]};
}

JKTriple IntegralEngine::coulomb_exchange_grad(SpinorbitalKind sk,
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
      return coulomb_exchange_kernel_grad<R, Sph>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    } else {
      return coulomb_exchange_kernel_grad<R, Cart>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    }
    break;
  case U:
    if (spherical) {
      return coulomb_exchange_kernel_grad<U, Sph>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    } else {
      return coulomb_exchange_kernel_grad<U, Cart>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    }

  case G:
    if (spherical) {
      return coulomb_exchange_kernel_grad<G, Sph>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    } else {
      return coulomb_exchange_kernel_grad<G, Cart>(
          m_env, m_aobasis, m_shellpairs, mo, m_precision, Schwarz);
    }
  }
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
MatTriple
fock_kernel_grad(IntEnv &env, const AOBasis &basis,
                 const ShellPairList &shellpairs, const MolecularOrbitals &mo,
                 double precision = 1e-12, const Mat &Schwarz = Mat()) {
  auto [J, K] = coulomb_exchange_kernel_grad<sk, kind>(env, basis, shellpairs,
                                                       mo, precision, Schwarz);
  J.x -= K.x;
  J.y -= K.y;
  J.z -= K.z;
  return J;
}

MatTriple IntegralEngine::fock_operator_grad(SpinorbitalKind sk,
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
      return fock_kernel_grad<R, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                      m_precision, Schwarz);
    } else {
      return fock_kernel_grad<R, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                       m_precision, Schwarz);
    }
    break;
  case U:
    if (spherical) {
      return fock_kernel_grad<U, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                      m_precision, Schwarz);
    } else {
      return fock_kernel_grad<U, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                       m_precision, Schwarz);
    }

  case G:
    if (spherical) {
      return fock_kernel_grad<G, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                      m_precision, Schwarz);
    } else {
      return fock_kernel_grad<G, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                       m_precision, Schwarz);
    }
  }
}

} // namespace occ::qm
