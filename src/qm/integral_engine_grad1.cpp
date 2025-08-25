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
  const auto nbf = basis.nbf();
  MatTriple result;
  result.x = Mat::Zero(nbf, nbf);
  result.y = Mat::Zero(nbf, nbf);
  result.z = Mat::Zero(nbf, nbf);

  occ::parallel::thread_local_storage<MatTriple> results_local(
      [nbf]() {
        MatTriple r;
        r.x = Mat::Zero(nbf, nbf);
        r.y = Mat::Zero(nbf, nbf);
        r.z = Mat::Zero(nbf, nbf);
        return r;
      });

  auto f = [&results_local](const Result &args) {
    auto &result = results_local.local();
    const auto num_elements = args.dims[0] * args.dims[1];
    Eigen::Map<const Mat> tmpx(args.buffer, args.dims[0], args.dims[1]),
        tmpy(args.buffer + num_elements, args.dims[0], args.dims[1]),
        tmpz(args.buffer + num_elements * 2, args.dims[0], args.dims[1]);

    result.x.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += tmpx;
    result.y.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += tmpy;
    result.z.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) += tmpz;
  };

  if (shellpairs.size() > 0) {
    // Use TBB for better load balancing with shell pairs
    const auto nsh = basis.size();
    occ::parallel::parallel_for(size_t(0), nsh, [&](size_t p) {
      IntEnv thread_env = env;
      occ::qm::cint::Optimizer opt(thread_env, op, 2, 1);
      auto bufsize = thread_env.buffer_size_1e(op, 1);
      auto buffer = std::make_unique<double[]>(bufsize);
      const auto &first_bf = basis.first_bf();
      
      int bf1 = first_bf[p];
      for (const auto &q : shellpairs[p]) {
        int bf2 = first_bf[q];
        std::array<int, 2> idxs{static_cast<int>(p), static_cast<int>(q)};
        Result args{0,
                    idxs,
                    {bf1, bf2},
                    thread_env.two_center_helper_grad<op, kind>(
                        idxs, opt.optimizer_ptr(), buffer.get(), nullptr),
                    buffer.get()};
        if (args.dims[0] > -1)
          f(args);

        if (p != q) {
          std::array<int, 2> idxs2{static_cast<int>(q), static_cast<int>(p)};
          Result args2{0,
                       idxs2,
                       {bf2, bf1},
                       thread_env.two_center_helper_grad<op, kind>(
                           idxs2, opt.optimizer_ptr(), buffer.get(), nullptr),
                       buffer.get()};
          if (args2.dims[0] > -1)
            f(args2);
        }
      }
    });
  } else {
    // Use TBB for better load balancing without shell pairs
    const auto nsh = basis.size();
    occ::parallel::parallel_for_2d(size_t(0), nsh, size_t(0), nsh, [&](size_t p, size_t q) {
      if (q > p) return;  // Only lower triangle
      
      IntEnv thread_env = env;
      occ::qm::cint::Optimizer opt(thread_env, op, 2, 1);
      auto bufsize = thread_env.buffer_size_1e(op, 1);
      auto buffer = std::make_unique<double[]>(bufsize);
      const auto &first_bf = basis.first_bf();
      
      int bf1 = first_bf[p];
      int bf2 = first_bf[q];
      std::array<int, 2> idxs{static_cast<int>(p), static_cast<int>(q)};
      Result args{0,
                  idxs,
                  {bf1, bf2},
                  thread_env.two_center_helper_grad<op, kind>(
                      idxs, opt.optimizer_ptr(), buffer.get(), nullptr),
                  buffer.get()};
      if (args.dims[0] > -1)
        f(args);
    });
  }

  // Reduce thread-local results
  for (const auto &local_result : results_local) {
    result.x.noalias() += local_result.x;
    result.y.noalias() += local_result.y;
    result.z.noalias() += local_result.z;
  }
  return result;
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
  constexpr Op op = Op::coulomb;

  const auto nbf = basis.nbf();
  Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);
  const auto &D = mo.D;

  // Use TBB-based thread-local storage
  occ::parallel::thread_local_storage<MatTriple> results_local(
    [nbf]() {
      MatTriple r;
      r.x = Mat::Zero(nbf, nbf);
      r.y = Mat::Zero(nbf, nbf);
      r.z = Mat::Zero(nbf, nbf);
      return r;
    }
  );

  auto f = [&D, &results_local](const Result &args) {
    auto &dest = results_local.local();
    detail::four_center_inner_loop(detail::delegate_j_grad<sk>, args, D, dest);
  };

  occ::timing::start(occ::timing::category::fock);
  detail::evaluate_four_center_tbb<op, kind>(
      f, env, basis, shellpairs, Dnorm, Schwarz, precision);
  occ::timing::stop(occ::timing::category::fock);

  // Reduce thread-local results
  MatTriple result;
  result.x = Mat::Zero(nbf, nbf);
  result.y = Mat::Zero(nbf, nbf);
  result.z = Mat::Zero(nbf, nbf);
  
  for (const auto &local_result : results_local) {
    result.x.noalias() += local_result.x;
    result.y.noalias() += local_result.y;
    result.z.noalias() += local_result.z;
  }

  result.scale_by(-2.0);
  return result;
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
  constexpr Op op = Op::coulomb;

  const auto nbf = basis.nbf();
  Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);
  const auto &D = mo.D;

  // Use TBB-based thread-local storage for J and K matrices
  occ::parallel::thread_local_storage<MatTriple> jmats_local(
    [nbf]() {
      MatTriple r;
      r.x = Mat::Zero(nbf, nbf);
      r.y = Mat::Zero(nbf, nbf);
      r.z = Mat::Zero(nbf, nbf);
      return r;
    }
  );
  occ::parallel::thread_local_storage<MatTriple> kmats_local(
    [nbf]() {
      MatTriple r;
      r.x = Mat::Zero(nbf, nbf);
      r.y = Mat::Zero(nbf, nbf);
      r.z = Mat::Zero(nbf, nbf);
      return r;
    }
  );

  auto f = [&D, &jmats_local, &kmats_local](const Result &args) {
    auto &dest_j = jmats_local.local();
    detail::four_center_inner_loop(detail::delegate_j_grad<sk>, args, D, dest_j);

    auto &dest_k = kmats_local.local();
    detail::four_center_inner_loop(detail::delegate_k_grad<sk>, args, D, dest_k);
  };

  occ::timing::start(occ::timing::category::fock);
  detail::evaluate_four_center_tbb<op, kind>(
      f, env, basis, shellpairs, Dnorm, Schwarz, precision);
  occ::timing::stop(occ::timing::category::fock);

  // Reduce thread-local results
  MatTriple jmat, kmat;
  jmat.x = Mat::Zero(nbf, nbf);
  jmat.y = Mat::Zero(nbf, nbf);
  jmat.z = Mat::Zero(nbf, nbf);
  kmat.x = Mat::Zero(nbf, nbf);
  kmat.y = Mat::Zero(nbf, nbf);
  kmat.z = Mat::Zero(nbf, nbf);

  for (const auto &local_j : jmats_local) {
    jmat.x.noalias() += local_j.x;
    jmat.y.noalias() += local_j.y;
    jmat.z.noalias() += local_j.z;
  }
  for (const auto &local_k : kmats_local) {
    kmat.x.noalias() += local_k.x;
    kmat.y.noalias() += local_k.y;
    kmat.z.noalias() += local_k.z;
  }

  jmat.scale_by(-2.0);
  kmat.scale_by(0.5);
  return {jmat, kmat};
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
