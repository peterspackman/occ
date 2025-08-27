#include "detail/hessian_kernels.h"
#include "detail/jk_hess.h"
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
MatSix one_electron_operator_hess_kernel(const AOBasis &basis, IntEnv &env,
                                         const ShellPairList &shellpairs) {
  using Result = IntegralEngine::IntegralResult<2>;
  auto nthreads = occ::parallel::get_num_threads();
  const auto nbf = basis.nbf();
  MatSix result;
  result.xx = Mat::Zero(nbf, nbf);
  result.yy = Mat::Zero(nbf, nbf);
  result.zz = Mat::Zero(nbf, nbf);
  result.xy = Mat::Zero(nbf, nbf);
  result.xz = Mat::Zero(nbf, nbf);
  result.yz = Mat::Zero(nbf, nbf);

  // Use TBB thread-local storage instead of thread-indexed arrays
  occ::parallel::thread_local_storage<MatSix> tl_results([&result]() { return result; });

  auto f = [&tl_results](const Result &args) {
    auto &local_result = tl_results.local();
    const auto num_elements = args.dims[0] * args.dims[1];
    
    // MatSix has 6 components: xx, yy, zz, xy, xz, yz
    Eigen::Map<const Mat> tmpxx(args.buffer, args.dims[0], args.dims[1]),
        tmpyy(args.buffer + num_elements, args.dims[0], args.dims[1]),
        tmpzz(args.buffer + num_elements * 2, args.dims[0], args.dims[1]),
        tmpxy(args.buffer + num_elements * 3, args.dims[0], args.dims[1]),
        tmpxz(args.buffer + num_elements * 4, args.dims[0], args.dims[1]),
        tmpyz(args.buffer + num_elements * 5, args.dims[0], args.dims[1]);

    local_result.xx.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmpxx;
    local_result.yy.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmpyy;
    local_result.zz.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmpzz;
    local_result.xy.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmpxy;
    local_result.xz.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmpxz;
    local_result.yz.block(args.bf[0], args.bf[1], args.dims[0], args.dims[1]) = tmpyz;
  };

  occ::parallel::parallel_for(size_t(0), size_t(nthreads), [&](size_t thread_id) {
    if (shellpairs.size() > 0) {
      detail::evaluate_two_center_with_shellpairs_hess<op, kind>(
          f, env, basis, shellpairs, int(thread_id));
    } else {
      detail::evaluate_two_center_hess<op, kind>(f, env, basis, int(thread_id));
    }
  });

  // Reduce thread-local results
  for (const auto &local_result : tl_results) {
    result.xx += local_result.xx;
    result.yy += local_result.yy;
    result.zz += local_result.zz;
    result.xy += local_result.xy;
    result.xz += local_result.xz;
    result.yz += local_result.yz;
  }
  return result;
}

MatSix
IntegralEngine::one_electron_operator_hess(Op op,
                                           bool use_shellpair_list) const {
  bool spherical = is_spherical();
  constexpr auto Cart = ShellKind::Cartesian;
  constexpr auto Sph = ShellKind::Spherical;
  ShellPairList empty_shellpairs = {};
  const auto &shellpairs = use_shellpair_list ? m_shellpairs : empty_shellpairs;
  switch (op) {
  case Op::overlap: {
    if (spherical) {
      return one_electron_operator_hess_kernel<Op::overlap, Sph>(
          m_aobasis, m_env, shellpairs);
    } else {
      return one_electron_operator_hess_kernel<Op::overlap, Cart>(
          m_aobasis, m_env, shellpairs);
    }
    break;
  }
  case Op::nuclear: {
    if (spherical) {
      return one_electron_operator_hess_kernel<Op::nuclear, Sph>(
          m_aobasis, m_env, shellpairs);
    } else {
      return one_electron_operator_hess_kernel<Op::nuclear, Cart>(
          m_aobasis, m_env, shellpairs);
    }
    break;
  }
  case Op::kinetic: {
    if (spherical) {
      return one_electron_operator_hess_kernel<Op::kinetic, Sph>(
          m_aobasis, m_env, shellpairs);
    } else {
      return one_electron_operator_hess_kernel<Op::kinetic, Cart>(
          m_aobasis, m_env, shellpairs);
    }
    break;
  }
  default:
    throw std::runtime_error("Invalid operator for two-center integral");
    break;
  }
}

MatSix
IntegralEngine::rinv_operator_hess_atom(size_t atom_index,
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
  MatSix result;

  if (spherical) {
    result = one_electron_operator_hess_kernel<Op::rinv, Sph>(m_aobasis, m_env,
                                                              shellpairs);
  } else {
    result = one_electron_operator_hess_kernel<Op::rinv, Cart>(m_aobasis, m_env,
                                                               shellpairs);
  }
  m_env.set_rinv_origin({0.0, 0.0, 0.0});
  return result;
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
MatSix
coulomb_kernel_hess(cint::IntegralEnvironment &env, const AOBasis &basis,
                    const ShellPairList &shellpairs,
                    const MolecularOrbitals &mo, double precision = 1e-12,
                    const Mat &Schwarz = Mat()) {
  using Result = IntegralEngine::IntegralResult<4>;
  auto nthreads = occ::parallel::get_num_threads();
  constexpr Op op = Op::coulomb;

  const auto nbf = basis.nbf();

  // Use TBB thread-local storage instead of thread-indexed arrays
  auto zero_result = detail::initialize_result_matrices_hess<sk>(nbf, 1)[0];
  occ::parallel::thread_local_storage<MatSix> tl_results([&zero_result]() { return zero_result; });
  
  Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);

  const auto &D = mo.D;
  auto f = [&D, &tl_results](const Result &args) {
    auto &dest = tl_results.local();
    detail::four_center_inner_loop_hess(detail::delegate_j_hess<sk>, args, D, dest);
  };
  
  occ::timing::start(occ::timing::category::fock);
  occ::parallel::parallel_for(size_t(0), size_t(nthreads), [&](size_t thread_id) {
    detail::evaluate_four_center_hess<op, kind>(
        f, env, basis, shellpairs, Dnorm, Schwarz, precision, int(thread_id));
  });
  occ::timing::stop(occ::timing::category::fock);

  // Reduce thread-local results
  for (const auto &local_result : tl_results) {
    zero_result.xx += local_result.xx;
    zero_result.yy += local_result.yy;
    zero_result.zz += local_result.zz;
    zero_result.xy += local_result.xy;
    zero_result.xz += local_result.xz;
    zero_result.yz += local_result.yz;
  }

  zero_result.scale_by(-2.0);
  return zero_result;
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
MatSix
exchange_kernel_hess(cint::IntegralEnvironment &env, const AOBasis &basis,
                     const ShellPairList &shellpairs,
                     const MolecularOrbitals &mo, double precision = 1e-12,
                     const Mat &Schwarz = Mat()) {
  using Result = IntegralEngine::IntegralResult<4>;
  auto nthreads = occ::parallel::get_num_threads();
  constexpr Op op = Op::coulomb;

  const auto nbf = basis.nbf();

  // Use TBB thread-local storage instead of thread-indexed arrays
  auto zero_result = detail::initialize_result_matrices_hess<sk>(nbf, 1)[0];
  occ::parallel::thread_local_storage<MatSix> tl_results([&zero_result]() { return zero_result; });
  
  Mat Dnorm = shellblock_norm<sk, kind>(basis, mo.D);

  const auto &D = mo.D;
  auto f = [&D, &tl_results](const Result &args) {
    auto &dest = tl_results.local();
    detail::four_center_inner_loop_hess(detail::delegate_k_hess<sk>, args, D, dest);
  };
  
  occ::timing::start(occ::timing::category::fock);
  occ::parallel::parallel_for(size_t(0), size_t(nthreads), [&](size_t thread_id) {
    detail::evaluate_four_center_hess<op, kind>(
        f, env, basis, shellpairs, Dnorm, Schwarz, precision, int(thread_id));
  });
  occ::timing::stop(occ::timing::category::fock);

  // Reduce thread-local results
  for (const auto &local_result : tl_results) {
    zero_result.xx += local_result.xx;
    zero_result.yy += local_result.yy;
    zero_result.zz += local_result.zz;
    zero_result.xy += local_result.xy;
    zero_result.xz += local_result.xz;
    zero_result.yz += local_result.yz;
  }

  // Exchange has positive sign (opposite to Coulomb)
  zero_result.scale_by(2.0);
  return zero_result;
}

MatSix IntegralEngine::exchange_hess(SpinorbitalKind sk,
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
      return exchange_kernel_hess<R, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                        m_precision, Schwarz);
    } else {
      return exchange_kernel_hess<R, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                         m_precision, Schwarz);
    }
    break;
  case U:
    if (spherical) {
      return exchange_kernel_hess<U, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                        m_precision, Schwarz);
    } else {
      return exchange_kernel_hess<U, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                         m_precision, Schwarz);
    }

  case G:
    if (spherical) {
      return exchange_kernel_hess<G, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                        m_precision, Schwarz);
    } else {
      return exchange_kernel_hess<G, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                         m_precision, Schwarz);
    }
  }
}

MatSix IntegralEngine::coulomb_hess(SpinorbitalKind sk,
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
      return coulomb_kernel_hess<R, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                        m_precision, Schwarz);
    } else {
      return coulomb_kernel_hess<R, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                         m_precision, Schwarz);
    }
    break;
  case U:
    if (spherical) {
      return coulomb_kernel_hess<U, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                        m_precision, Schwarz);
    } else {
      return coulomb_kernel_hess<U, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                         m_precision, Schwarz);
    }

  case G:
    if (spherical) {
      return coulomb_kernel_hess<G, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                        m_precision, Schwarz);
    } else {
      return coulomb_kernel_hess<G, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                         m_precision, Schwarz);
    }
  }
}

template <SpinorbitalKind sk, ShellKind kind = ShellKind::Cartesian>
MatSix
fock_kernel_hess(IntEnv &env, const AOBasis &basis,
                 const ShellPairList &shellpairs, const MolecularOrbitals &mo,
                 double precision = 1e-12, const Mat &Schwarz = Mat()) {
  // Compute Coulomb and Exchange contributions
  auto coulomb = coulomb_kernel_hess<sk, kind>(env, basis, shellpairs, mo, precision, Schwarz);
  auto exchange = exchange_kernel_hess<sk, kind>(env, basis, shellpairs, mo, precision, Schwarz);
  
  // Combine J and K contributions: F = J - K (for RHF)
  coulomb.xx += exchange.xx;
  coulomb.yy += exchange.yy; 
  coulomb.zz += exchange.zz;
  coulomb.xy += exchange.xy;
  coulomb.xz += exchange.xz;
  coulomb.yz += exchange.yz;
  
  return coulomb;
}

MatSix IntegralEngine::fock_operator_hess(SpinorbitalKind sk,
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
      return fock_kernel_hess<R, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                     m_precision, Schwarz);
    } else {
      return fock_kernel_hess<R, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                      m_precision, Schwarz);
    }
    break;
  case U:
    if (spherical) {
      return fock_kernel_hess<U, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                     m_precision, Schwarz);
    } else {
      return fock_kernel_hess<U, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                      m_precision, Schwarz);
    }

  case G:
    if (spherical) {
      return fock_kernel_hess<G, Sph>(m_env, m_aobasis, m_shellpairs, mo,
                                     m_precision, Schwarz);
    } else {
      return fock_kernel_hess<G, Cart>(m_env, m_aobasis, m_shellpairs, mo,
                                      m_precision, Schwarz);
    }
  }
}

} // namespace occ::qm
