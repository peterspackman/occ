#include "detail/ecp_kernels.h"
#include "detail/jk.h"
#include "detail/kernels.h"
#include "detail/multipole_kernel.h"
#include "detail/schwarz_kernel.h"
#include "detail/three_center_kernels.h"
#include <chrono>
#include <cmath>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/qm/integral_engine.h>

namespace occ::qm {

using ShellList = std::vector<Shell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = Shell::Kind;
using Op = cint::Operator;
using occ::core::PointCharge;

Mat IntegralEngine::one_electron_operator(Op op,
                                          bool use_shellpair_list) const {
  bool spherical = is_spherical();
  constexpr auto Cart = ShellKind::Cartesian;
  constexpr auto Sph = ShellKind::Spherical;
  ShellPairList empty_shellpairs = {};
  const auto &shellpairs = use_shellpair_list ? m_shellpairs : empty_shellpairs;
  switch (op) {
  case Op::overlap: {
    if (spherical) {
      return detail::one_electron_operator_kernel<Op::overlap, Sph>(
          m_aobasis, m_env, shellpairs);
    } else {
      return detail::one_electron_operator_kernel<Op::overlap, Cart>(
          m_aobasis, m_env, shellpairs);
    }
    break;
  }
  case Op::nuclear: {
    if (spherical) {
      return detail::one_electron_operator_kernel<Op::nuclear, Sph>(
          m_aobasis, m_env, shellpairs);
    } else {
      return detail::one_electron_operator_kernel<Op::nuclear, Cart>(
          m_aobasis, m_env, shellpairs);
    }
    break;
  }
  case Op::kinetic: {
    if (spherical) {
      return detail::one_electron_operator_kernel<Op::kinetic, Sph>(
          m_aobasis, m_env, shellpairs);
    } else {
      return detail::one_electron_operator_kernel<Op::kinetic, Cart>(
          m_aobasis, m_env, shellpairs);
    }
    break;
  }
  case Op::coulomb: {
    if (spherical) {
      return detail::one_electron_operator_kernel<Op::coulomb, Sph>(
          m_aobasis, m_env, shellpairs);
    } else {
      return detail::one_electron_operator_kernel<Op::coulomb, Cart>(
          m_aobasis, m_env, shellpairs);
    }
    break;
  }
  default:
    throw std::runtime_error("Invalid operator for two-center integral");
    break;
  }
}

Mat IntegralEngine::rinv_operator_atom_center(size_t atom_index,
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
  Mat result;
  if (spherical) {
    result = detail::one_electron_operator_kernel<Op::rinv, Sph>(
        m_aobasis, m_env, shellpairs);
  } else {
    result = detail::one_electron_operator_kernel<Op::rinv, Cart>(
        m_aobasis, m_env, shellpairs);
  }
  m_env.set_rinv_origin({0.0, 0.0, 0.0});
  return result;
}

#if HAVE_ECPINT

Mat IntegralEngine::effective_core_potential(bool use_shellpair_list) const {
  if (!have_effective_core_potentials())
    throw std::runtime_error(
        "Called effective_core_potential without any ECPs");

  occ::timing::start(occ::timing::category::ecp);
  bool spherical = is_spherical();
  constexpr auto Cart = ShellKind::Cartesian;
  constexpr auto Sph = ShellKind::Spherical;
  ShellPairList empty_shellpairs = {};
  const auto &shellpairs = use_shellpair_list ? m_shellpairs : empty_shellpairs;
  Mat result;
  if (spherical) {
    result = detail::ecp_operator_kernel<Sph>(m_aobasis, m_ecp_gaussian_shells,
                                              m_ecp, m_ecp_ao_max_l,
                                              m_ecp_max_l, shellpairs);
  } else {
    result = detail::ecp_operator_kernel<Cart>(m_aobasis, m_ecp_gaussian_shells,
                                               m_ecp, m_ecp_ao_max_l,
                                               m_ecp_max_l, shellpairs);
  }
  occ::timing::stop(occ::timing::category::ecp);
  return result;
}

void IntegralEngine::set_effective_core_potentials(
    const ShellList &ecp_shells, const std::vector<int> &ecp_electrons) {
  const auto &atoms = m_aobasis.atoms();
  for (const auto &sh : m_aobasis.shells()) {
    libecpint::GaussianShell ecpint_shell(
        {sh.origin(0), sh.origin(1), sh.origin(2)}, sh.l);
    const Mat &coeffs_norm = sh.contraction_coefficients;
    m_ecp_ao_max_l = std::max(static_cast<int>(sh.l), m_ecp_ao_max_l);
    for (int i = 0; i < sh.num_primitives(); i++) {
      double c = coeffs_norm(i, 0);
      if (sh.l == 0) {
        c *= 0.28209479177387814; // 1 / (2 * sqrt(pi))
      }
      if (sh.l == 1) {
        c *= 0.4886025119029199; // sqrt(3) / (2 * sqrt(pi))
      }
      ecpint_shell.addPrim(sh.exponents(i), c);
    }
    m_ecp_gaussian_shells.push_back(ecpint_shell);
  }
  for (int i = 0; i < ecp_electrons.size(); i++) {
    int charge = atoms[i].atomic_number - ecp_electrons[i];
    occ::log::debug("setting atom {} charge to {}", i, charge);
    m_env.set_atom_charge(i, charge);
  }

  Vec3 pt = ecp_shells[0].origin;
  // For some reason need to merge all shells that share a center.
  // This code relies on shells with the same center being grouped.
  //
  libecpint::ECP ecp(pt.data());
  for (const auto &sh : ecp_shells) {
    if ((pt - sh.origin).norm() > 1e-3) {
      ecp.sort();
      ecp.atom_id = m_ecp.size();
      m_ecp.push_back(ecp);
      pt = sh.origin;
      ecp = libecpint::ECP(pt.data());
    }
    for (int i = 0; i < sh.num_primitives(); i++) {
      m_ecp_max_l = std::max(static_cast<int>(sh.l), m_ecp_max_l);
      ecp.addPrimitive(sh.ecp_r_exponents(i), sh.l, sh.exponents(i),
                       sh.contraction_coefficients(i, 0), false);
    }
  }

  // add the last ECP to the end
  ecp.sort();
  ecp.atom_id = m_ecp.size();
  m_ecp.push_back(ecp);

  m_have_ecp = true;
}
#else

Mat IntegralEngine::effective_core_potential(bool use_shellpair_list) const {
  throw std::runtime_error("Not compiled with libecpint");
}
void IntegralEngine::set_effective_core_potentials(
    const ShellList &ecp_shells, const std::vector<int> &ecp_electrons) {
  throw std::runtime_error("Not compiled with libecpint");
}

#endif

Vec IntegralEngine::multipole(int order, const MolecularOrbitals &mo,
                              const Vec3 &origin) const {
  bool spherical = is_spherical();
  constexpr auto R = SpinorbitalKind::Restricted;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  constexpr auto Cart = ShellKind::Cartesian;
  constexpr auto Sph = ShellKind::Spherical;
  if (mo.kind == R) {
    switch (order) {
    case 0:
      if (spherical) {
        return detail::multipole_kernel<0, R, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<0, R, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    case 1:
      if (spherical) {
        return detail::multipole_kernel<1, R, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<1, R, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    case 2:
      if (spherical) {
        return detail::multipole_kernel<2, R, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<2, R, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    case 3:
      if (spherical) {
        return detail::multipole_kernel<3, R, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<3, R, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    case 4:
      if (spherical) {
        return detail::multipole_kernel<4, R, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<4, R, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    default:
      throw std::runtime_error("Invalid multipole order");
      break;
    }
  } else if (mo.kind == U) {
    switch (order) {
    case 0:
      if (spherical) {
        return detail::multipole_kernel<0, U, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<0, U, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    case 1:
      if (spherical) {
        return detail::multipole_kernel<1, U, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<1, U, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    case 2:
      if (spherical) {
        return detail::multipole_kernel<2, U, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<2, U, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    case 3:
      if (spherical) {
        return detail::multipole_kernel<3, U, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<3, U, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    case 4:
      if (spherical) {
        return detail::multipole_kernel<4, U, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<4, U, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    default:
      throw std::runtime_error("Invalid multipole order");
      break;
    }
  } else { // if (sk == G)
    switch (order) {
    case 0:
      if (spherical) {
        return detail::multipole_kernel<0, G, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<0, G, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    case 1:
      if (spherical) {
        return detail::multipole_kernel<1, G, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<1, G, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    case 2:
      if (spherical) {
        return detail::multipole_kernel<2, G, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<2, G, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    case 3:
      if (spherical) {
        return detail::multipole_kernel<3, G, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<3, G, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    case 4:
      if (spherical) {
        return detail::multipole_kernel<4, G, Sph>(m_aobasis, m_env,
                                                   m_shellpairs, mo, origin);
      } else {
        return detail::multipole_kernel<4, G, Cart>(m_aobasis, m_env,
                                                    m_shellpairs, mo, origin);
      }
      break;
    default:
      throw std::runtime_error("Invalid multipole order");
      break;
    }
  }
}

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

  occ::parallel::thread_local_storage<Mat> F_local(Mat::Zero(nbf, nbf));

  // construct the 2-electron repulsion integrals engine
  auto shell2bf = m_aobasis.first_bf();
  auto shell2bf_D = m_auxbasis.first_bf();

  // Calculate total number of shell quartets for parallel_for
  size_t total_quartets = 0;
  for (int s1 = 0; s1 != nsh; ++s1) {
    for (int s2 = 0; s2 <= s1; ++s2) {
      for (int s3 = 0; s3 < nsh_aux; ++s3) {
        int s4_begin = is_shell_diagonal ? s3 : 0;
        int s4_fence = is_shell_diagonal ? s3 + 1 : nsh_aux;
        total_quartets += (s4_fence - s4_begin);
      }
    }
  }

  occ::parallel::parallel_for(size_t(0), total_quartets, [&](size_t idx) {
    auto &F = F_local.local();
    occ::qm::cint::Optimizer opt(m_env, Op::coulomb, 4);
    auto buffer = std::make_unique<double[]>(m_env.buffer_size_2e());

    // Convert linear index back to s1,s2,s3,s4
    size_t current_idx = 0;
    bool found = false;
    
    for (int s1 = 0; s1 != nsh && !found; ++s1) {
      int bf1_first = shell2bf[s1];
      int n1 = m_aobasis[s1].size();

      for (int s2 = 0; s2 <= s1 && !found; ++s2) {
        int bf2_first = shell2bf[s2];
        int n2 = m_aobasis[s2].size();

        for (int s3 = 0; s3 < nsh_aux && !found; ++s3) {
          int bf3_first = shell2bf_D[s3];
          int n3 = D_bs[s3].size();

          int s4_begin = is_shell_diagonal ? s3 : 0;
          int s4_fence = is_shell_diagonal ? s3 + 1 : nsh_aux;

          for (int s4 = s4_begin; s4 != s4_fence; ++s4) {
            if (current_idx == idx) {
              found = true;

              int bf4_first = shell2bf_D[s4];
              int n4 = D_bs[s4].size();

              // compute the permutational degeneracy
              double s12_deg = (s1 == s2) ? 1.0 : 2.0;

              std::array<int, 4> dims;
              if (s3 >= s4) {
                double s34_deg = (s3 == s4) ? 1.0 : 2.0;
                double s1234_deg = s12_deg * s34_deg;
                std::array<int, 4> idxs{s1, s2, s3 + nsh, s4 + nsh};
                if (spherical) {
                  dims = m_env.four_center_helper<op, Sph>(
                      idxs, opt.optimizer_ptr(), buffer.get(), nullptr);
                } else {
                  dims = m_env.four_center_helper<op, Cart>(
                      idxs, opt.optimizer_ptr(), buffer.get(), nullptr);
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

              std::array<int, 4> idxs{s1, s3 + nsh, s2, s4 + nsh};
              if (spherical) {
                dims = m_env.four_center_helper<op, Sph>(
                    idxs, opt.optimizer_ptr(), buffer.get(), nullptr);
              } else {
                dims = m_env.four_center_helper<op, Cart>(
                    idxs, opt.optimizer_ptr(), buffer.get(), nullptr);
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
              break;
            }
            current_idx++;
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

Mat IntegralEngine::schwarz() const {
  if (is_spherical()) {
    return detail::schwarz_kernel<ShellKind::Spherical>(m_env, m_aobasis,
                                                        m_shellpairs);
  } else {
    return detail::schwarz_kernel<ShellKind::Cartesian>(m_env, m_aobasis,
                                                        m_shellpairs);
  }
}

/*
 * Three-center integrals
 */
Mat IntegralEngine::point_charge_potential(
    const std::vector<PointCharge> &charges, double alpha) {
  ShellList dummy_shells;
  dummy_shells.reserve(charges.size());
  for (size_t i = 0; i < charges.size(); i++) {
    dummy_shells.push_back(Shell(charges[i], alpha));
  }
  set_auxiliary_basis(dummy_shells, true);
  if (is_spherical()) {
    return detail::point_charge_potential_kernel<ShellKind::Spherical>(
        m_env, m_aobasis, m_auxbasis, m_shellpairs);
  } else {
    return detail::point_charge_potential_kernel<ShellKind::Cartesian>(
        m_env, m_aobasis, m_auxbasis, m_shellpairs);
  }
}

Mat IntegralEngine::wolf_point_charge_potential(
    const std::vector<PointCharge> &external_charges,
    const std::vector<double> &atomic_partial_charges, double alpha,
    double cutoff) {

  // TODO cutoff based on distance from centroid of molecule?
  std::vector<PointCharge> charges;
  for (const auto pc : external_charges) {
    auto rij = pc.position().norm();
    if (rij > cutoff)
      continue;
    charges.push_back(pc);
  }

  // Gaussian charge distribution potential
  Mat Vext, Vintra;
  ShellList dummy_shells;

  // add nuclear charge
  dummy_shells.reserve(charges.size());
  for (const auto &charge : charges) {
    dummy_shells.push_back(Shell(charge));
  }
  // unit shell
  set_auxiliary_basis(dummy_shells, true);

  double old_omega = m_env.range_separated_omega();
  m_env.set_range_separated_omega(-alpha);

  if (is_spherical()) {
    Vext = detail::point_charge_potential_kernel<ShellKind::Spherical>(
        m_env, m_aobasis, m_auxbasis, m_shellpairs);
  } else {
    Vext = detail::point_charge_potential_kernel<ShellKind::Cartesian>(
        m_env, m_aobasis, m_auxbasis, m_shellpairs);
  }

  dummy_shells.clear();
  double total_charge = 0.0;
  const auto &atoms = m_aobasis.atoms();
  for (int i = 0; i < atoms.size(); i++) {
    dummy_shells.push_back(
        Shell(PointCharge(atomic_partial_charges[i], atoms[i].position())));
    total_charge += atomic_partial_charges[i];
  }
  set_auxiliary_basis(dummy_shells, true);
  m_env.set_range_separated_omega(alpha);

  if (is_spherical()) {
    Vintra = detail::point_charge_potential_kernel<ShellKind::Spherical>(
        m_env, m_aobasis, m_auxbasis, m_shellpairs);
  } else {
    Vintra = detail::point_charge_potential_kernel<ShellKind::Cartesian>(
        m_env, m_aobasis, m_auxbasis, m_shellpairs);
  }

  m_env.set_range_separated_omega(old_omega);

  total_charge += std::accumulate(
      charges.begin(), charges.end(),
      0.0, // initial value
      [](double sum, const PointCharge &pc) { return sum + pc.charge(); });
  double background_term = total_charge * std::erfc(alpha * cutoff) / cutoff;

  Mat S = one_electron_operator(Op::overlap);
  return Vext - Vintra + background_term * S;
}

#if HAVE_ECPINT
Vec electric_potential_ecp_kernel(std::vector<libecpint::ECP> &ecps,
                                  int ecp_lmax, const Mat3N &points) {
  Vec result = Vec::Zero(points.cols());
  for (int pt = 0; pt < points.cols(); pt++) {
    for (const auto &U : ecps) {
      double dx = points(0, pt) - U.center_[0];
      double dy = points(1, pt) - U.center_[1];
      double dz = points(2, pt) - U.center_[2];
      double r = std::sqrt(dx * dx + dy * dy + dz * dz);

      double fac = 1.0;
      for (int l = 0; l <= U.getL(); l++) {
        result(pt) += fac * U.evaluate(r, l);
        fac *= r;
      }
    }
  }
  return result;
}
#endif

Vec IntegralEngine::electric_potential(const MolecularOrbitals &mo,
                                       const Mat3N &points) {
  constexpr auto R = SpinorbitalKind::Restricted;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;
  constexpr auto Sph = ShellKind::Spherical;
  constexpr auto Cart = ShellKind::Cartesian;
  ShellList dummy_shells;
  dummy_shells.reserve(points.cols());
  Vec result = Vec::Zero(points.cols());
  for (size_t i = 0; i < points.cols(); i++) {
    dummy_shells.push_back(Shell(PointCharge(1.0, points.col(i))));
  }
  set_auxiliary_basis(dummy_shells, true);

  // Below code could be used if ECPs are needed for electric potential,
  // not sure if it's correct
  /*
  if (m_have_ecp) {
      result += electric_potential_ecp_kernel(m_ecp, m_ecp_max_l, points);
  }
  */

  if (is_spherical()) {
    switch (mo.kind) {
    default: // Restricted
      result += detail::electric_potential_kernel<R, Sph>(
          m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
      break;
    case U:
      result += detail::electric_potential_kernel<U, Sph>(
          m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
      break;
    case G:
      result += detail::electric_potential_kernel<G, Sph>(
          m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
      break;
    }
  } else {
    switch (mo.kind) {
    default: // Restricted
      result += detail::electric_potential_kernel<R, Cart>(
          m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
      break;
    case U:
      result += detail::electric_potential_kernel<U, Cart>(
          m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
      break;
    case G:
      result += detail::electric_potential_kernel<G, Cart>(
          m_env, m_aobasis, m_auxbasis, m_shellpairs, mo);
      break;
    }
  }
  return result;
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
