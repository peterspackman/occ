#include "detail/ecp_kernels.h"
#include "detail/multipole_kernel.h"
#include "detail/schwarz_kernel.h"
#include "detail/two_center_kernels.h"

#if HAVE_ECPINT
  #include <occ/core/log.h> 
#endif

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

Mat IntegralEngine::schwarz() const {
  if (is_spherical()) {
    return detail::schwarz_kernel<ShellKind::Spherical>(m_env, m_aobasis,
                                                        m_shellpairs);
  } else {
    return detail::schwarz_kernel<ShellKind::Cartesian>(m_env, m_aobasis,
                                                        m_shellpairs);
  }
}

} // namespace occ::qm
