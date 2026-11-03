#include "detail/ecp_native_kernels.h"
#include "detail/multipole_kernel.h"
#include "detail/schwarz_kernel.h"
#include "detail/two_center_kernels.h"

#include <occ/core/log.h>

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

Mat IntegralEngine::effective_core_potential(bool use_shellpair_list) const {
  if (!have_effective_core_potentials())
    throw std::runtime_error(
        "Called effective_core_potential without any ECPs");

  occ::timing::start(occ::timing::category::ecp);
  constexpr auto Cart = ShellKind::Cartesian;
  constexpr auto Sph = ShellKind::Spherical;
  ShellPairList empty_shellpairs = {};
  const auto &shellpairs = use_shellpair_list ? m_shellpairs : empty_shellpairs;
  Mat result;
  if (is_spherical())
    result = detail::ecp_native_operator_kernel<Sph>(m_aobasis, *m_ecp_native,
                                                     shellpairs);
  else
    result = detail::ecp_native_operator_kernel<Cart>(m_aobasis, *m_ecp_native,
                                                      shellpairs);
  occ::timing::stop(occ::timing::category::ecp);
  return result;
}

Mat3N IntegralEngine::effective_core_potential_gradient(
    const Mat &D, bool use_shellpair_list) const {
  if (!have_effective_core_potentials())
    return Mat3N::Zero(3, m_aobasis.atoms().size());

  constexpr auto Cart = ShellKind::Cartesian;
  constexpr auto Sph = ShellKind::Spherical;
  ShellPairList empty_shellpairs = {};
  const auto &shellpairs = use_shellpair_list ? m_shellpairs : empty_shellpairs;
  if (is_spherical())
    return detail::ecp_native_gradient_kernel<Sph>(m_aobasis, *m_ecp_native,
                                                   shellpairs, D);
  return detail::ecp_native_gradient_kernel<Cart>(m_aobasis, *m_ecp_native,
                                                  shellpairs, D);
}

Mat IntegralEngine::effective_core_potential_hessian(const Mat &P) const {
  const int natom = static_cast<int>(m_aobasis.atoms().size());
  if (!have_effective_core_potentials())
    return Mat::Zero(3 * natom, 3 * natom);
  if (is_spherical())
    return detail::ecp_native_hessian_kernel<ShellKind::Spherical>(
        m_aobasis, *m_ecp_native, m_shellpairs, P);
  return detail::ecp_native_hessian_kernel<ShellKind::Cartesian>(
      m_aobasis, *m_ecp_native, m_shellpairs, P);
}

void IntegralEngine::add_effective_core_potential_derivatives(
    std::vector<Mat> &H1) const {
  if (!have_effective_core_potentials())
    return;
  if (is_spherical())
    detail::ecp_native_derivative_matrices_kernel<ShellKind::Spherical>(
        m_aobasis, *m_ecp_native, m_shellpairs, H1);
  else
    detail::ecp_native_derivative_matrices_kernel<ShellKind::Cartesian>(
        m_aobasis, *m_ecp_native, m_shellpairs, H1);
}

void IntegralEngine::set_effective_core_potentials(
    const ShellList &ecp_shells, const std::vector<int> &ecp_electrons) {
  if (ecp_shells != m_aobasis.ecp_shells())
    throw std::invalid_argument(
        "set_effective_core_potentials: ECP shells must be the basis's own "
        "(AOBasis::ecp_shells)");
  const auto &atoms = m_aobasis.atoms();
  for (int i = 0; i < ecp_electrons.size(); i++) {
    int charge = atoms[i].atomic_number - ecp_electrons[i];
    occ::log::debug("setting atom {} charge to {}", i, charge);
    m_env.set_atom_charge(i, charge);
  }
  // The evaluator takes the ECP shells (grouped by atom) from the basis, with
  // derivatives up to second order for gradients and Hessians.
  m_ecp_native = std::make_shared<const occ::ints::EcpEvaluator>(m_aobasis, 2);
  m_have_ecp = true;
}

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

namespace {
template <ShellKind kind>
MatTriple dipole_matrices_kernel(const AOBasis &basis, IntEnv &env,
                                 const ShellPairList &shellpairs,
                                 const Vec3 &origin) {
  const auto nbf = basis.nbf();
  const auto &first_bf = basis.first_bf();
  MatTriple result = MatTriple::Zero(nbf, nbf);
  env.set_common_origin({origin.x(), origin.y(), origin.z()});
  auto buffer = std::make_unique<double[]>(env.buffer_size_1e(Op::dipole));
  for (size_t p = 0; p < basis.size(); p++) {
    for (const auto q : shellpairs[p]) {
      std::array<int, 2> idxs{static_cast<int>(p), static_cast<int>(q)};
      auto dims = env.two_center_helper<Op::dipole, kind>(idxs, nullptr,
                                                          buffer.get(), nullptr);
      if (dims[0] < 0)
        continue;
      const size_t n = static_cast<size_t>(dims[0]) * dims[1];
      for (int x = 0; x < 3; x++) {
        Eigen::Map<const Mat> blk(buffer.get() + x * n, dims[0], dims[1]);
        result[x].block(first_bf[p], first_bf[q], dims[0], dims[1]) = blk;
        result[x].block(first_bf[q], first_bf[p], dims[1], dims[0]) =
            blk.transpose();
      }
    }
  }
  env.set_common_origin({0.0, 0.0, 0.0});
  return result;
}
} // namespace

MatTriple IntegralEngine::dipole_matrices(const Vec3 &origin) const {
  if (is_spherical())
    return dipole_matrices_kernel<ShellKind::Spherical>(m_aobasis, m_env,
                                                        m_shellpairs, origin);
  return dipole_matrices_kernel<ShellKind::Cartesian>(m_aobasis, m_env,
                                                      m_shellpairs, origin);
}

} // namespace occ::qm
