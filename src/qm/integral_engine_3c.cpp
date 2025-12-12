#include "detail/three_center_kernels.h"
#include <cmath>
#include <occ/qm/integral_engine.h>

namespace occ::qm {

using ShellList = std::vector<Shell>;
using AtomList = std::vector<occ::core::Atom>;
using ShellPairList = std::vector<std::vector<size_t>>;
using IntEnv = cint::IntegralEnvironment;
using ShellKind = Shell::Kind;
using Op = cint::Operator;
using occ::core::PointCharge;

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

} // namespace occ::qm
