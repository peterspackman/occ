#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/dft/dft.h>
#include <occ/qm/gradients.h>

namespace occ::dft {

qm::JKTriple DFT::compute_JK_gradient(const MolecularOrbitals &mo,
                                      const Mat &Schwarz) const {

  occ::timing::start(occ::timing::category::dft_fock_gradient);
  occ::log::debug("Computing DFT JK gradient");

  MatTriple J;
  MatTriple K = MatTriple::Zero(mo.n_ao, mo.n_ao);
  double k_factor = exact_exchange_factor();

  // Check for range-separated hybrid - not yet implemented
  RangeSeparatedParameters rs = range_separated_parameters();
  if (rs.omega != 0.0) {
    occ::log::warn("Range-separated hybrid gradient not yet implemented, using "
                   "standard exchange");
  }

  if (k_factor > 0.0) {
    auto JK = m_hf.compute_JK_gradient(mo, Schwarz);

    JK.K.x *= k_factor;
    JK.K.y *= k_factor;
    JK.K.z *= k_factor;

    K = JK.K;
    J = JK.J;
  } else {
    // Pure DFT case (no exact exchange)
    J = compute_J_gradient(mo, Schwarz);
  }

  occ::timing::stop(occ::timing::category::dft_fock_gradient);
  return {J, K};
}

MatTriple DFT::compute_fock_gradient(const MolecularOrbitals &mo,
                                     const Mat &Schwarz) const {
  occ::timing::start(occ::timing::category::dft_fock_gradient);
  auto [J, K] = compute_JK_gradient(mo, Schwarz);

  J.x -= K.x;
  J.y -= K.y;
  J.z -= K.z;
  return J;
}

Mat3N DFT::compute_xc_gradient(const MolecularOrbitals &mo,
                               const Mat &Schwarz) const {
  occ::timing::start(occ::timing::category::dft_gradient);
  occ::log::debug("Computing DFT XC gradient");
  constexpr auto G = SpinorbitalKind::General;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto R = SpinorbitalKind::Restricted;

  const auto &basis = m_hf.aobasis();
  const auto &atoms = m_hf.atoms();
  const size_t natoms = atoms.size();
  const size_t nbf = basis.nbf();

  Mat3N gradient = Mat3N::Zero(3, natoms);
  int deriv = density_derivative();

  if (mo.kind == R) {
    if (deriv == 0) {
      gradient = compute_xc_gradient_impl<0, R>(mo, Schwarz);
    } else if (deriv == 1) {
      gradient = compute_xc_gradient_impl<1, R>(mo, Schwarz);
    } else if (deriv == 2) {
      gradient = compute_xc_gradient_impl<2, R>(mo, Schwarz);
    }
  } else {
    if (deriv == 0) {
      gradient = compute_xc_gradient_impl<0, U>(mo, Schwarz);
    } else if (deriv == 1) {
      gradient = compute_xc_gradient_impl<1, U>(mo, Schwarz);
    } else if (deriv == 2) {
      gradient = compute_xc_gradient_impl<2, U>(mo, Schwarz);
    }
  }

  occ::timing::stop(occ::timing::category::dft_gradient);

  return gradient;
}

} // namespace occ::dft
