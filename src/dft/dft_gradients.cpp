#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/dft/dft.h>

namespace occ::dft {

qm::JKTriple DFT::compute_JK_gradient(const MolecularOrbitals &mo,
                                      const Mat &Schwarz) {

  occ::timing::start(occ::timing::category::dft_fock_gradient);
  occ::log::debug("Computing DFT JK gradient");

  MatTriple J;
  // For unrestricted, gradient matrices have shape (2*n_ao, n_ao) like density matrix
  size_t K_rows = (mo.kind == SpinorbitalKind::Unrestricted) ? 2 * mo.n_ao : mo.n_ao;
  MatTriple K = MatTriple::Zero(K_rows, mo.n_ao);
  double k_factor = exact_exchange_factor();
  RangeSeparatedParameters rs = range_separated_parameters();

  if (rs.omega != 0.0) {
    // Range-separated hybrid: K = (α + β) * K_short - β * K_long
    occ::log::debug("Computing range-separated hybrid gradient with ω = {:.6f}", rs.omega);
    
    // Compute short-range (ω=0) JK gradients
    auto JK_short = m_hf.compute_JK_gradient(mo, Schwarz);
    J = JK_short.J; // Coulomb is same for both ranges
    
    // Compute long-range (ω=rs.omega) K gradients  
    m_hf.set_range_separated_omega(rs.omega);
    auto JK_long = m_hf.compute_JK_gradient(mo, Schwarz);
    m_hf.set_range_separated_omega(0.0); // Reset omega
    
    // Combine: K = (α + β) * K_short - β * K_long
    K.x = (rs.alpha + rs.beta) * JK_short.K.x - rs.beta * JK_long.K.x;
    K.y = (rs.alpha + rs.beta) * JK_short.K.y - rs.beta * JK_long.K.y;
    K.z = (rs.alpha + rs.beta) * JK_short.K.z - rs.beta * JK_long.K.z;
    
  } else if (k_factor > 0.0) {
    // Global hybrid
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
                                     const Mat &Schwarz) {
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
  constexpr auto G = SpinorbitalKind::General;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto R = SpinorbitalKind::Restricted;

  const auto &basis = m_hf.aobasis();
  const auto &atoms = m_hf.atoms();
  const size_t natoms = atoms.size();
  const size_t nbf = basis.nbf();

  Mat3N gradient = Mat3N::Zero(3, natoms);
  int deriv = density_derivative();
  occ::log::debug("Computing DFT XC gradient: density_derivative = {}", deriv);


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

  // Add VV10 nonlocal correlation gradient if present
  gradient += compute_nlc_gradient(mo);

  occ::timing::stop(occ::timing::category::dft_gradient);

  return gradient;
}

} // namespace occ::dft
