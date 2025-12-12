#include <occ/qm/gradients.h>
#include <occ/disp/dftd4.h>
#include <occ/xdm/xdm.h>
#include <occ/xdm/xdm_parameters.h>

namespace occ::qm {

namespace impl {

std::pair<double, Mat3N> compute_d4_dispersion(
    const std::vector<core::Atom> &atoms,
    int charge,
    const std::string &functional) {

  disp::D4Dispersion d4(atoms);
  d4.set_charge(charge);

  bool success = d4.set_functional(functional);
  if (!success) {
    occ::log::warn("D4 parameters not found for functional '{}', using default PBE parameters", functional);
  }

  return d4.energy_and_gradient();
}

std::pair<double, Mat3N> compute_xdm_dispersion(
    const AOBasis &basis,
    const MolecularOrbitals &mo,
    int charge,
    const std::string &functional,
    const std::optional<occ::xdm::XDM::Parameters> &params_override) {

  occ::xdm::XDM::Parameters params;

  if (params_override.has_value()) {
    // Use provided parameters (from config flags)
    params = params_override.value();
    occ::log::debug("Using user-specified XDM parameters: a1={:.4f}, a2={:.4f}",
                    params.a1, params.a2);
  } else {
    // Look up functional-specific parameters
    auto params_opt = occ::xdm::get_xdm_parameters(functional);
    params = params_opt.value_or(
      occ::xdm::XDM::Parameters{0.7, 1.4}  // Default parameters
    );

    if (!params_opt.has_value()) {
      occ::log::warn("XDM parameters not found for functional '{}', using defaults (a1={:.4f}, a2={:.4f})",
                     functional, params.a1, params.a2);
    } else {
      occ::log::debug("XDM parameters for '{}': a1={:.4f}, a2={:.4f}",
                      functional, params.a1, params.a2);
    }
  }

  // Create XDM calculator
  occ::xdm::XDM xdm(basis, charge, params);

  // Compute energy (populates moments internally)
  double e_xdm = xdm.energy(mo);

  // Compute gradient (reuses cached moments)
  const Mat3N &grad_xdm = xdm.forces(mo);

  return {e_xdm, grad_xdm};
}

} // namespace impl

} // namespace occ::qm
