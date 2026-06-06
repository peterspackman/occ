#include <occ/disp/d4.h>
#include <occ/qm/gradients.h>

namespace occ::qm {

namespace impl {

std::pair<double, Mat3N> compute_d4_dispersion(
    const std::vector<core::Atom> &atoms,
    int charge,
    const std::string &functional) {

  disp::D4Dispersion d4(atoms, disp::RefqMode::DFT);
  try {
    d4.set_functional(functional);
  } catch (const std::exception &e) {
    occ::log::warn("D4 parameters not found for functional '{}' ({}), using "
                   "default PBE parameters", functional, e.what());
    d4.set_functional("pbe");
  }
  d4.set_charges_eeq(static_cast<double>(charge));
  return d4.energy_and_gradient();
}

} // namespace impl

} // namespace occ::qm
