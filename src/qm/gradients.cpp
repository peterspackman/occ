#include <occ/qm/gradients.h>
#include <occ/disp/dftd4.h>
#include <occ/core/molecule.h>

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

} // namespace impl

} // namespace occ::qm
