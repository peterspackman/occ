#include <fmt/core.h>
#include <fmt/ranges.h>
#include <occ/core/eeq.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/dft/hirshfeld.h>
#include <occ/driver/population.h>
#include <occ/qm/chelpg.h>

namespace occ::driver {

const std::vector<std::string> &partial_charge_schemes() {
  static const std::vector<std::string> schemes{
      "mulliken", "lowdin", "chelpg", "hirshfeld", "eeq", "xtb"};
  return schemes;
}

Vec partial_charges(const qm::Wavefunction &wfn, const std::string &scheme) {
  const std::string name = occ::util::to_lower_copy(scheme);
  if (name == "mulliken")
    return wfn.mulliken_charges();
  if (name == "lowdin")
    return wfn.lowdin_charges();
  if (name == "chelpg")
    return qm::chelpg_charges(wfn);
  if (name == "hirshfeld") {
    if (wfn.have_xtb_data)
      throw std::runtime_error(
          "Hirshfeld charges need an all-electron density; a GFN2-xTB "
          "wavefunction only has the valence density");
    return dft::calculate_hirshfeld_charges(wfn.basis, wfn.mo, wfn.charge());
  }
  if (name == "eeq") {
    const Mat3N positions = wfn.positions() * occ::units::BOHR_TO_ANGSTROM;
    return occ::core::charges::eeq_partial_charges(wfn.atomic_numbers(),
                                                   positions, wfn.charge());
  }
  if (name == "xtb") {
    if (!wfn.have_xtb_data)
      throw std::runtime_error(
          "xtb charges need a GFN2-xTB wavefunction (occ tb or occ scf "
          "gfn2-xtb)");
    return wfn.xtb_atomic_charges;
  }
  throw std::runtime_error(
      fmt::format("Unknown partial charge scheme '{}' (expected one of: {})",
                  scheme, fmt::join(partial_charge_schemes(), ", ")));
}

} // namespace occ::driver
