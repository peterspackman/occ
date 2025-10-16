#pragma once
#include <occ/core/util.h>
#include <occ/qm/spinorbital.h>
#include <string>
#include <utility>

namespace occ::driver {

enum class MethodKind {
  HF,
  DFT,
  MP2,
};

struct MethodSpec {
  std::string base_method;
  std::string dispersion;

  MethodSpec(const std::string &base, const std::string &disp = "")
      : base_method(base), dispersion(disp) {}
};

/**
 * @brief Parse method string to separate base method and dispersion type
 *
 * Parses method strings like "pbe-d4", "wb97x" into base method
 * name and optional dispersion correction type.
 *
 * @param method_string Full method string (e.g., "pbe-d4")
 * @return MethodSpec with base_method and dispersion fields
 *
 * Examples:
 *   "pbe-d4" -> MethodSpec("pbe", "d4")
 *   "b3lyp-d4" -> MethodSpec("b3lyp", "d4")
 *   "pbe" -> MethodSpec("pbe", "")
 */
inline MethodSpec parse_method_string(const std::string &method_string) {
  // Look for dispersion suffix like "-d4"
  size_t dash_pos = method_string.rfind('-');

  if (dash_pos == std::string::npos) {
    // No dash found, no dispersion
    return MethodSpec(method_string, "");
  }

  std::string base = method_string.substr(0, dash_pos);
  std::string suffix = method_string.substr(dash_pos + 1);

  // Check if suffix is a known dispersion type
  if (suffix == "d4") {
    return MethodSpec(base, suffix);
  }

  // Not a dispersion suffix, treat whole string as method name
  return MethodSpec(method_string, "");
}

inline qm::SpinorbitalKind determine_spinorbital_kind(const std::string &name,
                                                      int multiplicity,
                                                      MethodKind method_kind) {
  auto lc = occ::util::to_lower_copy(name);
  switch (method_kind) {
  default: { // default to HartreeFock
    if (lc[0] == 'g')
      return qm::SpinorbitalKind::General;
    else if (lc[0] == 'u' || multiplicity > 1)
      return qm::SpinorbitalKind::Unrestricted;
    else
      return qm::SpinorbitalKind::Restricted;
    break;
  }
  case MethodKind::DFT: {
    if (lc[0] == 'u' || multiplicity > 1)
      return qm::SpinorbitalKind::Unrestricted;
    else
      return qm::SpinorbitalKind::Restricted;
  }
  case MethodKind::MP2: {
    if (lc[0] == 'u' || multiplicity > 1)
      return qm::SpinorbitalKind::Unrestricted;
    else
      return qm::SpinorbitalKind::Restricted;
  }
  }
}

inline MethodKind method_kind_from_string(const std::string &name) {
  auto lc = occ::util::to_lower_copy(name);
  if (lc == "hf" || lc == "rhf" || lc == "uhf" || lc == "ghf" || lc == "scf" ||
      lc == "hartree-fock" || lc == "hartree fock") {
    return MethodKind::HF;
  }
  if (lc == "mp2" || lc == "rmp2" || lc == "ump2" || lc == "moller-plesset" ||
      lc == "moller plesset") {
    return MethodKind::MP2;
  }
  return MethodKind::DFT;
};

} // namespace occ::driver
