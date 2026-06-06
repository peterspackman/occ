#pragma once
#include <occ/core/util.h>
#include <string>

namespace occ::core {

/**
 * @brief Spin treatment for molecular orbitals / SCF.
 *
 * Unscoped enum: the enumerators are intentionally usable unqualified within
 * the enclosing namespace. The canonical definition lives here (occ_core) so
 * that low-level consumers (e.g. input/config structs) can reference it
 * without depending on occ_qm; occ::qm re-exports the type and enumerators for
 * backwards compatibility (see occ/qm/spinorbital.h).
 */
enum SpinorbitalKind { Restricted, Unrestricted, General };

inline bool get_spinorbital_kind_from_string(const std::string &s,
                                             SpinorbitalKind &sk) {
  std::string lc = occ::util::to_lower_copy(s);
  bool valid = false;
  if ((lc == "r") || (lc == "rhf") || (lc == "restricted") ||
      (lc == "spin-restricted")) {
    sk = SpinorbitalKind::Restricted;
    valid = true;
  } else if ((lc == "u") || (lc == "uhf") || (lc == "unrestricted") ||
             (lc == "spin-unrestricted")) {
    sk = SpinorbitalKind::Unrestricted;
    valid = true;
  } else if ((lc == "g") || (lc == "ghf") || (lc == "general")) {
    sk = SpinorbitalKind::General;
    valid = true;
  }
  return valid;
}

constexpr const char *spinorbital_kind_to_string(const SpinorbitalKind &sk) {
  if (sk == SpinorbitalKind::Restricted)
    return "restricted";
  else if (sk == SpinorbitalKind::Unrestricted)
    return "unrestricted";
  else
    return "general";
}

} // namespace occ::core
