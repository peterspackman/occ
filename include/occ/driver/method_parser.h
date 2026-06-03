#pragma once
#include <array>
#include <occ/core/util.h>
#include <occ/qm/spinorbital.h>
#include <string>
#include <string_view>
#include <utility>

namespace occ::driver {

enum class MethodKind {
  HF,
  DFT,
  MP2,
  CCSD,
  CCSD_T,
  GFN2,
};

struct MethodSpec {
  std::string base_method; ///< name with dispersion + "ri-/df-/thc-" stripped
  std::string dispersion;  ///< "", "d4" or "xdm"
  MethodKind kind{MethodKind::DFT}; ///< classified method family
  std::string backend; ///< correlation backend from a name prefix: "", "df" or "thc"

  MethodSpec(const std::string &base, const std::string &disp = "",
             MethodKind k = MethodKind::DFT)
      : base_method(base), dispersion(disp), kind(k) {}
};

namespace impl {

// Dispersion suffixes recognised on a method string (e.g. "pbe-d4"). Only
// suffixes whose correction the drivers actually apply are listed; anything
// else (e.g. the "-d" in the functional name "b97-d") is left as part of the
// base name for libxc to resolve.
inline constexpr std::array<std::string_view, 2> dispersion_suffixes{"d4",
                                                                      "xdm"};

// Leading prefixes that select a correlation backend, e.g. "ri-ccsd(t)",
// "thc-mp2". Only applied when the remainder is a correlation method
// (MP2/CCSD/CCSD(T)); "ri" and "df" are synonyms for density fitting.
inline constexpr auto backend_prefixes =
    std::to_array<std::pair<std::string_view, std::string_view>>({
        {"ri-", "df"},
        {"df-", "df"},
        {"thc-", "thc"},
    });

// Exact (lowercased) method-name aliases -> kind. DFT is the implicit fallback
// for any base name not listed here, so DFT functionals need no entries. To
// recognise a new spelling of an existing method, add a line.
inline constexpr auto method_aliases =
    std::to_array<std::pair<std::string_view, MethodKind>>({
        {"hf", MethodKind::HF},
        {"rhf", MethodKind::HF},
        {"uhf", MethodKind::HF},
        {"ghf", MethodKind::HF},
        {"scf", MethodKind::HF},
        {"hartree-fock", MethodKind::HF},
        {"hartree fock", MethodKind::HF},
        {"hartreefock", MethodKind::HF},
        {"mp2", MethodKind::MP2},
        {"rmp2", MethodKind::MP2},
        {"ump2", MethodKind::MP2},
        {"moller-plesset", MethodKind::MP2},
        {"moller plesset", MethodKind::MP2},
        {"ccsd", MethodKind::CCSD},
        {"rccsd", MethodKind::CCSD},
        {"uccsd", MethodKind::CCSD},
        {"ccsd(t)", MethodKind::CCSD_T},
        {"ccsd-t", MethodKind::CCSD_T},
        {"ccsd_t", MethodKind::CCSD_T},
        {"ccsdt", MethodKind::CCSD_T},
        {"rccsd(t)", MethodKind::CCSD_T},
        {"uccsd(t)", MethodKind::CCSD_T},
        {"gfn2", MethodKind::GFN2},
        {"gfn2-xtb", MethodKind::GFN2},
        {"gfn2_xtb", MethodKind::GFN2},
    });

inline MethodKind classify(const std::string &lowercased_base) {
  for (const auto &[name, k] : method_aliases)
    if (lowercased_base == name)
      return k;
  return MethodKind::DFT;
}

inline bool is_correlation(MethodKind k) {
  return k == MethodKind::MP2 || k == MethodKind::CCSD ||
         k == MethodKind::CCSD_T;
}

// Split off a dispersion suffix and classify the base (no backend prefix).
inline MethodSpec split_and_classify(const std::string &name) {
  std::string base = name;
  std::string dispersion;
  size_t dash_pos = name.rfind('-');
  if (dash_pos != std::string::npos) {
    std::string suffix = occ::util::to_lower_copy(name.substr(dash_pos + 1));
    for (std::string_view d : dispersion_suffixes) {
      if (suffix == d) {
        base = name.substr(0, dash_pos);
        dispersion = suffix;
        break;
      }
    }
  }
  return MethodSpec(base, dispersion,
                    classify(occ::util::to_lower_copy(base)));
}

} // namespace impl

/**
 * @brief Parse a method string into base method, dispersion, kind and backend.
 *
 * Single source of truth: a leading "ri-"/"df-"/"thc-" backend prefix (only for
 * correlation methods) is split off, then a dispersion suffix, then the base is
 * classified. Anything not matching a known method alias is a DFT functional.
 *
 * Examples:
 *   "pbe-d4"      -> {base "pbe",    disp "d4", kind DFT}
 *   "hf-d4"       -> {base "hf",     disp "d4", kind HF}
 *   "ccsd(t)"     -> {base "ccsd(t)",           kind CCSD_T, backend ""}
 *   "ri-ccsd(t)"  -> {base "ccsd(t)",           kind CCSD_T, backend "df"}
 *   "thc-mp2"     -> {base "mp2",               kind MP2,    backend "thc"}
 *   "b97-d"       -> {base "b97-d",             kind DFT}  (prefix/suffix kept)
 */
inline MethodSpec parse_method_string(const std::string &method_string) {
  std::string lc = occ::util::to_lower_copy(method_string);
  for (const auto &[prefix, backend] : impl::backend_prefixes) {
    if (lc.rfind(prefix, 0) == 0) {
      MethodSpec inner =
          impl::split_and_classify(method_string.substr(prefix.size()));
      // A backend prefix only applies to correlation methods; otherwise the
      // prefix is part of the (functional) name.
      if (impl::is_correlation(inner.kind)) {
        inner.backend = std::string(backend);
        return inner;
      }
      break;
    }
  }
  return impl::split_and_classify(method_string);
}

inline MethodKind method_kind_from_string(const std::string &name) {
  return parse_method_string(name).kind;
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
  case MethodKind::CCSD:
  case MethodKind::CCSD_T: {
    // Restricted closed-shell only; an unrestricted reference is flagged later.
    if (lc[0] == 'u' || multiplicity > 1)
      return qm::SpinorbitalKind::Unrestricted;
    else
      return qm::SpinorbitalKind::Restricted;
  }
  }
}

} // namespace occ::driver
