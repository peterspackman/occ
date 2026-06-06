#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/spinorbital.h>
#include <occ/core/util.h>
#include <occ/qm/opmatrix.h>

namespace occ::qm {

// SpinorbitalKind and its string helpers now live in occ_core so low-level
// libraries can use them without depending on occ_qm; re-export the type, its
// enumerators, and the helpers here to preserve the occ::qm:: spellings (and
// ADL for callers operating on the enum).
using occ::core::SpinorbitalKind;
using occ::core::Restricted;
using occ::core::Unrestricted;
using occ::core::General;
using occ::core::get_spinorbital_kind_from_string;
using occ::core::spinorbital_kind_to_string;

template <SpinorbitalKind kind>
constexpr std::pair<size_t, size_t> matrix_dimensions(size_t nbf) {
  switch (kind) {
  case Restricted:
    return {nbf, nbf};
  case Unrestricted:
    return {2 * nbf, nbf};
  case General:
    return {2 * nbf, 2 * nbf};
  }
}

inline constexpr std::pair<size_t, size_t>
matrix_dimensions(SpinorbitalKind kind, size_t nbf) {
  switch (kind) {
  case Unrestricted:
    return {2 * nbf, nbf};
  case General:
    return {2 * nbf, 2 * nbf};
  default:
    return {nbf, nbf};
  }
}

} // namespace occ::qm
