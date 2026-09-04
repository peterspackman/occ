#pragma once
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace occ::qm {

/// Where the SCF starts from.
///
/// Every kind here ends in the same place -- diagonalise a Fock matrix built
/// from something cheaper than the answer -- and differs only in what that
/// something is. See <occ/qm/initial_guess.h> for how each is built.
///
/// This lives apart from the rest of the guess API, and depends on nothing, so
/// that input handling can name a guess without taking on a dependency on the
/// SCF itself.
enum class GuessKind {
  /// Choose per system; see `select_guess`.
  Auto,
  /// No guess: diagonalise the bare core Hamiltonian. Always available, and
  /// always the worst starting point that still converges.
  Core,
  /// Superposition of atomic densities: tabulated ground-state occupations
  /// spread over a minimal basis, projected into the orbital basis. Cheap and
  /// good, but only for elements the shipped minimal basis covers.
  Soad,
  /// Superposition of atomic potentials: a fitted effective potential per
  /// element added to the core Hamiltonian. Covers every element, but has no
  /// sound treatment of effective core potentials.
  Sap,
  /// Superposition of converged isolated-atom densities, one small SCF per
  /// element in this calculation's own basis. Costs more than the others and
  /// works wherever the orbital basis does, ECPs included.
  AtomicScf,
};

inline std::string_view guess_kind_name(GuessKind kind) {
  switch (kind) {
  case GuessKind::Auto:
    return "auto";
  case GuessKind::Core:
    return "core";
  case GuessKind::Soad:
    return "soad";
  case GuessKind::Sap:
    return "sap";
  case GuessKind::AtomicScf:
    return "atomic";
  }
  return "unknown";
}

/// The names a user may write, and what each selects. Shared by the command
/// line and by any other input route, so they cannot drift apart.
inline const std::vector<std::pair<std::string, GuessKind>> &
guess_kind_names() {
  static const std::vector<std::pair<std::string, GuessKind>> names{
      {"auto", GuessKind::Auto},       {"core", GuessKind::Core},
      {"hcore", GuessKind::Core},      {"soad", GuessKind::Soad},
      {"sad", GuessKind::Soad},        {"sap", GuessKind::Sap},
      {"atomic", GuessKind::AtomicScf}};
  return names;
}

/// Parse a guess name. Returns nothing for an unrecognised one.
inline std::optional<GuessKind> guess_kind_from_string(std::string_view name) {
  for (const auto &[text, kind] : guess_kind_names())
    if (text == name)
      return kind;
  return std::nullopt;
}

} // namespace occ::qm
