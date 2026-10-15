#pragma once
#include <string>
#include <string_view>

namespace occ {

/// The occ release version, e.g. "0.9.6".
std::string_view version();

/// The git commit occ was built from, with a "-dirty" suffix if the checkout
/// had uncommitted changes, or "unknown" when built outside a git checkout.
/// Captured at build time, so it is never stale.
std::string_view git_commit();

/// "<version> (<git commit>)", for banners and output provenance.
std::string version_string();

} // namespace occ
