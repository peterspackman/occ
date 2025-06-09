#pragma once
#include <array>
#include <vector>
#include <occ/qm/shell.h>

namespace occ::qm::basis_sets {

// Build shells for specific atoms
std::vector<Shell> build_sto3g_shells(const std::vector<occ::core::Atom>& atoms);

} // namespace occ::qm::basis_sets
