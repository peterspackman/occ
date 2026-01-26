#pragma once
#include <array>
#include <occ/gto/shell.h>
#include <vector>

namespace occ::qm::basis_sets {

// Build shells for specific atoms
std::vector<Shell>
build_sto3g_shells(const std::vector<occ::core::Atom> &atoms);

} // namespace occ::qm::basis_sets
