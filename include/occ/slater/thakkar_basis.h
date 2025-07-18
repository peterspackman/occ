#pragma once
#include <array>
#include <occ/slater/slaterbasis.h>
#include <string_view>
#include <vector>

namespace occ::slater::basis_sets {

// Build Thakkar Slater basis for all atom types including oxidation states
SlaterBasisSetMap build_thakkar_basis();

} // namespace occ::slater::basis_sets
