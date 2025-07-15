#pragma once
#include <array>
#include <vector>
#include <string_view>
#include <occ/slater/slaterbasis.h>

namespace occ::slater::basis_sets {

// Build Thakkar Slater basis for all atom types including oxidation states
SlaterBasisSetMap build_thakkar_basis();

} // namespace occ::slater::basis_sets
