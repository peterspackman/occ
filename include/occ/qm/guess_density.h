#pragma once
#include <occ/core/log.h>
#include <vector>

namespace occ::qm::guess {

int minimal_basis_nao(int Z, bool spherical);
std::vector<double> minimal_basis_occupation_vector(size_t Z, bool spherical);

} // namespace occ::qm::guess
