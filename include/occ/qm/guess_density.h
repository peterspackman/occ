#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/log.h>
#include <occ/gto/shell.h>
#include <vector>

namespace occ::qm::guess {

using gto::AOBasis;

int minimal_basis_nao(int Z, bool spherical);
std::vector<double> minimal_basis_occupation_vector(size_t Z, bool spherical);

// SAP (Superposition of Atomic Potentials) guess
Mat compute_sap_matrix(const std::vector<occ::core::Atom> &atoms,
                       const AOBasis &basis,
                       const std::string &sap_basis_name = "sap_grasp_small");

} // namespace occ::qm::guess
