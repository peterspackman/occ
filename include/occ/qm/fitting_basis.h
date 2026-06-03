#pragma once
#include <string>

namespace occ::qm {

// Which family of auxiliary (fitting) basis to resolve.
enum class FittingKind {
  JK,          // Coulomb(+exchange) fitting for SCF density fitting
  Correlation, // RI/C fitting for RI-MP2 / DF-CCSD
};

// Resolve the recommended auxiliary basis name for a given orbital basis.
//
// Consults share/basis/fitting_defaults.json (located via OCC_DATA_PATH /
// set_data_directory): an exact, case-insensitive match on the orbital basis
// name is preferred, otherwise the kind's default is returned. If the data file
// is missing or unreadable, built-in defaults are used. The returned name is a
// basis that can be passed straight to AOBasis::load / load_basis_set.
std::string resolve_fitting_basis(const std::string &orbital_basis_name,
                                  FittingKind kind);

// Number of basis functions above which the automatic acceleration policy
// switches exact exchange from DF-K to seminumerical COSX. Read from
// fitting_defaults.json ("policy.cosx_nbf_crossover"); defaults to 600.
int cosx_nbf_crossover();

} // namespace occ::qm
