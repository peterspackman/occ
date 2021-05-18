#include <occ/core/linear_algebra.h>
#include <occ/qm/basisset.h>
#include <vector>

namespace occ::qm {

using occ::MatRM;
using occ::Vec;
using occ::qm::BasisSet;

std::pair<MatRM, Vec> merge_molecular_orbitals(const MatRM&, const MatRM&, const Vec&, const Vec&);
BasisSet merge_basis_sets(const BasisSet&, const BasisSet&);
std::vector<libint2::Atom> merge_atoms(const std::vector<libint2::Atom>&, const std::vector<libint2::Atom>&);

}
