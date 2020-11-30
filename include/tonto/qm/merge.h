#include <tonto/core/linear_algebra.h>
#include <tonto/qm/basisset.h>
#include <vector>

namespace tonto::qm {

using tonto::MatRM;
using tonto::Vec;
using tonto::qm::BasisSet;

std::pair<MatRM, Vec> merge_molecular_orbitals(const MatRM&, const MatRM&, const Vec&, const Vec&);
BasisSet merge_basis_sets(const BasisSet&, const BasisSet&);
std::vector<libint2::Atom> merge_atoms(const std::vector<libint2::Atom>&, const std::vector<libint2::Atom>&);

}
