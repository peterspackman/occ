#include <occ/core/linear_algebra.h>
#include <occ/qm/basisset.h>
#include <vector>

namespace occ::qm {

using occ::Vec;
using occ::qm::BasisSet;

std::pair<Mat, Vec> merge_molecular_orbitals(const Mat&, const Mat&, const Vec&, const Vec&);
BasisSet merge_basis_sets(const BasisSet&, const BasisSet&);
std::vector<occ::core::Atom> merge_atoms(const std::vector<occ::core::Atom>&, const std::vector<occ::core::Atom>&);

}
