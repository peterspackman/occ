#include <occ/core/linear_algebra.h>
#include <occ/qm/shell.h>
#include <vector>

namespace occ::qm {

using occ::Vec;

std::pair<Mat, Vec> merge_molecular_orbitals(const Mat &, const Mat &,
                                             const Vec &, const Vec &,
                                             bool sort_by_energy = false);
AOBasis merge_basis_sets(const AOBasis &, const AOBasis &);
std::vector<occ::core::Atom> merge_atoms(const std::vector<occ::core::Atom> &,
                                         const std::vector<occ::core::Atom> &);

} // namespace occ::qm
