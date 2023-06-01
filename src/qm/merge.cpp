#include <occ/qm/merge.h>

namespace occ::qm {

std::pair<Mat, Vec> merge_molecular_orbitals(const Mat &mo_a, const Mat &mo_b,
                                             const Vec &e_a, const Vec &e_b,
                                             bool sort_by_energy) {
    Mat merged =
        Mat::Zero(mo_a.rows() + mo_b.rows(), mo_a.cols() + mo_b.cols());
    Vec merged_energies(e_a.rows() + e_b.rows());
    merged_energies.topRows(e_a.rows()) = e_a;
    merged_energies.bottomRows(e_b.rows()) = e_b;
    std::vector<Eigen::Index> idxs;
    idxs.reserve(merged_energies.rows());
    for (Eigen::Index i = 0; i < merged_energies.rows(); i++)
        idxs.push_back(i);
    if (sort_by_energy) {
        std::stable_sort(idxs.begin(), idxs.end(),
                         [&merged_energies](Eigen::Index a, Eigen::Index b) {
                             return merged_energies(a) < merged_energies(b);
                         });
    }
    Vec sorted_energies(merged_energies.rows());
    for (Eigen::Index i = 0; i < merged_energies.rows(); i++) {
        Eigen::Index c = idxs[i];
        sorted_energies(i) = merged_energies(c);
        if (c >= mo_a.cols()) {
            merged.col(i).bottomRows(mo_b.rows()) = mo_b.col(c - mo_a.cols());
        } else {
            merged.col(i).topRows(mo_a.rows()) = mo_a.col(c);
        }
    }
    return {merged, sorted_energies};
}

AOBasis merge_basis_sets(const AOBasis &basis_a, const AOBasis &basis_b) {
    AOBasis merged = basis_a;
    merged.merge(basis_b);
    return merged;
}

std::vector<occ::core::Atom>
merge_atoms(const std::vector<occ::core::Atom> &atoms_a,
            const std::vector<occ::core::Atom> &atoms_b) {
    std::vector<occ::core::Atom> merged = atoms_a;
    merged.reserve(atoms_a.size() + atoms_b.size());
    merged.insert(merged.end(), atoms_b.begin(), atoms_b.end());
    return merged;
}
} // namespace occ::qm
