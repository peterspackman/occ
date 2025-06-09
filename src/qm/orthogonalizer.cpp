#include <occ/qm/orthogonalizer.h>

namespace occ::qm {

// Implementation
void CanonicalOrthogonalizer::build(const Mat &overlap, double threshold) {
  auto result = occ::core::conditioning_orthogonalizer(overlap, threshold);
  m_X = result.result;
  m_Xinv = result.result_inverse;
  m_condition_number = result.result_condition_number;
  m_is_computed = true;
}

Mat CanonicalOrthogonalizer::to_orthogonal_basis(const Mat &matrix) const {
  if (!m_is_computed) {
    throw std::runtime_error(
        "Cannot transform: basis orthogonalization not built");
  }
  return m_X.transpose() * matrix * m_X;
}

Mat CanonicalOrthogonalizer::from_orthogonal_basis(const Mat &matrix) const {
  if (!m_is_computed) {
    throw std::runtime_error(
        "Cannot transform: basis orthogonalization not built");
  }
  return m_Xinv.transpose() * matrix * m_Xinv;
}

void CanonicalOrthogonalizer::orthogonalize_molecular_orbitals(
    MolecularOrbitals &mo, const Mat &hamiltonian) const {
  if (!m_is_computed) {
    throw std::runtime_error(
        "Cannot orthogonalize MOs: basis orthogonalization not built");
  }
  mo.update(m_X, hamiltonian); // Assuming MO has an update method that takes X and F
}

} // namespace occ::qm
