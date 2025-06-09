#pragma once
#include <occ/core/conditioning_orthogonalizer.h>
#include <occ/core/linear_algebra.h>
#include <occ/qm/mo.h>

namespace occ::qm {

// Basis orthogonalization using symmetric/canonical orthogonalization
class CanonicalOrthogonalizer {
public:
  CanonicalOrthogonalizer() : m_condition_number(0.0), m_is_computed(false) {}

  // Build orthogonalization from overlap matrix
  void build(const Mat &overlap,
             double threshold = 1.0 / std::numeric_limits<double>::epsilon());

  // Access transformation matrices
  inline const Mat &transformation_matrix() const {
    if (!m_is_computed)
      throw std::runtime_error("Orthogonalization not built");
    return m_X;
  }

  inline const Mat &inverse_transformation_matrix() const {
    if (!m_is_computed)
      throw std::runtime_error("Orthogonalization not built");
    return m_Xinv;
  }

  inline double condition_number() const { return m_condition_number; }
  inline bool is_built() const { return m_is_computed; }

  // Transform matrices to/from orthogonal basis
  Mat to_orthogonal_basis(const Mat &matrix) const;
  Mat from_orthogonal_basis(const Mat &matrix) const;

  // Update molecular orbitals using orthogonalization
  void orthogonalize_molecular_orbitals(MolecularOrbitals &mo,
                                        const Mat &hamiltonian) const;

  // Reset state
  inline void reset() {
    m_X = Mat();
    m_Xinv = Mat();
    m_condition_number = 0.0;
    m_is_computed = false;
  }

  bool is_well_conditioned() const {
    return m_is_computed && m_condition_number > 0.0;
  }

private:
  Mat m_X;                   // Orthogonalization transformation matrix
  Mat m_Xinv;                // Inverse transformation matrix
  double m_condition_number; // Condition number of X^T * X
  bool m_is_computed{false};
};

} // namespace occ::qm
