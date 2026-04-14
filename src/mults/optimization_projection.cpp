#include <occ/mults/optimization_projection.h>
#include <stdexcept>

namespace occ::mults {

// ==================== RigidBodyProjection ====================

RigidBodyProjection::RigidBodyProjection(size_t n_molecules)
    : m_n_molecules(n_molecules),
      m_full_dof(6 * n_molecules),
      m_projected_dof(6 * n_molecules - 6) {

    if (n_molecules < 2) {
        throw std::invalid_argument("RigidBodyProjection requires at least 2 molecules");
    }

    build_projection_matrix();
}

void RigidBodyProjection::build_projection_matrix() {
    // For N molecules with 6N DOF, we project out 6 rigid-body modes
    // leaving (6N-6) internal degrees of freedom.
    //
    // Strategy (following Orient's BUILDO):
    // 1. Fix molecule 1 at origin with identity rotation (6 constraints)
    // 2. Optimize molecules 2..N relative to molecule 1
    // 3. Projection matrix P extracts DOF for molecules 2..N

    // Initialize projection matrix: (6N-6) × 6N
    m_P = Mat::Zero(m_projected_dof, m_full_dof);

    // For N=2 (special case - most common):
    // P is 6×12 matrix that extracts molecule 2's coordinates
    // x_proj = [x2, y2, z2, p2x, p2y, p2z]
    // x_full = [x1, y1, z1, p1x, p1y, p1z, x2, y2, z2, p2x, p2y, p2z]

    if (m_n_molecules == 2) {
        // Identity block extracting molecule 2's 6 DOF
        m_P.block(0, 6, 6, 6) = Mat::Identity(6, 6);
    }
    else {
        // For N>2: Extract DOF for molecules 2..N
        // This gives (N-1)*6 = 6N-6 degrees of freedom
        size_t dof_per_molecule = 6;
        for (size_t i = 1; i < m_n_molecules; i++) {
            size_t row_start = (i - 1) * dof_per_molecule;
            size_t col_start = i * dof_per_molecule;
            m_P.block(row_start, col_start, dof_per_molecule, dof_per_molecule) =
                Mat::Identity(dof_per_molecule, dof_per_molecule);
        }
    }
}

Vec RigidBodyProjection::project_gradient(const Vec& grad_full) const {
    if (grad_full.size() != m_full_dof) {
        throw std::invalid_argument("Gradient size mismatch in project_gradient");
    }

    // Project gradient: grad_proj = P^T * grad_full
    // But since we're extracting subspace, P^T * grad = extract relevant components
    return m_P * grad_full;
}

Vec RigidBodyProjection::reconstruct_full(const Vec& x_proj, const Vec& x_base) const {
    if (x_proj.size() != m_projected_dof) {
        throw std::invalid_argument("Projected coordinate size mismatch in reconstruct_full");
    }
    if (x_base.size() != m_full_dof) {
        throw std::invalid_argument("Base coordinate size mismatch in reconstruct_full");
    }

    // Reconstruct: x_full = x_base + P^T * x_proj
    // Molecule 1 stays at x_base[0:6], molecules 2..N are updated
    Vec x_full = x_base;

    // Add projected coordinates to base
    // P^T maps (6N-6) → 6N, filling in molecules 2..N
    x_full += m_P.transpose() * x_proj;

    return x_full;
}

Vec RigidBodyProjection::project_coordinates(const Vec& x_full, const Vec& x_base) const {
    if (x_full.size() != m_full_dof) {
        throw std::invalid_argument("Full coordinate size mismatch in project_coordinates");
    }
    if (x_base.size() != m_full_dof) {
        throw std::invalid_argument("Base coordinate size mismatch in project_coordinates");
    }

    // Project coordinates: x_proj = P * (x_full - x_base)
    Vec dx = x_full - x_base;
    return m_P * dx;
}

// ==================== TwoMoleculeProjection ====================

Vec TwoMoleculeProjection::project_gradient(const Vec& grad_full) const {
    if (grad_full.size() != 12) {
        throw std::invalid_argument("TwoMoleculeProjection expects 12-element gradient");
    }

    // Extract gradient for molecule 2 only (last 6 components)
    // Molecule 1 is fixed, so its gradient components are ignored
    return grad_full.segment<6>(6);
}

Vec TwoMoleculeProjection::reconstruct_full(const Vec& x_proj, const Vec& x_base) const {
    if (x_proj.size() != 6) {
        throw std::invalid_argument("TwoMoleculeProjection expects 6-element projected coordinates");
    }
    if (x_base.size() != 12) {
        throw std::invalid_argument("TwoMoleculeProjection expects 12-element base configuration");
    }

    Vec x_full(12);

    // Molecule 1: Keep at base configuration (fixed)
    x_full.segment<6>(0) = x_base.segment<6>(0);

    // Molecule 2: Add projected coordinates to base
    x_full.segment<6>(6) = x_base.segment<6>(6) + x_proj;

    return x_full;
}

Vec TwoMoleculeProjection::project_coordinates(const Vec& x_full, const Vec& x_base) const {
    if (x_full.size() != 12) {
        throw std::invalid_argument("TwoMoleculeProjection expects 12-element full coordinates");
    }
    if (x_base.size() != 12) {
        throw std::invalid_argument("TwoMoleculeProjection expects 12-element base configuration");
    }

    // Project to molecule 2's coordinates relative to base
    // x_proj = x_full[6:12] - x_base[6:12]
    return x_full.segment<6>(6) - x_base.segment<6>(6);
}

} // namespace occ::mults
