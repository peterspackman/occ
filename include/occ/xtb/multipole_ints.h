#pragma once
#include <array>
#include <occ/core/linear_algebra.h>
#include <occ/qm/integral_engine.h>

namespace occ::xtb {

// Per-AO dipole integral matrices: returns three nbf×nbf matrices for the
// (x, y, z) components of <φ_μ | r_α - O_α | φ_ν>.  `origin` is in Bohr.
MatTriple dipole_ao_matrices(qm::IntegralEngine &engine,
                             const Vec3 &origin = Vec3::Zero());

// Per-AO quadrupole integral matrices for <φ_μ | (r_α-O_α)(r_β-O_β) | φ_ν>.
// Returns 6 unique matrices in the order {xx, xy, xz, yy, yz, zz}.
std::array<Mat, 6> quadrupole_ao_matrices(qm::IntegralEngine &engine,
                                          const Vec3 &origin = Vec3::Zero());

} // namespace occ::xtb
