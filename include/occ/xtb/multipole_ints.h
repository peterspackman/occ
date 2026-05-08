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

// Ket-side AO multipole derivative integrals at common origin O = 0.
//
// Dipole: libcint `int1e_irp` evaluates <φ_μ | r_α | ∇_β φ_ν> for the 9
// combinations (α = dipole component, β = spatial gradient direction).
// Returned as 3 MatTriples: `irp[α].(x|y|z)(μ, ν) = <φ_μ | r_α | ∇_β φ_ν>`
// with α the outer index and β = x/y/z encoded in MatTriple.
//
// Use the chain rule
//   ∂D_α(μ, ν, O=0) / ∂R_(atom of ν)_β = - <φ_μ | r_α | ∂_β φ_ν>
//                                       = - irp[α].(β)(μ, ν)
//   ∂D_α(μ, ν, O=0) / ∂R_(atom of μ)_β = + δ_αβ · S(μ, ν) + irp[α].(β)(μ, ν)
// (the second follows from translation invariance of the O = 0 dipole).
using DipoleGradAO = std::array<MatTriple, 3>;

DipoleGradAO dipole_ao_grad(qm::IntegralEngine &engine);

// Quadrupole: libcint `int1e_irrp` evaluates <φ_μ | r_α r_β | ∇_γ φ_ν> with
// 9 × 3 = 27 components per AO pair. By quadrupole symmetry r_α r_β = r_β r_α
// only 6 (α, β) combinations are unique, returned in {xx, xy, xz, yy, yz, zz}
// order to match `quadrupole_ao_matrices`. Each entry holds a MatTriple of 3
// spatial deriv components γ = x/y/z.
//
//   ∂Q_αβ(μ, ν, O=0) / ∂R_(atom of ν)_γ = - <φ_μ | r_α r_β | ∂_γ φ_ν>
//                                        = - irrp[k].(γ)(μ, ν), k ∈ {0..5}
//   ∂Q_αβ(μ, ν, O=0) / ∂R_(atom of μ)_γ =  δ_αγ <φ_μ | r_β | φ_ν>
//                                        + δ_βγ <φ_μ | r_α | φ_ν>
//                                        + irrp[k].(γ)(μ, ν)
using QuadrupoleGradAO = std::array<MatTriple, 6>;

QuadrupoleGradAO quadrupole_ao_grad(qm::IntegralEngine &engine);

} // namespace occ::xtb
