#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>
#include <occ/gto/shell.h>

namespace occ::xtb {

class Gfn2Parameters;
struct ShellTable;

// Compute the per-shell self-energies including the CN-shift:
//   ε_shell = ε_l - kCN_l * CN[atom]
// Returned in eV (matches xtb's internal storage). Convert at use.
Vec compute_self_energies(const ShellTable &shells, const Vec &cn);

// Build the H0 (extended Hückel core Hamiltonian) matrix in the AO basis.
// The returned matrix is in atomic units (Hartree), AO-by-AO. AO ordering is
// whatever the supplied AOBasis uses (libcint Spherical for occ::xtb).
//
// The function expects that the AOBasis was constructed from the same
// `atoms` and parameters via `build_aobasis(...)` so the per-shell mapping is
// trivial (ShellTable shell index == AOBasis shell index).
Mat build_h0(const std::vector<core::Atom> &atoms,
             const Gfn2Parameters &params, const ShellTable &shells,
             const gto::AOBasis &basis, const Mat &overlap, const Vec &cn);

} // namespace occ::xtb
