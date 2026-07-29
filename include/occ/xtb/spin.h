#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>

namespace occ::xtb {

class Gfn2Parameters;
struct ShellTable;

/// On-site spin-coupling matrix W (n_shells × n_shells, Hartree), block
/// diagonal by atom — the GFN-xTB spin interaction has no inter-atomic part.
///
/// Given shell magnetizations m (Mulliken pop(α) − pop(β)), the spin energy is
/// `E_spin = ½ mᵀWm` and the conjugate shell potential is `v = Wm`. The
/// constants are negative, so polarizing the spin density lowers the energy.
///
/// `scale` multiplies every entry; 0 disables spin polarization, recovering
/// the common-Fock open-shell treatment of plain `xtb --uhf`. Any other value
/// requires a `spin_constants` block for every element in the parameter set,
/// and throws naming the element if one is missing.
Mat spin_coupling_matrix(const std::vector<core::Atom> &atoms,
                         const ShellTable &shells,
                         const Gfn2Parameters &params, double scale = 1.0);

} // namespace occ::xtb
