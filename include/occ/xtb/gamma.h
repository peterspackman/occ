#pragma once
#include <occ/core/atom.h>
#include <occ/core/linear_algebra.h>

namespace occ::xtb {

class Gfn2Parameters;

// Per-shell metadata used by the SCC: which atom owns this shell, which
// element-shell-index it corresponds to, the shell-resolved Hubbard hardness,
// the shell self-energy, etc. Built once from (atoms, params) and reused.
struct ShellTable {
  std::vector<int> atom;          // owning atom index
  std::vector<int> elem_shell;    // index within element->shells[]
  Vec hardness;                   // per-shell η (atomic units)
  Vec self_energy_ev;             // per-shell ε (eV — convert at use)
  Vec kcn;                        // per-shell CN coefficient
  Vec shell_poly;                 // per-shell distance polynom coefficient
  Vec ref_occ;                    // per-shell reference occupation
  IVec ang_mom;                   // per-shell angular momentum (l)
  IVec n_quantum;                 // per-shell principal quantum number
  Vec third_order;                // per-shell Γ_3 = third_order_atom * gam3shell[l]
};

// Build the per-shell metadata for the given atoms.
ShellTable build_shell_table(const std::vector<core::Atom> &atoms,
                             const Gfn2Parameters &params);

// Klopman-Ohno gamma matrix with arithmetic averaging (GFN2's convention):
//   - cross-atom: γ_ij(R) = (R^α + g_ij^{-α})^{-1/α}, α = globals.alphaj
//   - same-atom, different shell: γ = g_ij = ½(η_i + η_j)
//   - same shell (diagonal): γ = η_i
// Atomic positions in Bohr.
Mat klopman_ohno_gamma(const std::vector<core::Atom> &atoms,
                       const ShellTable &shells,
                       const Gfn2Parameters &params);

// Analytical gradient of ½ q^T γ q with respect to nuclear positions.
// Only cross-atom γ entries contribute (same-atom γ has no R-dependence).
Mat3N klopman_ohno_gamma_energy_gradient(
    const std::vector<core::Atom> &atoms, const ShellTable &shells,
    const Gfn2Parameters &params, const Mat &gamma_matrix, const Vec &qsh);

} // namespace occ::xtb
