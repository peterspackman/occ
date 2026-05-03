#include <cmath>
#include <occ/xtb/gamma.h>
#include <occ/xtb/gfn2_parameters.h>
#include <stdexcept>

namespace occ::xtb {

ShellTable build_shell_table(const std::vector<core::Atom> &atoms,
                             const Gfn2Parameters &params) {
  ShellTable t;
  // First pass: count shells.
  size_t n = 0;
  for (const auto &a : atoms) {
    const auto *e = params.element(a.atomic_number);
    if (!e) {
      throw std::runtime_error("ShellTable: missing parameters for Z=" +
                               std::to_string(a.atomic_number));
    }
    n += e->shells.size();
  }
  t.atom.resize(n);
  t.elem_shell.resize(n);
  t.hardness.resize(n);
  t.self_energy_ev.resize(n);
  t.kcn.resize(n);
  t.shell_poly.resize(n);
  t.ref_occ.resize(n);
  t.ang_mom.resize(n);
  t.n_quantum.resize(n);
  t.third_order.resize(n);

  const auto &g = params.globals();
  size_t s = 0;
  for (size_t ai = 0; ai < atoms.size(); ++ai) {
    const auto *e = params.element(atoms[ai].atomic_number);
    for (size_t si = 0; si < e->shells.size(); ++si) {
      const auto &sh = e->shells[si];
      t.atom[s] = static_cast<int>(ai);
      t.elem_shell[s] = static_cast<int>(si);
      // shellHardness[ish, Zp] = atomicHardness * (1 + lpar_l)
      t.hardness(s) = e->atomic_hardness * (1.0 + sh.shell_hardness_au);
      t.self_energy_ev(s) = sh.self_energy_ev;
      t.kcn(s) = sh.kcn_au;
      t.shell_poly(s) = sh.shell_poly;
      t.ref_occ(s) = sh.ref_occ;
      t.ang_mom(s) = sh.l;
      t.n_quantum(s) = sh.n;
      // Per-shell third-order coefficient. xtb uses gam3shell[iKind, l]; for
      // GFN2 the two iKind values are identical so we read entry [l][0].
      const int l = sh.l;
      const double gam3_factor = (l >= 0 && l < 4) ? g.gam3shell[l][0] : 0.0;
      t.third_order(s) = e->third_order_atom_au * gam3_factor;
      ++s;
    }
  }
  return t;
}

Mat klopman_ohno_gamma(const std::vector<core::Atom> &atoms,
                       const ShellTable &shells,
                       const Gfn2Parameters &params) {
  const double g_exp = params.globals().alphaj;
  if (g_exp <= 0.0) {
    throw std::runtime_error(
        "klopman_ohno_gamma: alphaj must be positive (got " +
        std::to_string(g_exp) + ")");
  }
  const double inv_g_exp = 1.0 / g_exp;

  const int n = static_cast<int>(shells.atom.size());
  Mat J = Mat::Zero(n, n);

  for (int i = 0; i < n; ++i) {
    const int ai = shells.atom[i];
    // Diagonal
    J(i, i) = shells.hardness(i);
    for (int j = 0; j < i; ++j) {
      const int aj = shells.atom[j];
      const double gij = 0.5 * (shells.hardness(i) + shells.hardness(j));
      double v;
      if (ai == aj) {
        // Same atom, different shell — limit r → 0.
        v = gij;
      } else {
        const double dx = atoms[ai].x - atoms[aj].x;
        const double dy = atoms[ai].y - atoms[aj].y;
        const double dz = atoms[ai].z - atoms[aj].z;
        const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
        const double r_to_g = std::pow(r, g_exp);
        const double inv_g_to_g = std::pow(1.0 / gij, g_exp);
        v = std::pow(r_to_g + inv_g_to_g, -inv_g_exp);
      }
      J(i, j) = v;
      J(j, i) = v;
    }
  }
  return J;
}

Mat3N klopman_ohno_gamma_energy_gradient(
    const std::vector<core::Atom> &atoms, const ShellTable &shells,
    const Gfn2Parameters &params, const Mat & /*gamma_matrix*/,
    const Vec &qsh) {
  // d/dR_A ( ½ Σ_ij q_i q_j γ_ij ).
  // Only off-atom (Ai ≠ Aj) entries are R-dependent; same-atom γ is constant.
  // After expanding the i↔j symmetry, the ½ in ½ q^T γ q cancels with the
  // factor-2 from summing both orderings of each unordered pair, giving:
  //   dE/dR_A = Σ_{i on A, j off A} q_i q_j · (dγ/dR)·(r_A − r_{A_j})/R
  // The double `for` loop below visits each ordered (i, j) once and assigns
  // the contribution to grad(A_i). The (j, i) visit handles A_j by symmetry.
  const double g_exp = params.globals().alphaj;

  const int n_atoms = static_cast<int>(atoms.size());
  const int n_shells = static_cast<int>(shells.atom.size());
  Mat3N grad = Mat3N::Zero(3, n_atoms);

  for (int i = 0; i < n_shells; ++i) {
    const int Ai = shells.atom[i];
    for (int j = 0; j < n_shells; ++j) {
      const int Aj = shells.atom[j];
      if (Ai == Aj) continue;
      const double dx = atoms[Ai].x - atoms[Aj].x;
      const double dy = atoms[Ai].y - atoms[Aj].y;
      const double dz = atoms[Ai].z - atoms[Aj].z;
      const double R2 = dx * dx + dy * dy + dz * dz;
      const double R = std::sqrt(R2);
      // Recompute γ to avoid relying on `gamma_matrix`'s internal layout
      // matching what we expect; cheap.
      const double gij = 0.5 * (shells.hardness(i) + shells.hardness(j));
      const double r_to_g = std::pow(R, g_exp);
      const double inv_g_to_g = std::pow(1.0 / gij, g_exp);
      const double g_val = std::pow(r_to_g + inv_g_to_g, -1.0 / g_exp);
      // dγ/dR = -R^(α-1) · γ^(α+1)
      const double dgamma_dR = -std::pow(R, g_exp - 1.0) *
                                std::pow(g_val, g_exp + 1.0);
      const double scal = qsh(i) * qsh(j) * dgamma_dR / R;
      grad(0, Ai) += scal * dx;
      grad(1, Ai) += scal * dy;
      grad(2, Ai) += scal * dz;
      // The (j,i) iteration covers atom Aj with the opposite sign, so we
      // do not need to push to Aj here.
    }
  }
  return grad;
}

} // namespace occ::xtb
