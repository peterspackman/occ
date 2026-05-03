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

} // namespace occ::xtb
