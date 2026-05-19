#include <array>
#include <cmath>
#include <occ/core/units.h>
#include <occ/xtb/gamma.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/h0.h>
#include <stdexcept>

namespace occ::xtb {

namespace {

// Mantina/Truhlar atomic radii — same table xtb uses in
// src/param/atomicrad.f90. In Bohr. Index by Z.
constexpr double angstrom_to_bohr = occ::units::ANGSTROM_TO_BOHR;
constexpr std::array<double, 87> atomic_rad_bohr = {
    0.0,
    0.32 * angstrom_to_bohr, 0.37 * angstrom_to_bohr,             // H, He
    1.30 * angstrom_to_bohr, 0.99 * angstrom_to_bohr, 0.84 * angstrom_to_bohr,
    0.75 * angstrom_to_bohr, 0.71 * angstrom_to_bohr, 0.64 * angstrom_to_bohr,
    0.60 * angstrom_to_bohr, 0.62 * angstrom_to_bohr,             // Li-Ne
    1.60 * angstrom_to_bohr, 1.40 * angstrom_to_bohr, 1.24 * angstrom_to_bohr,
    1.14 * angstrom_to_bohr, 1.09 * angstrom_to_bohr, 1.04 * angstrom_to_bohr,
    1.00 * angstrom_to_bohr, 1.01 * angstrom_to_bohr,             // Na-Ar
    2.00 * angstrom_to_bohr, 1.74 * angstrom_to_bohr,             // K, Ca
    1.59 * angstrom_to_bohr, 1.48 * angstrom_to_bohr, 1.44 * angstrom_to_bohr,
    1.30 * angstrom_to_bohr, 1.29 * angstrom_to_bohr, 1.24 * angstrom_to_bohr,
    1.18 * angstrom_to_bohr, 1.17 * angstrom_to_bohr, 1.22 * angstrom_to_bohr,
    1.20 * angstrom_to_bohr,                                      // Sc-Zn
    1.23 * angstrom_to_bohr, 1.20 * angstrom_to_bohr, 1.20 * angstrom_to_bohr,
    1.18 * angstrom_to_bohr, 1.17 * angstrom_to_bohr, 1.16 * angstrom_to_bohr, // Ga-Kr
    2.15 * angstrom_to_bohr, 1.90 * angstrom_to_bohr,             // Rb, Sr
    1.76 * angstrom_to_bohr, 1.64 * angstrom_to_bohr, 1.56 * angstrom_to_bohr,
    1.46 * angstrom_to_bohr, 1.38 * angstrom_to_bohr, 1.36 * angstrom_to_bohr,
    1.34 * angstrom_to_bohr, 1.30 * angstrom_to_bohr, 1.36 * angstrom_to_bohr,
    1.40 * angstrom_to_bohr,                                      // Y-Cd
    1.42 * angstrom_to_bohr, 1.40 * angstrom_to_bohr, 1.40 * angstrom_to_bohr,
    1.37 * angstrom_to_bohr, 1.36 * angstrom_to_bohr, 1.36 * angstrom_to_bohr, // In-Xe
    2.38 * angstrom_to_bohr, 2.06 * angstrom_to_bohr,             // Cs, Ba
    1.94 * angstrom_to_bohr, 1.84 * angstrom_to_bohr, 1.90 * angstrom_to_bohr,
    1.88 * angstrom_to_bohr, 1.86 * angstrom_to_bohr, 1.85 * angstrom_to_bohr,
    1.83 * angstrom_to_bohr,                                      // La-Eu
    1.82 * angstrom_to_bohr, 1.81 * angstrom_to_bohr, 1.80 * angstrom_to_bohr,
    1.79 * angstrom_to_bohr, 1.77 * angstrom_to_bohr, 1.77 * angstrom_to_bohr,
    1.78 * angstrom_to_bohr,                                      // Gd-Yb
    1.74 * angstrom_to_bohr, 1.64 * angstrom_to_bohr, 1.58 * angstrom_to_bohr,
    1.50 * angstrom_to_bohr, 1.41 * angstrom_to_bohr,             // Lu-Re
    1.36 * angstrom_to_bohr, 1.32 * angstrom_to_bohr, 1.30 * angstrom_to_bohr,
    1.30 * angstrom_to_bohr, 1.32 * angstrom_to_bohr,             // Os-Hg
    1.44 * angstrom_to_bohr, 1.45 * angstrom_to_bohr, 1.50 * angstrom_to_bohr,
    1.42 * angstrom_to_bohr, 1.48 * angstrom_to_bohr, 1.46 * angstrom_to_bohr, // Tl-Rn
};

double atomic_rad(int z) {
  if (z < 1 || z > 86) {
    throw std::runtime_error("h0: unsupported element Z=" + std::to_string(z));
  }
  return atomic_rad_bohr[z];
}

// Build the GFN2 K_l1l2 shell-pair scaling matrix from the per-l kshell array
// + the {ksp, ksd, kpd} overrides.
//   K[l, l]   = kshell[l]
//   K[l1, l2] = 0.5 (kshell[l1] + kshell[l2])  unless overridden
//   ksd, kpd, ksp set the overrides for {0,2}, {1,2}, {0,1} respectively.
Eigen::Matrix4d build_kscale(const GlobalParam &g) {
  Eigen::Matrix4d K;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      K(i, j) = (i == j) ? g.kshell[i]
                         : 0.5 * (g.kshell[i] + g.kshell[j]);
  if (g.ksp != 0.0) { K(0, 1) = K(1, 0) = g.ksp; }
  if (g.ksd != 0.0) { K(0, 2) = K(2, 0) = g.ksd; }
  if (g.kpd != 0.0) { K(1, 2) = K(2, 1) = g.kpd; }
  return K;
}

// shellPoly distance enhancement factor:
//   r0 = R_atom_A + R_atom_B   (Mantina radii, Bohr)
//   r  = R_AB / r0
//   rf1 = 1 + 0.01 * iPoly * r^0.5
//   rf2 = 1 + 0.01 * jPoly * r^0.5
//   shellPoly = rf1 * rf2
double shell_poly_factor(double i_poly, double j_poly,
                         double i_rad, double j_rad, double r_ab) {
  const double r0 = i_rad + j_rad;
  const double r = r_ab / r0;
  const double sr = std::sqrt(r);
  const double rf1 = 1.0 + 0.01 * i_poly * sr;
  const double rf2 = 1.0 + 0.01 * j_poly * sr;
  return rf1 * rf2;
}

} // namespace

Vec compute_self_energies(const ShellTable &shells, const Vec &cn) {
  const int n = static_cast<int>(shells.atom.size());
  Vec se(n);
  for (int i = 0; i < n; ++i) {
    se(i) = shells.self_energy_ev(i) - shells.kcn(i) * cn(shells.atom[i]);
  }
  return se;
}

Mat build_h0(const std::vector<core::Atom> &atoms,
             const Gfn2Parameters &params, const ShellTable &shells,
             const gto::AOBasis &basis, const Mat &overlap, const Vec &cn) {
  const auto &g = params.globals();
  const Eigen::Matrix4d K = build_kscale(g);
  const double w_exp = 0.5; // hardcoded in xtb's gfn2.f90

  Vec se_ev = compute_self_energies(shells, cn);

  const int nbf = static_cast<int>(basis.nbf());
  Mat H0 = Mat::Zero(nbf, nbf);

  if (basis.size() != shells.atom.size()) {
    throw std::runtime_error(
        "build_h0: AOBasis shell count (" + std::to_string(basis.size()) +
        ") does not match ShellTable size (" +
        std::to_string(shells.atom.size()) + ")");
  }

  // Cache per-shell info pulled from atoms / parameters.
  std::vector<double> slater(shells.atom.size());
  std::vector<double> a_rad(atoms.size());
  for (size_t i = 0; i < atoms.size(); ++i) {
    a_rad[i] = atomic_rad(atoms[i].atomic_number);
  }
  for (size_t s = 0; s < shells.atom.size(); ++s) {
    const auto *e = params.element(atoms[shells.atom[s]].atomic_number);
    slater[s] = e->shells[shells.elem_shell[s]].slater_exponent;
  }

  const auto &first_bf = basis.first_bf();
  // Walk shell pairs.
  for (size_t si = 0; si < basis.size(); ++si) {
    const int ai = shells.atom[si];
    const int li = shells.ang_mom(si);
    const int bf_i0 = first_bf[si];
    const int n_i = static_cast<int>(basis[si].size());
    const double zi = slater[si];
    const double poly_i = shells.shell_poly(si);

    // Diagonal block (same shell, same atom): H0[bf,bf] = self_energy[shell],
    // off-block at same atom is zero (overlap is zero by orthogonality).
    for (int mu = 0; mu < n_i; ++mu) {
      H0(bf_i0 + mu, bf_i0 + mu) = se_ev(si);
    }

    for (size_t sj = 0; sj < si; ++sj) {
      const int aj = shells.atom[sj];
      if (ai == aj) continue; // same-atom different-shell: H0 = 0
      const int lj = shells.ang_mom(sj);
      const int bf_j0 = first_bf[sj];
      const int n_j = static_cast<int>(basis[sj].size());
      const double zj = slater[sj];
      const double poly_j = shells.shell_poly(sj);

      const double dx = atoms[ai].x - atoms[aj].x;
      const double dy = atoms[ai].y - atoms[aj].y;
      const double dz = atoms[ai].z - atoms[aj].z;
      const double r_ab = std::sqrt(dx * dx + dy * dy + dz * dz);

      // K_l1l2 with ΔEN polynomial (valence-valence branch only — GFN2's
      // param tables do not include any non-valence shells).
      const auto *e_i = params.element(atoms[ai].atomic_number);
      const auto *e_j = params.element(atoms[aj].atomic_number);
      // xtb stores enshell per angular momentum; GFN2 sets all four to the
      // same value (g.enscale, =2.0). The pair coupling factor is
      //   0.005 * (enshell_l_i + enshell_l_j) = 0.01 * g.enscale
      // For full GFN1 generality this would need per-l values.
      const double den = (e_i->pauling_en - e_j->pauling_en);
      const double den2 = den * den;
      const double enpoly_coef = 0.01 * g.enscale;
      const double enpoly = 1.0 + enpoly_coef * den2 *
                                      (1.0 + g.enscale4 * den2);
      double km = K(li, lj) * enpoly; // pairParam = 1 in GFN2
      // Slater-exponent steepness factor.
      const double zfac = (2.0 * std::sqrt(zi * zj) / (zi + zj));
      km *= std::pow(zfac, w_exp);

      // shellPoly distance polynomial.
      const double sp = shell_poly_factor(poly_i, poly_j, a_rad[ai], a_rad[aj],
                                          r_ab);

      // Average self-energy (eV).
      const double hav = 0.5 * (se_ev(si) + se_ev(sj)) * sp;

      // Fill the (n_j × n_i) block.
      for (int mu = 0; mu < n_i; ++mu) {
        for (int nu = 0; nu < n_j; ++nu) {
          const double s_ij = overlap(bf_i0 + mu, bf_j0 + nu);
          const double v = s_ij * km * hav;
          H0(bf_i0 + mu, bf_j0 + nu) = v;
          H0(bf_j0 + nu, bf_i0 + mu) = v;
        }
      }
    }
  }

  // H0 is currently in eV (because self_energy_ev is in eV); convert to
  // Hartree to be consistent with everything else.
  H0 *= 1.0 / occ::units::AU_TO_EV; // eV → Hartree
  return H0;
}

} // namespace occ::xtb
