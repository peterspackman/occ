#include <array>
#include <cmath>
#include <occ/core/units.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/h0_gradient.h>
#include <stdexcept>

namespace occ::xtb {

namespace {

// Same Mantina/Truhlar atomic radii table as h0.cpp uses for shellPoly.
// Duplicated here (small, immutable, easier than exposing).
constexpr double angstrom_to_bohr = occ::units::ANGSTROM_TO_BOHR;
constexpr std::array<double, 87> atomic_rad_bohr = {
    0.0,
    0.32 * angstrom_to_bohr, 0.37 * angstrom_to_bohr,
    1.30 * angstrom_to_bohr, 0.99 * angstrom_to_bohr, 0.84 * angstrom_to_bohr,
    0.75 * angstrom_to_bohr, 0.71 * angstrom_to_bohr, 0.64 * angstrom_to_bohr,
    0.60 * angstrom_to_bohr, 0.62 * angstrom_to_bohr,
    1.60 * angstrom_to_bohr, 1.40 * angstrom_to_bohr, 1.24 * angstrom_to_bohr,
    1.14 * angstrom_to_bohr, 1.09 * angstrom_to_bohr, 1.04 * angstrom_to_bohr,
    1.00 * angstrom_to_bohr, 1.01 * angstrom_to_bohr,
    2.00 * angstrom_to_bohr, 1.74 * angstrom_to_bohr,
    1.59 * angstrom_to_bohr, 1.48 * angstrom_to_bohr, 1.44 * angstrom_to_bohr,
    1.30 * angstrom_to_bohr, 1.29 * angstrom_to_bohr, 1.24 * angstrom_to_bohr,
    1.18 * angstrom_to_bohr, 1.17 * angstrom_to_bohr, 1.22 * angstrom_to_bohr,
    1.20 * angstrom_to_bohr,
    1.23 * angstrom_to_bohr, 1.20 * angstrom_to_bohr, 1.20 * angstrom_to_bohr,
    1.18 * angstrom_to_bohr, 1.17 * angstrom_to_bohr, 1.16 * angstrom_to_bohr,
    2.15 * angstrom_to_bohr, 1.90 * angstrom_to_bohr,
    1.76 * angstrom_to_bohr, 1.64 * angstrom_to_bohr, 1.56 * angstrom_to_bohr,
    1.46 * angstrom_to_bohr, 1.38 * angstrom_to_bohr, 1.36 * angstrom_to_bohr,
    1.34 * angstrom_to_bohr, 1.30 * angstrom_to_bohr, 1.36 * angstrom_to_bohr,
    1.40 * angstrom_to_bohr,
    1.42 * angstrom_to_bohr, 1.40 * angstrom_to_bohr, 1.40 * angstrom_to_bohr,
    1.37 * angstrom_to_bohr, 1.36 * angstrom_to_bohr, 1.36 * angstrom_to_bohr,
    2.38 * angstrom_to_bohr, 2.06 * angstrom_to_bohr,
    1.94 * angstrom_to_bohr, 1.84 * angstrom_to_bohr, 1.90 * angstrom_to_bohr,
    1.88 * angstrom_to_bohr, 1.86 * angstrom_to_bohr, 1.85 * angstrom_to_bohr,
    1.83 * angstrom_to_bohr,
    1.82 * angstrom_to_bohr, 1.81 * angstrom_to_bohr, 1.80 * angstrom_to_bohr,
    1.79 * angstrom_to_bohr, 1.77 * angstrom_to_bohr, 1.77 * angstrom_to_bohr,
    1.78 * angstrom_to_bohr,
    1.74 * angstrom_to_bohr, 1.64 * angstrom_to_bohr, 1.58 * angstrom_to_bohr,
    1.50 * angstrom_to_bohr, 1.41 * angstrom_to_bohr,
    1.36 * angstrom_to_bohr, 1.32 * angstrom_to_bohr, 1.30 * angstrom_to_bohr,
    1.30 * angstrom_to_bohr, 1.32 * angstrom_to_bohr,
    1.44 * angstrom_to_bohr, 1.45 * angstrom_to_bohr, 1.50 * angstrom_to_bohr,
    1.42 * angstrom_to_bohr, 1.48 * angstrom_to_bohr, 1.46 * angstrom_to_bohr,
};

double atomic_rad(int z) {
  if (z < 1 || z > 86) {
    throw std::runtime_error("h0_gradient: unsupported element Z=" +
                             std::to_string(z));
  }
  return atomic_rad_bohr[z];
}

// Build the K_l1l2 shell-pair scaling matrix exactly as h0.cpp does. Kept
// internal here so we don't expose an internal symbol.
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

// shellPoly enhancement factor and its derivative wrt R_AB.
//   r0 = R_atom_A + R_atom_B (Mantina radii, Bohr)
//   r  = R_AB / r0
//   rf1 = 1 + 0.01 * iPoly * sqrt(r)
//   rf2 = 1 + 0.01 * jPoly * sqrt(r)
//   Π   = rf1 * rf2
struct PolyFactor {
  double pi;     // Π(R_AB)
  double dpi_dR; // dΠ/dR_AB
};

PolyFactor shell_poly_with_grad(double i_poly, double j_poly, double i_rad,
                                double j_rad, double r_ab) {
  const double r0 = i_rad + j_rad;
  const double r = r_ab / r0;
  const double sr = std::sqrt(r);
  const double rf1 = 1.0 + 0.01 * i_poly * sr;
  const double rf2 = 1.0 + 0.01 * j_poly * sr;
  const double pi = rf1 * rf2;
  // dΠ/dR_AB = (dΠ/dr)·(dr/dR_AB) = (dΠ/dr)/r0
  // dΠ/dr   = (0.01/(2·sqrt(r)))·(i_poly · rf2 + j_poly · rf1)
  const double dpi_dr = (0.005 / sr) * (i_poly * rf2 + j_poly * rf1);
  return {pi, dpi_dr / r0};
}

} // namespace

Mat3N h0_scc_gradient(const std::vector<core::Atom> &atoms,
                      const Gfn2Parameters &params, const ShellTable &shells,
                      const gto::AOBasis &basis, qm::IntegralEngine &engine,
                      const Mat &S, const Mat &P, const Mat &W,
                      const Vec &V_shell, const Vec &cn,
                      const std::vector<Mat3N> &dcn) {
  const auto &g = params.globals();
  const Eigen::Matrix4d K = build_kscale(g);
  const double w_exp = 0.5; // hardcoded in xtb's gfn2.f90
  const double inv_au_to_ev = 1.0 / occ::units::AU_TO_EV;

  const int n_atoms = static_cast<int>(atoms.size());
  const int nbf = static_cast<int>(basis.nbf());
  if (basis.size() != shells.atom.size()) {
    throw std::runtime_error("h0_scc_gradient: basis/shells size mismatch");
  }

  Mat3N grad = Mat3N::Zero(3, n_atoms);

  // Per-shell self-energy in eV (with CN-shift applied), and the derived
  // per-shell Slater exponent + atomic radius for repeated lookup.
  Vec se_ev(shells.atom.size());
  Vec slater(shells.atom.size());
  std::vector<double> a_rad(n_atoms);
  for (int i = 0; i < n_atoms; ++i) {
    a_rad[i] = atomic_rad(atoms[i].atomic_number);
  }
  for (size_t s = 0; s < shells.atom.size(); ++s) {
    se_ev(s) = shells.self_energy_ev(s) - shells.kcn(s) * cn(shells.atom[s]);
    const auto *e = params.element(atoms[shells.atom[s]].atomic_number);
    slater(s) = e->shells[shells.elem_shell[s]].slater_exponent;
  }

  // Build:
  //   X_μν   = H0_μν / S_μν  (the part of H0 that multiplies the overlap).
  //            For the diagonal we set X_μμ = se_ev/AU_TO_EV (so X .* S = H0).
  //   Z_μν   = P_μν · X_μν − W_μν  →  Σ Z·∂S/∂R captures both the H0-via-S
  //            term and the Pulay term in one pass.
  //
  // Also accumulate two scalar accumulators for chain-rule contributions:
  //   dE_dCN(B): Σ Z-of-H0 weighted by kCN(s) on shell s of atom B
  //   gradient contributions per atom from ∂Π/∂R (loop directly below)
  Mat X = Mat::Zero(nbf, nbf);
  // Diagonal: H0[μ,μ] = self_energy[shell of μ] / AU_TO_EV
  const auto &bf_to_shell = basis.bf_to_shell();
  for (int mu = 0; mu < nbf; ++mu) {
    X(mu, mu) = se_ev(bf_to_shell[mu]) * inv_au_to_ev;
  }

  // dE/dCN_B accumulator. Diagonal contribution: dH0_μμ/dCN_B = -kCN(s_μ)/AU_TO_EV
  // when the shell of μ is on atom B (else 0). Folded into the loop below
  // so it tracks the off-diagonal CN dependence too.
  Vec dE_dCN = Vec::Zero(n_atoms);
  for (int mu = 0; mu < nbf; ++mu) {
    const int sh = bf_to_shell[mu];
    const int B = shells.atom[sh];
    dE_dCN(B) -= P(mu, mu) * shells.kcn(sh) * inv_au_to_ev;
  }

  // Off-diagonal H0 build mirrors the structure of build_h0(), accumulating
  // into X, dE_dCN, and the per-atom ∂Π/∂R contribution to grad.
  const auto &first_bf = basis.first_bf();
  for (size_t si = 0; si < basis.size(); ++si) {
    const int ai = shells.atom[si];
    const int li = shells.ang_mom(si);
    const int bf_i0 = first_bf[si];
    const int n_i = static_cast<int>(basis[si].size());
    const double zi = slater(si);
    const double poly_i = shells.shell_poly(si);

    for (size_t sj = 0; sj < si; ++sj) {
      const int aj = shells.atom[sj];
      if (ai == aj) continue;
      const int lj = shells.ang_mom(sj);
      const int bf_j0 = first_bf[sj];
      const int n_j = static_cast<int>(basis[sj].size());
      const double zj = slater(sj);
      const double poly_j = shells.shell_poly(sj);

      const double dx = atoms[ai].x - atoms[aj].x;
      const double dy = atoms[ai].y - atoms[aj].y;
      const double dz = atoms[ai].z - atoms[aj].z;
      const double r_ab = std::sqrt(dx * dx + dy * dy + dz * dz);

      // K · enpoly · zfac^wexp factor (no R-dependence beyond zfac, which
      // depends on Slater exponents — atom-type quantity, not R).
      const auto *e_i = params.element(atoms[ai].atomic_number);
      const auto *e_j = params.element(atoms[aj].atomic_number);
      const double den = (e_i->pauling_en - e_j->pauling_en);
      const double den2 = den * den;
      const double enpoly_coef = 0.01 * g.enscale;
      const double enpoly =
          1.0 + enpoly_coef * den2 * (1.0 + g.enscale4 * den2);
      const double zfac = (2.0 * std::sqrt(zi * zj) / (zi + zj));
      const double km_no_h = K(li, lj) * enpoly * std::pow(zfac, w_exp);

      const auto poly = shell_poly_with_grad(poly_i, poly_j, a_rad[ai],
                                             a_rad[aj], r_ab);

      // hav = 0.5 (h_A + h_B)·Π ; X_μν = km_no_h · hav / AU_TO_EV
      const double h_sum = se_ev(si) + se_ev(sj);
      const double hav = 0.5 * h_sum * poly.pi;
      const double x_block = km_no_h * hav * inv_au_to_ev;

      // Fill the X block (H0 / S element-wise) for this shell pair.
      for (int mu = 0; mu < n_i; ++mu) {
        for (int nu = 0; nu < n_j; ++nu) {
          X(bf_i0 + mu, bf_j0 + nu) = x_block;
          X(bf_j0 + nu, bf_i0 + mu) = x_block;
        }
      }

      // For chain-rule contributions we need (Σ_block P·S) for this pair.
      double ps_sum = 0.0;
      for (int mu = 0; mu < n_i; ++mu) {
        for (int nu = 0; nu < n_j; ++nu) {
          // Symmetric P, S → counted twice (μν and νμ).
          ps_sum += 2.0 * P(bf_i0 + mu, bf_j0 + nu) * S(bf_i0 + mu, bf_j0 + nu);
        }
      }

      // (b) ∂Π/∂R contribution. ∂hav/∂R_AB = 0.5·h_sum·dpi_dR; the entire
      // off-diag block contributes ps_sum × (km_no_h/AU_TO_EV) × ∂hav/∂R.
      // ∂R_AB/∂r_A = (r_A − r_B)/R_AB; opposite sign for atom B.
      const double scal_pi =
          ps_sum * km_no_h * inv_au_to_ev * 0.5 * h_sum * poly.dpi_dR / r_ab;
      grad(0, ai) += scal_pi * dx;
      grad(1, ai) += scal_pi * dy;
      grad(2, ai) += scal_pi * dz;
      grad(0, aj) -= scal_pi * dx;
      grad(1, aj) -= scal_pi * dy;
      grad(2, aj) -= scal_pi * dz;

      // (c) ∂(0.5·h_sum)/∂R chain term. H0 depends on h_A = ε - kCN(sA)·CN_A,
      // so dE/dCN_A picks up −kCN(sA) × (the off-diag block weight).
      const double dE_dCN_block =
          ps_sum * km_no_h * inv_au_to_ev * 0.5 * poly.pi;
      dE_dCN(ai) += dE_dCN_block * (-shells.kcn(si));
      dE_dCN(aj) += dE_dCN_block * (-shells.kcn(sj));
    }
  }

  // Chain through CN: grad += Σ_B dE/dCN_B × ∂CN_B/∂R
  for (int B = 0; B < n_atoms; ++B) {
    grad += dE_dCN(B) * dcn[B];
  }

  // (a) Combined S-derivative term. The "via S" assembly bundles three
  // R-dependent contributions through ∂S/∂R:
  //
  //     Z[μν] = P[μν]·X[μν]              ← H0_off-diag-via-S
  //           − W[μν]                    ← Pulay
  //           − ½·P[μν]·(V_{s(μ)}+V_{s(ν)})  ← Tr(P · ∂V_q/∂R) via S
  //
  // V_q is the SCC Coulomb shift potential V_q[μν] = ½·S[μν]·(V_s+V_t).
  // Differentiating with V (= J·q) held fixed pulls a factor of dS/dR out.
  if (V_shell.size() != static_cast<Eigen::Index>(shells.atom.size())) {
    throw std::runtime_error("h0_scc_gradient: V_shell size mismatch");
  }
  Vec V_per_bf(nbf);
  for (int mu = 0; mu < nbf; ++mu) {
    V_per_bf(mu) = V_shell(bf_to_shell[mu]);
  }
  Mat Z = (P.array() * X.array()).matrix() - W;
  // -½ P .* (V_μ + V_ν) elementwise: V_μ + V_ν = V·1ᵀ + 1·Vᵀ (rank-2).
  Z.noalias() -=
      0.5 * (P.array() *
             (V_per_bf * Vec::Ones(nbf).transpose() +
              Vec::Ones(nbf) * V_per_bf.transpose())
                .array())
                .matrix();

  occ::qm::IntegralEngine::Op op_overlap =
      occ::qm::IntegralEngine::Op::overlap;
  MatTriple ovlp_grad = engine.one_electron_operator_grad(op_overlap);

  const auto &atom_to_shell = basis.atom_to_shell();
  // Per-atom assembly of Σ Z_μν · ∂S_μν/∂R. ovlp_grad.x[μ,ν] is
  // <∂_x φ_μ | φ_ν> = ∂S_μν/∂R_(atom of μ)_x (libcint cint1e_ipovlp).
  // The matrix is antisymmetric (IBP), so the full per-atom contribution is
  //   g_A,x = 2 Σ_{μ on A} ovlp.x.row(μ) · Z.row(μ)
  // (the (ν on A, μ not) part equals the (μ on A, ν not) part after using
  // Z symmetry + ovlp antisymmetry).
  for (int A = 0; A < n_atoms; ++A) {
    for (int s : atom_to_shell[A]) {
      const auto &sh = basis[s];
      const int bf0 = first_bf[s];
      const int sz = static_cast<int>(sh.size());
      for (int mu = bf0; mu < bf0 + sz; ++mu) {
        grad(0, A) -= 2.0 * ovlp_grad.x.row(mu).dot(Z.row(mu));
        grad(1, A) -= 2.0 * ovlp_grad.y.row(mu).dot(Z.row(mu));
        grad(2, A) -= 2.0 * ovlp_grad.z.row(mu).dot(Z.row(mu));
      }
    }
  }

  return grad;
}

} // namespace occ::xtb
