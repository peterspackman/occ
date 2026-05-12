#include <cmath>
#include <occ/core/parallel.h>
#include <occ/xtb/ewald_common.h>
#include <occ/xtb/gamma.h>
#include <occ/xtb/gfn2_parameters.h>
#include <occ/xtb/periodic_gamma.h>
#include <stdexcept>

namespace occ::xtb {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrtPi = 1.7724538509055160273;

// Klopman-Ohno γ at distance R for shell-pair Hubbard average g.
double ko_gamma(double R, double g, double alpha_KO) {
  // alpha_KO is GFN2's alphaj exponent (≠ Ewald α).
  const double r_to_a = std::pow(R, alpha_KO);
  const double inv_g_to_a = std::pow(1.0 / g, alpha_KO);
  return std::pow(r_to_a + inv_g_to_a, -1.0 / alpha_KO);
}

// 5th-order polynomial blend: 1 for r < rcut-1, 0 for r > rcut.
double fsmooth(double r, double rcut) {
  constexpr double offset = 1.0;
  if (r < rcut - offset) return 1.0;
  if (r > rcut) return 0.0;
  const double x = (r - (rcut - offset)) / offset;
  // c5*x^5 + c4*x^4 + c3*x^3 + 1, c = [-6, 15, -10, 1]
  return ((-6.0 * x + 15.0) * x - 10.0) * x * x * x + 1.0;
}

} // namespace

EwaldGammaData build_ewald_data(const PeriodicSystem &sys, double tol,
                                 double alpha_user, double residual_cutoff) {
  const double V = sys.volume();
  const double alpha =
      (alpha_user > 0.0) ? alpha_user : auto_ewald_alpha(V);
  const double erfc_cutoff = ewald_real_cutoff(alpha, tol);
  const double recip_cutoff = ewald_recip_cutoff(alpha, tol);
  // The residual γ(R) - 1/R decays as 1/R^3; for a charge-neutral density it
  // decays effectively as 1/R^4 thanks to dipole cancellation. 60 Bohr gives
  // ~1e-7 truncation for typical hardness ranges.
  const double resid_cutoff =
      (residual_cutoff > 0.0) ? residual_cutoff : 60.0;

  EwaldGammaData d;
  d.alpha = alpha;
  d.erfc_cutoff = erfc_cutoff;
  d.residual_cutoff = resid_cutoff;
  d.recip_cutoff = recip_cutoff;
  d.gamma_rcut = 10.0;  // γ_KO blends to 1/r over [9, 10] Bohr
  d.images =
      build_lattice_images(sys.lattice_bohr, std::max(erfc_cutoff, resid_cutoff));
  d.g_vectors = enumerate_g_vectors(sys.reciprocal_bohr(), recip_cutoff);
  d.g_coeffs.reserve(d.g_vectors.size());
  const double four_pi_over_V = 4.0 * kPi / V;
  const double inv4a2 = 1.0 / (4.0 * alpha * alpha);
  for (const auto &G : d.g_vectors) {
    const double g2 = G.squaredNorm();
    d.g_coeffs.push_back(four_pi_over_V * std::exp(-g2 * inv4a2) / g2);
  }
  d.background = -kPi / (V * alpha * alpha);
  d.self_term = -2.0 * alpha / kSqrtPi;
  return d;
}

namespace {

// Coulomb lattice sum: Σ_T 1/|R+T| computed via Ewald.
// For R != 0: includes T=0 in real-space erfc sum.
// For R == 0: excludes T=0, and the self_term cancels the would-be T=0
//             contribution from the reciprocal sum.
double ewald_coulomb_sum(const Vec3 &R, const EwaldGammaData &d,
                          bool same_site) {
  const double alpha = d.alpha;
  // Real-space erfc(α|R+T|)/|R+T|.
  double s_real = 0.0;
  for (const auto &im : d.images) {
    const Vec3 dR = R + im.t_bohr;
    const double r2 = dR.squaredNorm();
    if (r2 < 1e-20) continue;  // skip "self" position; handled by self_term
    const double r = std::sqrt(r2);
    if (r > d.erfc_cutoff) continue;
    s_real += std::erfc(alpha * r) / r;
  }
  // Reciprocal sum: (4π/V) Σ_{G≠0} (1/G²) exp(-G²/4α²) cos(G·R).
  double s_recip = 0.0;
  for (size_t k = 0; k < d.g_vectors.size(); ++k) {
    s_recip += d.g_coeffs[k] * std::cos(d.g_vectors[k].dot(R));
  }
  double s = s_real + s_recip + d.background;
  if (same_site) s += d.self_term;
  return s;
}

} // namespace

Mat periodic_klopman_ohno_gamma(const PeriodicSystem &sys,
                                 const ShellTable &shells,
                                 const Gfn2Parameters &params,
                                 const EwaldGammaData &d) {
  const double alpha_KO = params.globals().alphaj;
  if (alpha_KO <= 0.0) {
    throw std::runtime_error(
        "periodic_klopman_ohno_gamma: alphaj must be positive");
  }
  const int n_sh = static_cast<int>(shells.atom.size());
  const int n_at = sys.num_atoms();
  Mat J = Mat::Zero(n_sh, n_sh);

  // Pre-cache atom positions for speed.
  std::vector<Vec3> R(n_at);
  for (int A = 0; A < n_at; ++A) {
    R[A] = Vec3(sys.atoms[A].x, sys.atoms[A].y, sys.atoms[A].z);
  }

  // Shell pairs (i ≤ j) — independent across pairs. Thread the outer index;
  // each thread writes only to its own (i, j) cells of J. Symmetrise inside
  // the loop body so we don't need a separate pass.
  occ::parallel::parallel_for(size_t{0}, static_cast<size_t>(n_sh),
                                [&](size_t ii) {
    const int i = static_cast<int>(ii);
    const int Ai = shells.atom[i];
    for (int j = i; j < n_sh; ++j) {
      const int Aj = shells.atom[j];
      const double g_pair =
          0.5 * (shells.hardness(i) + shells.hardness(j));
      const Vec3 Rij = R[Ai] - R[Aj];
      const bool same_site = (Ai == Aj);

      // Residual lattice sum: Σ_T fcut(|R+T|) · [γ(R+T) - 1/|R+T|]
      const double rcut = d.gamma_rcut;
      double s_resid = 0.0;
      for (const auto &im : d.images) {
        const Vec3 dR = Rij + im.t_bohr;
        const double r2 = dR.squaredNorm();
        if (r2 < 1e-20) {
          s_resid += g_pair;
          continue;
        }
        const double r = std::sqrt(r2);
        if (r > rcut) continue;
        const double fcut = fsmooth(r, rcut);
        const double g_val = ko_gamma(r, g_pair, alpha_KO);
        s_resid += fcut * (g_val - 1.0 / r);
      }

      const double s_coul = ewald_coulomb_sum(Rij, d, same_site);

      const double total = s_resid + s_coul;
      J(i, j) = total;
      if (i != j) J(j, i) = total;
    }
  });
  return J;
}

Mat periodic_klopman_ohno_gamma_direct(const PeriodicSystem &sys,
                                        const ShellTable &shells,
                                        const Gfn2Parameters &params,
                                        double cutoff_bohr) {
  const double alpha_KO = params.globals().alphaj;
  const auto images = build_lattice_images(sys.lattice_bohr, cutoff_bohr);
  const int n_sh = static_cast<int>(shells.atom.size());
  const int n_at = sys.num_atoms();
  Mat J = Mat::Zero(n_sh, n_sh);

  std::vector<Vec3> R(n_at);
  for (int A = 0; A < n_at; ++A) {
    R[A] = Vec3(sys.atoms[A].x, sys.atoms[A].y, sys.atoms[A].z);
  }

  for (int i = 0; i < n_sh; ++i) {
    const int Ai = shells.atom[i];
    for (int j = i; j < n_sh; ++j) {
      const int Aj = shells.atom[j];
      const double g_pair =
          0.5 * (shells.hardness(i) + shells.hardness(j));
      const Vec3 Rij = R[Ai] - R[Aj];
      double s = 0.0;
      for (const auto &im : images) {
        const Vec3 dR = Rij + im.t_bohr;
        const double r2 = dR.squaredNorm();
        if (r2 < 1e-20) {
          // On-site limit.
          s += g_pair;
          continue;
        }
        const double r = std::sqrt(r2);
        if (r > cutoff_bohr) continue;
        s += ko_gamma(r, g_pair, alpha_KO);
      }
      J(i, j) = s;
      if (i != j) J(j, i) = s;
    }
  }
  return J;
}

} // namespace occ::xtb
