#include <cmath>
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

// Enumerate G-vectors with 0 < |G| <= recip_cutoff.
std::vector<Vec3> enumerate_g_vectors(const Mat3 &reciprocal_bohr,
                                       double recip_cutoff) {
  const Vec3 b1 = reciprocal_bohr.col(0);
  const Vec3 b2 = reciprocal_bohr.col(1);
  const Vec3 b3 = reciprocal_bohr.col(2);
  // Bound on integer indices: |n_i b_i| <= recip_cutoff implies
  // |n_i| <= recip_cutoff / d⊥(b_i) where d⊥ is reciprocal perpendicular.
  // We use the simpler bound |n_i| <= ceil(recip_cutoff / |b_i|) since
  // taking the maximum over directions is sufficient.
  auto bound = [&](const Vec3 &b) {
    return static_cast<int>(std::ceil(recip_cutoff / b.norm())) + 1;
  };
  const int n1 = bound(b1);
  const int n2 = bound(b2);
  const int n3 = bound(b3);
  std::vector<Vec3> out;
  out.reserve(static_cast<size_t>((2 * n1 + 1) * (2 * n2 + 1) * (2 * n3 + 1)));
  const double cutoff2 = recip_cutoff * recip_cutoff;
  for (int i = -n1; i <= n1; ++i) {
    for (int j = -n2; j <= n2; ++j) {
      for (int k = -n3; k <= n3; ++k) {
        Vec3 G = i * b1 + j * b2 + k * b3;
        const double g2 = G.squaredNorm();
        if (g2 < 1e-20 || g2 > cutoff2) continue;
        out.push_back(G);
      }
    }
  }
  return out;
}

} // namespace

EwaldGammaData build_ewald_data(const PeriodicSystem &sys, double tol,
                                 double alpha_user, double residual_cutoff) {
  const double V = sys.volume();
  const double alpha = (alpha_user > 0.0) ? alpha_user
                                          : kSqrtPi / std::cbrt(V);
  // erfc(x) ≤ exp(-x²)/(x√π); set x = sqrt(-ln(tol)) and round up.
  const double x = std::sqrt(-std::log(tol));
  const double erfc_cutoff = x / alpha + 1.0;  // +1 Bohr safety margin
  const double recip_cutoff = 2.0 * alpha * x;
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

  // Loop over shell pairs (i ≤ j, then symmetrize).
  for (int i = 0; i < n_sh; ++i) {
    const int Ai = shells.atom[i];
    for (int j = i; j < n_sh; ++j) {
      const int Aj = shells.atom[j];
      const double g_pair =
          0.5 * (shells.hardness(i) + shells.hardness(j));
      const Vec3 Rij = R[Ai] - R[Aj];
      const bool same_site = (Ai == Aj);

      // Residual lattice sum: Σ_T [γ(R+T) - 1/|R+T|]
      // For same_site, T=0 is the on-site limit γ(0) = g_pair (no 1/R).
      double s_resid = 0.0;
      for (const auto &im : d.images) {
        const Vec3 dR = Rij + im.t_bohr;
        const double r2 = dR.squaredNorm();
        if (r2 < 1e-20) {
          // T = -Rij (or T=0 for same-site). The on-site γ limit is g_pair.
          // No 1/R subtraction here (no Coulomb partner at R=0 to subtract).
          s_resid += g_pair;
          continue;
        }
        const double r = std::sqrt(r2);
        if (r > d.residual_cutoff) continue;
        const double g_val = ko_gamma(r, g_pair, alpha_KO);
        s_resid += g_val - 1.0 / r;
      }

      // Coulomb lattice sum via Ewald.
      const double s_coul = ewald_coulomb_sum(Rij, d, same_site);

      const double total = s_resid + s_coul;
      J(i, j) = total;
      if (i != j) J(j, i) = total;
    }
  }
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
