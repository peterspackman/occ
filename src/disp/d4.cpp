#include "d4_data.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/core/data_directory.h>
#include <occ/core/eeq.h>
#include <occ/core/log.h>
#include <occ/disp/d4.h>
#include <stdexcept>
#include <unordered_map>


namespace occ::disp {

namespace {

using namespace d4_data;

// ============================================================================
// D4 covalent coordination number (erf-counted, EN-weighted).
// xtb's `xtb_disp_ncoord::ncoord_d4` — Pyykko covalent radii (×4/3, in Bohr),
// Pauling EN damping factor `den = k4·exp(-((|ΔEN|+k5)^2)/k6)`, and erf count
// `0.5·(1 + erf(-kn·(r-r0)/r0))`.
// ============================================================================

namespace cn_consts {
constexpr double k4 = 4.10451;
constexpr double k5 = 19.08857;
constexpr double k6 = 2.0 * 11.28174 * 11.28174;
constexpr double kn = 7.50;
constexpr double bohr_per_angstrom = 1.0 / 0.52917726; // matches xtb conv

// Pyykko & Atsumi, Chem. Eur. J. 15 (2009), 188-197. Z=1..118 in Å,
// metals reduced by 10%. xtb scales by 4/3 to get the radii used in the
// counting function.
constexpr std::array<double, 119> rcov_angstrom = {
    0.0,
    0.32, 0.46,
    1.20, 0.94, 0.77, 0.75, 0.71, 0.63, 0.64, 0.67,
    1.40, 1.25, 1.13, 1.04, 1.10, 1.02, 0.99, 0.96,
    1.76, 1.54,
    1.33, 1.22, 1.21, 1.10, 1.07, 1.04, 1.00, 0.99, 1.01, 1.09,
    1.12, 1.09, 1.15, 1.10, 1.14, 1.17,
    1.89, 1.67,
    1.47, 1.39, 1.32, 1.24, 1.15, 1.13, 1.13, 1.08, 1.15, 1.23,
    1.28, 1.26, 1.26, 1.23, 1.32, 1.31,
    2.09, 1.76,
    1.62, 1.47, 1.58, 1.57, 1.56, 1.55, 1.51,
    1.52, 1.51, 1.50, 1.49, 1.49, 1.48, 1.53,
    1.46, 1.37, 1.31, 1.23, 1.18, 1.16, 1.11, 1.12, 1.13, 1.32,
    1.30, 1.30, 1.36, 1.31, 1.38, 1.42,
    2.01, 1.81,
    1.67, 1.58, 1.52, 1.53, 1.54, 1.55, 1.49,
    1.49, 1.51, 1.51, 1.48, 1.50, 1.56, 1.58,
    1.45, 1.41, 1.34, 1.29, 1.27, 1.21, 1.16, 1.15, 1.09, 1.22,
    1.22, 1.29, 1.46, 1.58, 1.48, 1.41,
};

// Pauling electronegativities (xtb's table). Index 0 unused.
constexpr std::array<double, 119> pauling_en = {
    0.0,
    2.20, 3.00,
    0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 4.50,
    0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 3.50,
    0.82, 1.00,
    1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90, 1.65,
    1.81, 2.01, 2.18, 2.55, 2.96, 3.00,
    0.82, 0.95,
    1.22, 1.33, 1.60, 2.16, 1.90, 2.20, 2.28, 2.20, 1.93, 1.69,
    1.78, 1.96, 2.05, 2.10, 2.66, 2.60,
    0.79, 0.89,
    1.10, 1.12, 1.13, 1.14, 1.15, 1.17, 1.18,
    1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26,
    1.27, 1.30, 1.50, 2.36, 1.90, 2.20, 2.20, 2.28, 2.54, 2.00,
    1.62, 2.33, 2.02, 2.00, 2.20, 2.20,
    0.79, 0.90,
    1.10, 1.30, 1.50, 1.38, 1.36, 1.28, 1.30,
    1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30,
    1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30, 1.30,
    1.30, 1.30, 1.30, 1.30, 1.30, 1.30,
};

// Bohr-radius-scaled covalent counter radius: 4/3 * R_cov(Å) * Å→Bohr.
constexpr double rcov_bohr(int Z) {
  return (4.0 / 3.0) * rcov_angstrom[Z] * bohr_per_angstrom;
}

inline double erf_count(double k, double r, double r0) {
  return 0.5 * (1.0 + std::erf(-k * (r - r0) / r0));
}

inline double d_erf_count(double k, double r, double r0) {
  // d/dr [0.5 (1 + erf(-k (r-r0)/r0))]
  //   = -k / (r0 √π) · exp(-(k(r-r0)/r0)^2)
  constexpr double inv_sqrt_pi = 0.5641895835477563;
  const double t = k * (r - r0) / r0;
  return -k / r0 * inv_sqrt_pi * std::exp(-t * t);
}

} // namespace cn_consts

// Returns CN vector and (when grad != nullptr) per-atom derivative tensor
// dCN[i] / dR_j as a stack of N x (3, N) matrices.
struct CnResult {
  Vec cn;
  std::vector<Mat3N> dcn; // populated only if requested
};

CnResult d4_coordination_numbers(const std::vector<core::Atom> &atoms,
                                 double cn_cutoff, bool with_gradient) {
  using namespace cn_consts;
  const int n = static_cast<int>(atoms.size());
  CnResult out;
  out.cn = Vec::Zero(n);
  if (with_gradient) {
    out.dcn.assign(n, Mat3N::Zero(3, n));
  }
  const double cutoff2 = cn_cutoff * cn_cutoff;
  for (int i = 0; i < n; ++i) {
    const int Zi = atoms[i].atomic_number;
    const double rci = rcov_bohr(Zi);
    const double eni = pauling_en[Zi];
    for (int j = 0; j < i; ++j) {
      const int Zj = atoms[j].atomic_number;
      const double dx = atoms[i].x - atoms[j].x;
      const double dy = atoms[i].y - atoms[j].y;
      const double dz = atoms[i].z - atoms[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff2) continue;
      const double r = std::sqrt(r2);
      const double r0 = rci + rcov_bohr(Zj);
      const double dEN = std::abs(eni - pauling_en[Zj]);
      const double den = k4 * std::exp(-((dEN + k5) * (dEN + k5)) / k6);
      const double tmp = den * erf_count(kn, r, r0);
      out.cn(i) += tmp;
      out.cn(j) += tmp;
      if (with_gradient) {
        const double dtmp_dr = den * d_erf_count(kn, r, r0);
        const double inv_r = 1.0 / r;
        const double gx = dtmp_dr * dx * inv_r;
        const double gy = dtmp_dr * dy * inv_r;
        const double gz = dtmp_dr * dz * inv_r;
        // CN_i depends on this pair → contributes to grad of CN_i wrt R_i, R_j
        out.dcn[i](0, i) += gx; out.dcn[i](1, i) += gy; out.dcn[i](2, i) += gz;
        out.dcn[i](0, j) -= gx; out.dcn[i](1, j) -= gy; out.dcn[i](2, j) -= gz;
        // Same pair contributes symmetrically to CN_j.
        out.dcn[j](0, i) += gx; out.dcn[j](1, i) += gy; out.dcn[j](2, i) += gz;
        out.dcn[j](0, j) -= gx; out.dcn[j](1, j) -= gy; out.dcn[j](2, j) -= gz;
      }
    }
  }
  return out;
}

// ============================================================================
// ζ function for charge-aware projection (Caldeweyher et al., JCP 150, 154122).
//   ζ(a, c, q_ref, q_mod) = exp(a - exp(c · (1 - q_ref / q_mod)))   if q_mod > 0
//                         = exp(a)                                  otherwise
// ============================================================================

inline double zeta(double a, double c, double qref, double qmod) {
  if (qmod <= 0.0) return std::exp(a);
  return std::exp(a * (1.0 - std::exp(c * (1.0 - qref / qmod))));
}

// ============================================================================
// Per-element pre-computed reference α(iω). Computed once per geometry from
// the reference data — depends on element identity but not on geometry, so we
// return for all atoms in the input.
//
// For each (atom i, reference iref) and frequency k we compute
//   ref_alpha[i][iref][k] = max(0, ascale[iref]·(alphaiw[iref][k]
//                                  - hcount[iref]·sscale[is]·secaiw[is][k]
//                                                ·ζ(ga, gam[is]·gc,
//                                                   secq[is]+zeff[is],
//                                                   refh[iref]+zeff[is])))
// where `is = refsys[iref]` is the secondary-atom (H/C/N/O/…) index used for
// fragment subtraction. Index by atom (not Z) so different atoms of the same
// element share the table.
// ============================================================================

using AlphaTable = std::vector<std::array<std::array<double, N_FREQ>, MAX_REF>>;

AlphaTable build_reference_alpha(const std::vector<core::Atom> &atoms,
                                 const D4Scaling &sc, RefqMode mode) {
  const auto &rd = reference_data();
  const auto &secq_active =
      (mode == RefqMode::GFN2) ? rd.secq_gfn2 : rd.secq_dft;
  const int n = static_cast<int>(atoms.size());
  AlphaTable out(n);
  for (int i = 0; i < n; ++i) {
    const int Z = atoms[i].atomic_number;
    const auto &ref = rd.elements[Z];
    const auto &refh = (mode == RefqMode::GFN2) ? ref.refh_gfn2 : ref.refh_dft;
    for (int iref = 0; iref < ref.refn; ++iref) {
      const int is = ref.refsys[iref];
      // xtb uses `iz = zeff(is)` where `is` is the secondary-atom index (1..17).
      // The secondary atoms happen to be H, He, …, Cl (index = atomic number).
      const double iz = (is > 0 && is <= N_SECONDARY) ? rd.zeff[is] : 0.0;
      const double sec_factor = (is > 0 && is <= N_SECONDARY)
                                    ? ref.hcount[iref] * rd.sscale[is]
                                    : 0.0;
      const double zarg =
          (is > 0 && is <= N_SECONDARY)
              ? zeta(sc.ga, rd.gam[is] * sc.gc, secq_active[is] + iz,
                     refh[iref] + iz)
              : 0.0;
      for (int k = 0; k < N_FREQ; ++k) {
        const double sec_aiw =
            (is > 0 && is <= N_SECONDARY) ? rd.secaiw[is][k] : 0.0;
        const double sec_al = sec_factor * sec_aiw * zarg;
        const double v = ref.ascale[iref] * (ref.alphaiw[iref][k] - sec_al);
        out[i][iref][k] = (v > 0.0) ? v : 0.0;
      }
    }
  }
  return out;
}

// ============================================================================
// Reference weights gw(iref, i): Gaussian of CN difference, normalized over
// references, multiplied by ζ(refq, q_atomic).
// ============================================================================

// Compute the per-element `ncount` table — how many Gaussian widths to sum
// for each reference. xtb (and cpp-d4 via `refc`) bin references by their
// integer-rounded refcn; if `k` references share the same bin, each one
// contributes ncount = k(k+1)/2 Gaussians, with widths wf, 2·wf, …, ncount·wf.
// This multi-width form smoothly interpolates the C6 between the discrete
// reference CN values.
std::array<int, MAX_REF> compute_ncount(const ElementRefs &ref) {
  std::array<int, 32> cncount{}; // CN bin counts
  cncount[0] = 1;                // xtb's seed value for the "no neighbour" bin
  for (int iref = 0; iref < ref.refn; ++iref) {
    const int bin = static_cast<int>(std::round(ref.refcn[iref]));
    if (bin >= 0 && bin < (int)cncount.size()) ++cncount[bin];
  }
  std::array<int, MAX_REF> nc{};
  for (int iref = 0; iref < ref.refn; ++iref) {
    const int bin = static_cast<int>(std::round(ref.refcn[iref]));
    const int k = (bin >= 0 && bin < (int)cncount.size()) ? cncount[bin] : 1;
    nc[iref] = k * (k + 1) / 2;
  }
  return nc;
}

Mat compute_reference_weights(const std::vector<core::Atom> &atoms,
                              const D4Scaling &sc, RefqMode mode,
                              const Vec &cn, const Vec &q_atomic) {
  const auto &rd = reference_data();
  const int n = static_cast<int>(atoms.size());
  Mat gw = Mat::Zero(MAX_REF, n);
  for (int i = 0; i < n; ++i) {
    const int Z = atoms[i].atomic_number;
    const auto &ref = rd.elements[Z];
    if (ref.refn == 0) continue;
    const auto &refq = (mode == RefqMode::GFN2) ? ref.refq_gfn2 : ref.refq_dft;
    const auto ncount = compute_ncount(ref);

    // Multi-Gaussian sum per reference (widths twf = igw · wf, igw = 1..ncount).
    double norm = 0.0;
    double maxcn = 0.0;
    for (int iref = 0; iref < ref.refn; ++iref) {
      maxcn = std::max(maxcn, ref.refcovcn[iref]);
      const double dcn = cn(i) - ref.refcovcn[iref];
      const double dcn2 = dcn * dcn;
      for (int igw = 1; igw <= ncount[iref]; ++igw) {
        norm += std::exp(-igw * sc.wf * dcn2);
      }
    }
    const double inv_norm = (norm > 0.0) ? 1.0 / norm : 0.0;
    const double iz = rd.zeff[Z];
    for (int iref = 0; iref < ref.refn; ++iref) {
      const double dcn = cn(i) - ref.refcovcn[iref];
      const double dcn2 = dcn * dcn;
      double expw = 0.0;
      for (int igw = 1; igw <= ncount[iref]; ++igw) {
        expw += std::exp(-igw * sc.wf * dcn2);
      }
      double gwk = expw * inv_norm;
      if (!std::isfinite(gwk)) {
        // Saturated case — assign full weight to highest-CN reference.
        gwk = (ref.refcovcn[iref] == maxcn) ? 1.0 : 0.0;
      }
      const double z = zeta(sc.ga, rd.gam[Z] * sc.gc, refq[iref] + iz,
                             q_atomic(i) + iz);
      gw(iref, i) = gwk * z;
    }
  }
  return gw;
}

// ============================================================================
// C6 matrix from per-atom α(iω) via Casimir-Polder integration.
//   α_i(ω) = Σ_iref gw(iref,i) · ref_alpha[i][iref][ω]
//   C6_ij  = (3/π) · Σ_ω weights[ω] · α_i(ω) · α_j(ω)
// where the (3/π) is folded into `casimir_polder_weights` already (it's in
// the D4 model definition; xtb scales by 3/π via `thopi`).
// ============================================================================

constexpr double thopi = 3.0 / 3.141592653589793;

Mat compute_c6_matrix(const std::vector<core::Atom> &atoms, const Mat &gw,
                       const AlphaTable &ref_alpha) {
  const auto &rd = reference_data();
  const int n = static_cast<int>(atoms.size());
  Mat c6 = Mat::Zero(n, n);
  for (int i = 0; i < n; ++i) {
    const int Zi = atoms[i].atomic_number;
    const int ni = rd.elements[Zi].refn;
    for (int j = 0; j <= i; ++j) {
      const int Zj = atoms[j].atomic_number;
      const int nj = rd.elements[Zj].refn;
      double cij = 0.0;
      for (int ir = 0; ir < ni; ++ir) {
        const double wi = gw(ir, i);
        if (wi == 0.0) continue;
        for (int jr = 0; jr < nj; ++jr) {
          const double wj = gw(jr, j);
          if (wj == 0.0) continue;
          // Casimir-Polder trapezoidal sum over 23 frequencies.
          double cp = 0.0;
          for (int k = 0; k < N_FREQ; ++k) {
            cp += rd.casimir_polder_weights[k] * ref_alpha[i][ir][k] *
                  ref_alpha[j][jr][k];
          }
          cij += wi * wj * thopi * cp;
        }
      }
      c6(i, j) = cij;
      c6(j, i) = cij;
    }
  }
  return c6;
}

// ============================================================================
// Two-body BJ-damped energy.
//   r0_ij = a1 √(3 · sqrtZr4r2_i · sqrtZr4r2_j) + a2     (cutoff radius)
//   E2    = -Σ_{i<j} [ s6 C6_ij / (r^6 + r0^6) + s8 C8_ij / (r^8 + r0^8) ]
//   C8_ij = 3 · sqrtZr4r2_i · sqrtZr4r2_j · C6_ij
// ============================================================================

double dispersion_2body(const std::vector<core::Atom> &atoms,
                         const D4Damping &dp, const Mat &c6,
                         double cutoff2_bohr) {
  const auto &rd = reference_data();
  const int n = static_cast<int>(atoms.size());
  const double cutoff2 = cutoff2_bohr * cutoff2_bohr;
  double e = 0.0;
  for (int i = 0; i < n; ++i) {
    const double qi = rd.sqrt_zr4r2[atoms[i].atomic_number];
    for (int j = 0; j < i; ++j) {
      const double qj = rd.sqrt_zr4r2[atoms[j].atomic_number];
      const double dx = atoms[i].x - atoms[j].x;
      const double dy = atoms[i].y - atoms[j].y;
      const double dz = atoms[i].z - atoms[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff2 || r2 < 1e-12) continue;
      const double r4r2ij = 3.0 * qi * qj;       // = (R0^2 in BJ formula)
      const double r0 = dp.a1 * std::sqrt(r4r2ij) + dp.a2;
      const double r0_2 = r0 * r0;
      const double r0_6 = r0_2 * r0_2 * r0_2;
      const double r0_8 = r0_6 * r0_2;
      const double r6 = r2 * r2 * r2;
      const double r8 = r6 * r2;
      const double cij = c6(i, j);
      const double c8ij = r4r2ij * cij;
      e -= dp.s6 * cij / (r6 + r0_6) + dp.s8 * c8ij / (r8 + r0_8);
    }
  }
  return e;
}

// ============================================================================
// Axilrod-Teller-Muto 3-body. xtb uses zero-damping with exponent `alp`:
//   E3 = s9 · Σ_{i<j<k} c9_ijk · (3 cos θ_a cos θ_b cos θ_c + 1) / (rij rik rjk)^3 · damp
//   c9_ijk = -√(C6_ij · C6_ik · C6_jk)
//   damp = 1 / (1 + 6 · (R0_ijk / r_avg)^alp), with r_avg = (rij rik rjk)^(1/3)
//   R0_ijk = (R0_ij · R0_ik · R0_jk)^(1/3)   where R0 is BJ cutoff above.
// ============================================================================

double dispersion_3body(const std::vector<core::Atom> &atoms,
                         const D4Damping &dp, const Mat &c6,
                         double cutoff2_bohr) {
  if (dp.s9 == 0.0) return 0.0;
  const auto &rd = reference_data();
  const int n = static_cast<int>(atoms.size());
  const double cutoff2 = cutoff2_bohr * cutoff2_bohr;
  // Pre-compute per-atom BJ R0_AB factors (they're symmetric in i, j).
  Mat r0ij(n, n);
  for (int i = 0; i < n; ++i) {
    const double qi = rd.sqrt_zr4r2[atoms[i].atomic_number];
    for (int j = 0; j <= i; ++j) {
      const double qj = rd.sqrt_zr4r2[atoms[j].atomic_number];
      const double r0 = dp.a1 * std::sqrt(3.0 * qi * qj) + dp.a2;
      r0ij(i, j) = r0;
      r0ij(j, i) = r0;
    }
  }
  double e = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      const double dxij = atoms[i].x - atoms[j].x;
      const double dyij = atoms[i].y - atoms[j].y;
      const double dzij = atoms[i].z - atoms[j].z;
      const double r2ij = dxij * dxij + dyij * dyij + dzij * dzij;
      if (r2ij > cutoff2 || r2ij < 1e-12) continue;
      for (int k = 0; k < j; ++k) {
        const double dxik = atoms[i].x - atoms[k].x;
        const double dyik = atoms[i].y - atoms[k].y;
        const double dzik = atoms[i].z - atoms[k].z;
        const double r2ik = dxik * dxik + dyik * dyik + dzik * dzik;
        if (r2ik > cutoff2 || r2ik < 1e-12) continue;
        const double dxjk = atoms[j].x - atoms[k].x;
        const double dyjk = atoms[j].y - atoms[k].y;
        const double dzjk = atoms[j].z - atoms[k].z;
        const double r2jk = dxjk * dxjk + dyjk * dyjk + dzjk * dzjk;
        if (r2jk > cutoff2 || r2jk < 1e-12) continue;
        const double rij = std::sqrt(r2ij);
        const double rik = std::sqrt(r2ik);
        const double rjk = std::sqrt(r2jk);
        // ATM C9 (POSITIVE — convention from Caldeweyher 2019 / xtb / cpp-d4).
        // The ATM term is repulsive for compact triangles, attractive for
        // collinear triples — net effect depends on geometry.
        const double c9 = std::sqrt(std::abs(c6(i, j) * c6(i, k) * c6(j, k)));
        // Angular factor (3 cosA cosB cosC + 1), expressed via the (r²+r²-r²)
        // form so we don't need explicit cosines:
        //   3 cosA cosB cosC = (r²ij+r²jk-r²ik)·(r²ij+r²ik-r²jk)·(r²ik+r²jk-r²ij)
        //                       / (r_ij r_ik r_jk)²    × 3/8
        const double r2ijk_sq = r2ij * r2ik * r2jk;
        const double rijk_p3 =
            (rij * rik * rjk) * r2ijk_sq;        // (r_ij r_ik r_jk)^3
        const double angular =
            (0.375 * (r2ij + r2jk - r2ik) * (r2ij + r2ik - r2jk) *
                 (r2ik + r2jk - r2ij) / r2ijk_sq +
             1.0) /
            rijk_p3;
        // Zero damping with R0_ijk = (R0_ij·R0_ik·R0_jk) (no cbrt — exponent
        // is alp/3 to compensate, matching cpp-d4's convention).
        const double r0_prod = r0ij(i, j) * r0ij(i, k) * r0ij(j, k);
        const double r_prod = rij * rik * rjk;
        const double damp =
            1.0 / (1.0 + 6.0 * std::pow(r0_prod / r_prod, dp.alp / 3.0));
        e += dp.s9 * c9 * angular * damp;
      }
    }
  }
  return e;
}

} // namespace

// ============================================================================
// Public API
// ============================================================================

Dispersion::Dispersion(std::vector<core::Atom> atoms, RefqMode mode)
    : m_atoms(std::move(atoms)), m_refq_mode(mode) {
  m_q = Vec::Zero(static_cast<int>(m_atoms.size()));
  // Touch the reference data to surface any load errors early.
  (void)d4_data::reference_data();
}

void Dispersion::set_charges_eeq(double net_charge) {
  // EEQ wants atomic_numbers (IVec) and positions (3×N in Å).
  const int n = static_cast<int>(m_atoms.size());
  IVec atnums(n);
  Mat3N positions_ang(3, n);
  constexpr double bohr_to_angstrom = 0.5291772108086;
  for (int i = 0; i < n; ++i) {
    atnums(i) = m_atoms[i].atomic_number;
    positions_ang(0, i) = m_atoms[i].x * bohr_to_angstrom;
    positions_ang(1, i) = m_atoms[i].y * bohr_to_angstrom;
    positions_ang(2, i) = m_atoms[i].z * bohr_to_angstrom;
  }
  m_q = occ::core::charges::eeq_partial_charges(atnums, positions_ang,
                                                net_charge);
}

namespace {

// Cached lookup of functional damping parameters from the JSON database.
const std::unordered_map<std::string, D4Damping> &functional_table() {
  using nlohmann::json;
  static const auto table = [] {
    std::unordered_map<std::string, D4Damping> m;
    namespace fs = std::filesystem;
    auto find_path = []() -> std::string {
      const char *base = occ::get_data_directory();
      if (base) {
        fs::path p = fs::path(base) / "dftd4" / "functionals.json";
        if (fs::exists(p)) return p.string();
      }
      if (fs::exists("dftd4/functionals.json")) return "dftd4/functionals.json";
      if (fs::exists("functionals.json")) return "functionals.json";
      throw std::runtime_error(
          "Cannot locate DFT-D4 functional parameter file (looked at "
          "share/dftd4/functionals.json, dftd4/functionals.json, "
          "functionals.json). Set OCC_DATA_PATH or run from a directory "
          "containing dftd4/functionals.json.");
    };
    std::ifstream in(find_path());
    json j;
    in >> j;
    for (const auto &[name, p] : j.at("functionals").items()) {
      D4Damping d{
          p.value("s6", 1.0), p.value("s8", 0.0), p.value("s9", 1.0),
          p.value("a1", 0.0), p.value("a2", 0.0), p.value("alp", 16),
      };
      m.emplace(name, d);
    }
    return m;
  }();
  return table;
}

} // namespace

void Dispersion::set_functional(const std::string &functional) {
  const auto &table = functional_table();
  auto it = table.find(functional);
  if (it == table.end()) {
    throw std::runtime_error("Unknown DFT-D4 functional: '" + functional +
                             "' (check share/dftd4/functionals.json)");
  }
  m_damping = it->second;
}

void Dispersion::update_positions(const std::vector<core::Atom> &atoms) {
  if (atoms.size() != m_atoms.size()) {
    throw std::runtime_error("Dispersion::update_positions: atom count mismatch");
  }
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    if (atoms[i].atomic_number != m_atoms[i].atomic_number) {
      throw std::runtime_error(
          "Dispersion::update_positions: atomic_number changed at index " +
          std::to_string(i));
    }
    m_atoms[i].x = atoms[i].x;
    m_atoms[i].y = atoms[i].y;
    m_atoms[i].z = atoms[i].z;
  }
}

Vec Dispersion::covalent_coordination_numbers() const {
  return d4_coordination_numbers(m_atoms, m_cutoff_cn, /*with_grad=*/false).cn;
}

double Dispersion::energy() const {
  const auto cn = covalent_coordination_numbers();
  const auto ref_alpha = build_reference_alpha(m_atoms, m_scaling, m_refq_mode);
  const Mat gw = compute_reference_weights(m_atoms, m_scaling, m_refq_mode, cn,
                                           m_q);
  const Mat c6 = compute_c6_matrix(m_atoms, gw, ref_alpha);
  const double e2 = dispersion_2body(m_atoms, m_damping, c6, m_cutoff_disp2);
  const double e3 = dispersion_3body(m_atoms, m_damping, c6, m_cutoff_disp3);
  return e2 + e3;
}

std::pair<double, Mat3N> Dispersion::energy_and_gradient() const {
  // Gradient implementation deferred — call energy() and finite-difference for
  // now; analytical gradient is the next phase.
  throw std::runtime_error(
      "Dispersion::energy_and_gradient not yet implemented (analytical gradient "
      "is a follow-up phase). Use energy() for now.");
}

} // namespace occ::disp
