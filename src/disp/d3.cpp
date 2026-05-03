#include "d3_data.h"
#include "d4_data.h" // shared sqrt_zr4r2 table
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <occ/core/data_directory.h>
#include <occ/core/log.h>
#include <occ/disp/d3.h>
#include <stdexcept>
#include <unordered_map>

namespace occ::disp {

namespace {

using namespace d3_data;

// ============================================================================
// D3 covalent coordination number — erf-counted, no EN factor (k1=16).
// xtb's `xtb_disp_ncoord::ncoord_d3` (and the Grimme s-dftd3 implementation).
// Pyykko & Atsumi 2009 covalent radii (×4/3, in Bohr) like the D4 path, but
// without the Pauling-EN damping factor.
// ============================================================================

namespace cn_consts {
constexpr double k1 = 16.0; // erf steepness (D3 default; D4 uses kn=7.5)
constexpr double bohr_per_angstrom = 1.0 / 0.52917726;

// Pyykko covalent radii (Å), values for metals reduced 10%. Same table used
// in d4.cpp; duplicated here to keep modules self-contained. Index by Z.
constexpr std::array<double, 95> rcov_angstrom = {
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
    1.67, 1.58, 1.52, 1.53, 1.54, 1.55,
};

constexpr double rcov_bohr(int Z) {
  return (4.0 / 3.0) * rcov_angstrom[Z] * bohr_per_angstrom;
}

inline double exp_count(double k, double r, double r0) {
  // D3's ncoord uses 1 / (1 + exp(-k(r0/r - 1))) — the "exponential" counting
  // function (NOT erf, despite the D4 name being similar).
  return 1.0 / (1.0 + std::exp(-k * (r0 / r - 1.0)));
}

inline double d_exp_count_dr(double k, double r, double r0) {
  // d/dr [1/(1 + e^{-k(r0/r - 1)})]
  //   = -count · (1 - count) · k·r0 / r²
  const double c = exp_count(k, r, r0);
  return -c * (1.0 - c) * k * r0 / (r * r);
}

} // namespace cn_consts

struct CnResult {
  Vec cn;
  std::vector<Mat3N> dcn; // populated only if requested
};

CnResult d3_coordination_numbers(const std::vector<core::Atom> &atoms,
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
    if (Zi < 1 || Zi >= (int)rcov_angstrom.size())
      throw std::runtime_error("D3: unsupported element Z=" + std::to_string(Zi));
    const double rci = rcov_bohr(Zi);
    for (int j = 0; j < i; ++j) {
      const int Zj = atoms[j].atomic_number;
      const double dx = atoms[i].x - atoms[j].x;
      const double dy = atoms[i].y - atoms[j].y;
      const double dz = atoms[i].z - atoms[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff2) continue;
      const double r = std::sqrt(r2);
      const double r0 = rci + rcov_bohr(Zj);
      const double c = exp_count(k1, r, r0);
      out.cn(i) += c;
      out.cn(j) += c;
      if (with_gradient) {
        const double dc_dr = d_exp_count_dr(k1, r, r0);
        const double inv_r = 1.0 / r;
        const double gx = dc_dr * dx * inv_r;
        const double gy = dc_dr * dy * inv_r;
        const double gz = dc_dr * dz * inv_r;
        out.dcn[i](0, i) += gx; out.dcn[i](1, i) += gy; out.dcn[i](2, i) += gz;
        out.dcn[i](0, j) -= gx; out.dcn[i](1, j) -= gy; out.dcn[i](2, j) -= gz;
        out.dcn[j](0, i) += gx; out.dcn[j](1, i) += gy; out.dcn[j](2, i) += gz;
        out.dcn[j](0, j) -= gx; out.dcn[j](1, j) -= gy; out.dcn[j](2, j) -= gz;
      }
    }
  }
  return out;
}

// Backward-compat helper for the energy-only path.
Vec d3_coordination_numbers(const std::vector<core::Atom> &atoms,
                             double cn_cutoff) {
  return d3_coordination_numbers(atoms, cn_cutoff, /*with_grad=*/false).cn;
}

// ============================================================================
// Get C6_AB(CN_A, CN_B) from the reference C6 table by Gaussian-weighted
// interpolation over the (refcn_i, refcn_j) grid points. Standard D3 form
// (Grimme et al. 2010, eq. 19).
//
//   C6_AB = Σ_{ki, kj} W_ki·W_kj · C6_ref[Zi, Zj, ki, kj]
//   W_ki  = exp(-k3 · (CN_A − refcn[Zi, ki])^2) / Σ_l exp(-k3 · (CN_A − refcn[Zi, l])^2)
//
// The k3=4 width parameter is part of D3's definition.
// ============================================================================

constexpr double k3 = 4.0;

double get_c6_pair(int Zi, int Zj, double cn_i, double cn_j) {
  const auto &rd = reference_data();
  const int ipair = ReferenceData::pair_index(Zi, Zj);
  const auto &block = rd.c6ab[ipair];
  // Block is indexed as [iref of Z=Zi][iref of Z=Zj] when Zi <= Zj. Need to
  // transpose if Zi > Zj.
  const bool transposed = Zi > Zj;
  const int Zlo = std::min(Zi, Zj);
  const int Zhi = std::max(Zi, Zj);
  const int nlo = rd.nref[Zlo];
  const int nhi = rd.nref[Zhi];
  const auto &cn_lo_table = rd.ref_cn[Zlo];
  const auto &cn_hi_table = rd.ref_cn[Zhi];
  const double cn_lo_value = transposed ? cn_j : cn_i;
  const double cn_hi_value = transposed ? cn_i : cn_j;

  // Per-axis Gaussian weights.
  std::array<double, MAX_REF> w_lo{}, w_hi{};
  for (int k = 0; k < nlo; ++k) {
    const double d = cn_lo_value - cn_lo_table[k];
    w_lo[k] = std::exp(-k3 * d * d);
  }
  for (int k = 0; k < nhi; ++k) {
    const double d = cn_hi_value - cn_hi_table[k];
    w_hi[k] = std::exp(-k3 * d * d);
  }
  // c6ab block layout: rows = high-Z ref, cols = low-Z ref (the iref index
  // for the LOWER atomic number is the inner Fortran index → column under our
  // row-major C++ storage). Only weight components attached to a NON-ZERO
  // reference contribute — both numerator AND denominator exclude missing
  // entries (xtb's get_c6 / Grimme's original D3 formulation).
  double num = 0.0, denom = 0.0;
  for (int kh = 0; kh < nhi; ++kh) {
    for (int kl = 0; kl < nlo; ++kl) {
      const double cref = block[kh][kl];
      if (cref == 0.0) continue;
      const double w = w_lo[kl] * w_hi[kh];
      num += w * cref;
      denom += w;
    }
  }
  if (denom > 0.0) return num / denom;
  // All weights collapsed — fall back to the closest non-zero reference.
  double best = std::numeric_limits<double>::infinity();
  double cref_best = 0.0;
  for (int kh = 0; kh < nhi; ++kh) {
    for (int kl = 0; kl < nlo; ++kl) {
      const double cref = block[kh][kl];
      if (cref == 0.0) continue;
      const double dlo = cn_lo_value - cn_lo_table[kl];
      const double dhi = cn_hi_value - cn_hi_table[kh];
      const double dist = dlo * dlo + dhi * dhi;
      if (dist < best) { best = dist; cref_best = cref; }
    }
  }
  return cref_best;
}

// ============================================================================
// C6 with derivative wrt CN_i and CN_j (per pair). Returns:
//   c6        : the interpolated dispersion coefficient
//   dc6_dcn_i : ∂C6_ij / ∂CN_i  (the input atom for the FIRST argument)
//   dc6_dcn_j : ∂C6_ij / ∂CN_j
// ============================================================================

struct C6Pair {
  double c6{0.0};
  double dc6_dcn_i{0.0};
  double dc6_dcn_j{0.0};
};

C6Pair get_c6_pair_with_grad(int Zi, int Zj, double cn_i, double cn_j) {
  const auto &rd = reference_data();
  const int ipair = ReferenceData::pair_index(Zi, Zj);
  const auto &block = rd.c6ab[ipair];
  const bool transposed = Zi > Zj;
  const int Zlo = std::min(Zi, Zj);
  const int Zhi = std::max(Zi, Zj);
  const int nlo = rd.nref[Zlo];
  const int nhi = rd.nref[Zhi];
  const auto &cn_lo_table = rd.ref_cn[Zlo];
  const auto &cn_hi_table = rd.ref_cn[Zhi];
  const double cn_lo_value = transposed ? cn_j : cn_i;
  const double cn_hi_value = transposed ? cn_i : cn_j;

  std::array<double, MAX_REF> w_lo{}, w_hi{}, dw_lo{}, dw_hi{};
  for (int k = 0; k < nlo; ++k) {
    const double d = cn_lo_value - cn_lo_table[k];
    w_lo[k] = std::exp(-k3 * d * d);
    dw_lo[k] = w_lo[k] * (-2.0 * k3 * d);
  }
  for (int k = 0; k < nhi; ++k) {
    const double d = cn_hi_value - cn_hi_table[k];
    w_hi[k] = std::exp(-k3 * d * d);
    dw_hi[k] = w_hi[k] * (-2.0 * k3 * d);
  }
  // Numerator & denominator (skip cref=0 entries — sparse table).
  // Plus dN/dcn_lo, dN/dcn_hi, dD/dcn_lo, dD/dcn_hi.
  double num = 0.0, denom = 0.0;
  double dn_dlo = 0.0, dn_dhi = 0.0, dd_dlo = 0.0, dd_dhi = 0.0;
  for (int kh = 0; kh < nhi; ++kh) {
    for (int kl = 0; kl < nlo; ++kl) {
      const double cref = block[kh][kl];
      if (cref == 0.0) continue;
      const double w = w_lo[kl] * w_hi[kh];
      num += w * cref;
      denom += w;
      const double dw_dlo = dw_lo[kl] * w_hi[kh];
      const double dw_dhi = w_lo[kl] * dw_hi[kh];
      dn_dlo += dw_dlo * cref;
      dn_dhi += dw_dhi * cref;
      dd_dlo += dw_dlo;
      dd_dhi += dw_dhi;
    }
  }
  C6Pair out;
  if (denom > 0.0) {
    const double inv_d = 1.0 / denom;
    out.c6 = num * inv_d;
    // d(num/denom)/dx = (dnum · denom - num · ddenom) / denom²
    //                 = (dnum - C6 · ddenom) / denom
    const double dc_dlo = (dn_dlo - out.c6 * dd_dlo) * inv_d;
    const double dc_dhi = (dn_dhi - out.c6 * dd_dhi) * inv_d;
    if (transposed) {
      out.dc6_dcn_i = dc_dhi;
      out.dc6_dcn_j = dc_dlo;
    } else {
      out.dc6_dcn_i = dc_dlo;
      out.dc6_dcn_j = dc_dhi;
    }
  } else {
    // No usable reference — fall back to closest non-zero entry as a value.
    out.c6 = 0.0;
    double best = std::numeric_limits<double>::infinity();
    for (int kh = 0; kh < nhi; ++kh) {
      for (int kl = 0; kl < nlo; ++kl) {
        const double cref = block[kh][kl];
        if (cref == 0.0) continue;
        const double dlo = cn_lo_value - cn_lo_table[kl];
        const double dhi = cn_hi_value - cn_hi_table[kh];
        const double dist = dlo * dlo + dhi * dhi;
        if (dist < best) { best = dist; out.c6 = cref; }
      }
    }
    // Derivatives at the saturated edge are zero.
    out.dc6_dcn_i = 0.0;
    out.dc6_dcn_j = 0.0;
  }
  return out;
}

// ============================================================================
// 2-body BJ-damped energy + ATM 3-body (same form as D4).
// sqrt_zr4r2 comes from the D4 reference-data JSON (shared between D3 and D4
// — both use Grimme's r4/r2 expectation values).
// ============================================================================

inline double sqrtZr4r2(int Z) {
  return ::occ::disp::d4_data::reference_data().sqrt_zr4r2[Z];
}

struct GradResult {
  double energy{0.0};
  Mat3N position;
  Vec dE_dcn;
};

// 2-body BJ + position/CN gradient. dE_dcn[i] = -Σ_{j≠i} edisp_ij · ∂C6_ij/∂CN_i
GradResult dispersion_2body_with_grad(const std::vector<core::Atom> &atoms,
                                       const D3Damping &dp, const Vec &cn,
                                       double cutoff_bohr) {
  const int n = static_cast<int>(atoms.size());
  const double cutoff2 = cutoff_bohr * cutoff_bohr;
  GradResult g{0.0, Mat3N::Zero(3, n), Vec::Zero(n)};
  for (int i = 0; i < n; ++i) {
    const int Zi = atoms[i].atomic_number;
    const double qi = sqrtZr4r2(Zi);
    for (int j = 0; j < i; ++j) {
      const int Zj = atoms[j].atomic_number;
      const double qj = sqrtZr4r2(Zj);
      const double dx = atoms[i].x - atoms[j].x;
      const double dy = atoms[i].y - atoms[j].y;
      const double dz = atoms[i].z - atoms[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff2 || r2 < 1e-12) continue;
      const double r4r2ij = 3.0 * qi * qj;
      const double r0 = dp.a1 * std::sqrt(r4r2ij) + dp.a2;
      const double r0_2 = r0 * r0;
      const double r0_6 = r0_2 * r0_2 * r0_2;
      const double r0_8 = r0_6 * r0_2;
      const double r6 = r2 * r2 * r2;
      const double r8 = r6 * r2;
      const double t6 = 1.0 / (r6 + r0_6);
      const double t8 = 1.0 / (r8 + r0_8);
      const auto cp = get_c6_pair_with_grad(Zi, Zj, cn(i), cn(j));
      const double cij = cp.c6;
      const double c8ij = r4r2ij * cij;
      g.energy -= dp.s6 * cij * t6 + dp.s8 * c8ij * t8;
      const double edisp = dp.s6 * t6 + dp.s8 * r4r2ij * t8;
      // Position derivative.
      const double r = std::sqrt(r2);
      const double dEdr = dp.s6 * cij * 6.0 * std::pow(r, 5) * t6 * t6 +
                          dp.s8 * c8ij * 8.0 * std::pow(r, 7) * t8 * t8;
      const double inv_r = 1.0 / r;
      const double gx = dEdr * dx * inv_r;
      const double gy = dEdr * dy * inv_r;
      const double gz = dEdr * dz * inv_r;
      g.position(0, i) += gx; g.position(1, i) += gy; g.position(2, i) += gz;
      g.position(0, j) -= gx; g.position(1, j) -= gy; g.position(2, j) -= gz;
      g.dE_dcn(i) -= edisp * cp.dc6_dcn_i;
      g.dE_dcn(j) -= edisp * cp.dc6_dcn_j;
    }
  }
  return g;
}

double dispersion_2body(const std::vector<core::Atom> &atoms,
                         const D3Damping &dp, const Vec &cn,
                         double cutoff_bohr) {
  const int n = static_cast<int>(atoms.size());
  const double cutoff2 = cutoff_bohr * cutoff_bohr;
  double e = 0.0;
  for (int i = 0; i < n; ++i) {
    const int Zi = atoms[i].atomic_number;
    const double qi = sqrtZr4r2(Zi);
    for (int j = 0; j < i; ++j) {
      const int Zj = atoms[j].atomic_number;
      const double qj = sqrtZr4r2(Zj);
      const double dx = atoms[i].x - atoms[j].x;
      const double dy = atoms[i].y - atoms[j].y;
      const double dz = atoms[i].z - atoms[j].z;
      const double r2 = dx * dx + dy * dy + dz * dz;
      if (r2 > cutoff2 || r2 < 1e-12) continue;
      const double r4r2ij = 3.0 * qi * qj;
      const double r0 = dp.a1 * std::sqrt(r4r2ij) + dp.a2;
      const double r0_2 = r0 * r0;
      const double r0_6 = r0_2 * r0_2 * r0_2;
      const double r0_8 = r0_6 * r0_2;
      const double r6 = r2 * r2 * r2;
      const double r8 = r6 * r2;
      const double cij = get_c6_pair(Zi, Zj, cn(i), cn(j));
      const double c8ij = r4r2ij * cij;
      e -= dp.s6 * cij / (r6 + r0_6) + dp.s8 * c8ij / (r8 + r0_8);
    }
  }
  return e;
}

// 3-body ATM gradient. Uses cpp-d4's closed-form d(angular)/d(r_pair)
// expressions, matching the D4 implementation.
GradResult dispersion_3body_with_grad(const std::vector<core::Atom> &atoms,
                                       const D3Damping &dp, const Vec &cn,
                                       double cutoff_bohr) {
  const int n = static_cast<int>(atoms.size());
  GradResult g{0.0, Mat3N::Zero(3, n), Vec::Zero(n)};
  if (dp.s9 == 0.0) return g;
  const double cutoff2 = cutoff_bohr * cutoff_bohr;
  // Pre-compute pairwise C6 + dC6/dCN_i + dC6/dCN_j and R0 (BJ damping).
  Mat c6(n, n), dc6_dcn(n, n), r0ij(n, n);
  for (int i = 0; i < n; ++i) {
    const int Zi = atoms[i].atomic_number;
    const double qi = sqrtZr4r2(Zi);
    for (int j = 0; j < n; ++j) {
      const int Zj = atoms[j].atomic_number;
      const auto cp = get_c6_pair_with_grad(Zi, Zj, cn(i), cn(j));
      c6(i, j) = cp.c6;
      dc6_dcn(i, j) = cp.dc6_dcn_i;
      const double qj = sqrtZr4r2(Zj);
      r0ij(i, j) = dp.a1 * std::sqrt(3.0 * qi * qj) + dp.a2;
    }
  }
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
        const double c6ij = c6(i, j), c6ik = c6(i, k), c6jk = c6(j, k);
        const double c9 = std::sqrt(std::abs(c6ij * c6ik * c6jk));
        const double rijk = rij * rik * rjk;
        const double r2ijk = r2ij * r2ik * r2jk;
        const double r3ijk = rijk * r2ijk;
        const double r5ijk = r2ijk * r3ijk;
        const double r0_prod = r0ij(i, j) * r0ij(i, k) * r0ij(j, k);
        const double tmp = std::pow(r0_prod / rijk, dp.alp / 3.0);
        const double fdmp = 1.0 / (1.0 + 6.0 * tmp);
        const double ang =
            (0.375 * (r2ij + r2jk - r2ik) * (r2ij + r2ik - r2jk) *
                 (r2ik + r2jk - r2ij) / r2ijk +
             1.0) /
            r3ijk;
        const double e_triple = dp.s9 * c9 * ang * fdmp;
        g.energy += e_triple;

        const double dfdmp = -2.0 * dp.alp * tmp * fdmp * fdmp;
        auto dang_factor = [&](double rA2, double rB2, double rC2) {
          return -0.375 *
                 (std::pow(rA2, 3) + std::pow(rA2, 2) * (rB2 + rC2) +
                  rA2 * (3.0 * rB2 * rB2 + 2.0 * rB2 * rC2 + 3.0 * rC2 * rC2) -
                  5.0 * std::pow(rB2 - rC2, 2) * (rB2 + rC2)) /
                 r5ijk;
        };
        const double dang_ij = dang_factor(r2ij, r2jk, r2ik);
        const double dang_ik = dang_factor(r2ik, r2jk, r2ij);
        const double dang_jk = dang_factor(r2jk, r2ik, r2ij);
        auto pair_grad = [&](double dang_p, double r2p, double dx, double dy,
                              double dz) -> std::array<double, 3> {
          const double scal = -dp.s9 * c9 * (-dang_p * fdmp + ang * dfdmp) / r2p;
          return {scal * dx, scal * dy, scal * dz};
        };
        const auto gij = pair_grad(dang_ij, r2ij, -dxij, -dyij, -dzij);
        const auto gik = pair_grad(dang_ik, r2ik, -dxik, -dyik, -dzik);
        const auto gjk = pair_grad(dang_jk, r2jk, -dxjk, -dyjk, -dzjk);
        g.position(0, i) += -gij[0] - gik[0];
        g.position(1, i) += -gij[1] - gik[1];
        g.position(2, i) += -gij[2] - gik[2];
        g.position(0, j) += gij[0] - gjk[0];
        g.position(1, j) += gij[1] - gjk[1];
        g.position(2, j) += gij[2] - gjk[2];
        g.position(0, k) += gik[0] + gjk[0];
        g.position(1, k) += gik[1] + gjk[1];
        g.position(2, k) += gik[2] + gjk[2];
        // CN chain rule (no q chain for D3).
        const double half_e = 0.5 * e_triple;
        if (c6ij != 0.0) {
          g.dE_dcn(i) += half_e * dc6_dcn(i, j) / c6ij;
          g.dE_dcn(j) += half_e * dc6_dcn(j, i) / c6ij;
        }
        if (c6ik != 0.0) {
          g.dE_dcn(i) += half_e * dc6_dcn(i, k) / c6ik;
          g.dE_dcn(k) += half_e * dc6_dcn(k, i) / c6ik;
        }
        if (c6jk != 0.0) {
          g.dE_dcn(j) += half_e * dc6_dcn(j, k) / c6jk;
          g.dE_dcn(k) += half_e * dc6_dcn(k, j) / c6jk;
        }
      }
    }
  }
  return g;
}

double dispersion_3body(const std::vector<core::Atom> &atoms,
                         const D3Damping &dp, const Vec &cn,
                         double cutoff_bohr) {
  if (dp.s9 == 0.0) return 0.0;
  const int n = static_cast<int>(atoms.size());
  const double cutoff2 = cutoff_bohr * cutoff_bohr;
  // Pre-compute pairwise C6 + R0 (small overhead; pays off in the triple loop).
  Mat c6(n, n), r0ij(n, n);
  for (int i = 0; i < n; ++i) {
    const int Zi = atoms[i].atomic_number;
    const double qi = sqrtZr4r2(Zi);
    for (int j = 0; j <= i; ++j) {
      const int Zj = atoms[j].atomic_number;
      const double qj = sqrtZr4r2(Zj);
      const double cij = get_c6_pair(Zi, Zj, cn(i), cn(j));
      c6(i, j) = cij;
      c6(j, i) = cij;
      const double r0 = dp.a1 * std::sqrt(3.0 * qi * qj) + dp.a2;
      r0ij(i, j) = r0;
      r0ij(j, i) = r0;
    }
  }
  double e = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      const double r2ij = std::pow(atoms[i].x - atoms[j].x, 2) +
                          std::pow(atoms[i].y - atoms[j].y, 2) +
                          std::pow(atoms[i].z - atoms[j].z, 2);
      if (r2ij > cutoff2 || r2ij < 1e-12) continue;
      for (int k = 0; k < j; ++k) {
        const double r2ik = std::pow(atoms[i].x - atoms[k].x, 2) +
                            std::pow(atoms[i].y - atoms[k].y, 2) +
                            std::pow(atoms[i].z - atoms[k].z, 2);
        if (r2ik > cutoff2 || r2ik < 1e-12) continue;
        const double r2jk = std::pow(atoms[j].x - atoms[k].x, 2) +
                            std::pow(atoms[j].y - atoms[k].y, 2) +
                            std::pow(atoms[j].z - atoms[k].z, 2);
        if (r2jk > cutoff2 || r2jk < 1e-12) continue;
        const double rij = std::sqrt(r2ij);
        const double rik = std::sqrt(r2ik);
        const double rjk = std::sqrt(r2jk);
        const double c9 = std::sqrt(std::abs(c6(i, j) * c6(i, k) * c6(j, k)));
        const double r2ijk_sq = r2ij * r2ik * r2jk;
        const double rijk_p3 = (rij * rik * rjk) * r2ijk_sq;
        const double angular = (0.375 * (r2ij + r2jk - r2ik) *
                                (r2ij + r2ik - r2jk) *
                                (r2ik + r2jk - r2ij) / r2ijk_sq +
                                1.0) / rijk_p3;
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

// ============================================================================
// Functional damping database for DFT-D3(BJ).
// ============================================================================

const std::unordered_map<std::string, D3Damping> &functional_table() {
  using nlohmann::json;
  static const auto table = [] {
    std::unordered_map<std::string, D3Damping> m;
    namespace fs = std::filesystem;
    auto find_path = []() -> std::string {
      const char *base = occ::get_data_directory();
      if (base) {
        fs::path p = fs::path(base) / "dftd3" / "functionals.json";
        if (fs::exists(p)) return p.string();
      }
      if (fs::exists("dftd3/functionals.json")) return "dftd3/functionals.json";
      if (fs::exists("functionals.json")) return "functionals.json";
      throw std::runtime_error(
          "Cannot locate DFT-D3 functional parameter file (looked at "
          "share/dftd3/functionals.json, dftd3/functionals.json, "
          "functionals.json). Set OCC_DATA_PATH or run from a directory "
          "containing dftd3/functionals.json.");
    };
    std::ifstream in(find_path());
    json j;
    in >> j;
    for (const auto &[name, p] : j.at("functionals").items()) {
      D3Damping d{
          p.value("s6", 1.0), p.value("s8", 0.0), p.value("s9", 1.0),
          p.value("a1", 0.0), p.value("a2", 0.0), p.value("alp", 14),
      };
      m.emplace(name, d);
    }
    return m;
  }();
  return table;
}

} // namespace

// ============================================================================
// Public API
// ============================================================================

DispersionD3::DispersionD3(std::vector<core::Atom> atoms)
    : m_atoms(std::move(atoms)) {
  for (const auto &a : m_atoms) {
    if (a.atomic_number < 1 || a.atomic_number > N_ELEMENTS) {
      throw std::runtime_error(
          "DispersionD3: element Z=" + std::to_string(a.atomic_number) +
          " is outside the supported range (1..94)");
    }
  }
  (void)d3_data::reference_data();
}

void DispersionD3::set_functional(const std::string &functional) {
  const auto &table = functional_table();
  auto it = table.find(functional);
  if (it == table.end()) {
    throw std::runtime_error("Unknown DFT-D3 functional: '" + functional +
                             "' (check share/dftd3/functionals.json)");
  }
  m_damping = it->second;
}

void DispersionD3::update_positions(const std::vector<core::Atom> &atoms) {
  if (atoms.size() != m_atoms.size()) {
    throw std::runtime_error("DispersionD3::update_positions: count mismatch");
  }
  for (std::size_t i = 0; i < atoms.size(); ++i) {
    if (atoms[i].atomic_number != m_atoms[i].atomic_number) {
      throw std::runtime_error(
          "DispersionD3::update_positions: atomic_number changed at index " +
          std::to_string(i));
    }
    m_atoms[i].x = atoms[i].x;
    m_atoms[i].y = atoms[i].y;
    m_atoms[i].z = atoms[i].z;
  }
}

Vec DispersionD3::coordination_numbers() const {
  return d3_coordination_numbers(m_atoms, m_cutoff_cn);
}

double DispersionD3::energy() const {
  const Vec cn = coordination_numbers();
  const double e2 = dispersion_2body(m_atoms, m_damping, cn, m_cutoff_disp2);
  const double e3 = dispersion_3body(m_atoms, m_damping, cn, m_cutoff_disp3);
  return e2 + e3;
}

std::pair<double, Mat3N> DispersionD3::energy_and_gradient() const {
  // Analytical gradient: position part + CN chain rule. D3 has no charge
  // dependence, so the q-chain term in D4 has no analogue here.
  const auto cn_with_grad =
      d3_coordination_numbers(m_atoms, m_cutoff_cn, /*with_grad=*/true);
  const auto g2 = dispersion_2body_with_grad(m_atoms, m_damping,
                                              cn_with_grad.cn, m_cutoff_disp2);
  const auto g3 = dispersion_3body_with_grad(m_atoms, m_damping,
                                              cn_with_grad.cn, m_cutoff_disp3);
  const int n = static_cast<int>(m_atoms.size());
  Mat3N grad = g2.position + g3.position;
  for (int a = 0; a < n; ++a) {
    const double w = g2.dE_dcn(a) + g3.dE_dcn(a);
    if (w == 0.0) continue;
    grad.noalias() += w * cn_with_grad.dcn[a];
  }
  return {g2.energy + g3.energy, grad};
}

} // namespace occ::disp
