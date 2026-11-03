#pragma once
// Pieces shared by the four-centre derivative kernels contracted inside the
// integral loop over unique quartets (p >= q, r >= s, pq >= rs): the two-
// electron gradient (jk_gradient_contracted.h), skeleton Hessian and skeleton
// Fock derivatives (integral_engine_hess.cpp).
//
// Energy convention, in terms of spin densities P^s (occ stores the
// restricted D as the per-spin density, so P^a = P^b = D; unrestricted blocks
// are stored halved, so P^s = 2 D^s):
//
//   E2 = jf/2 sum P^t_mn P^t_ls (mn|ls) - kf/2 sum_s sum P^s_ml P^s_ns (mn|ls)
//
// with P^t = P^a + P^b. jf scales Coulomb, kf the exact exchange.
#include <cmath>
#include <occ/gto/shell.h>
#include <occ/qm/mo.h>
#include <occ/qm/shellblock_norm.h>

namespace occ::qm::detail {

/// Spin densities, their total, and shell-block maxima for screening.
struct TwoParticleDensity {
  Mat Pa, Pb, Pt;
  Mat Nt; ///< shell-block maxima of Pt
  Mat Ns; ///< of the larger spin density (the exchange bound is per spin)
  bool unrestricted{false};
};

template <gto::Shell::Kind kind>
TwoParticleDensity two_particle_density(const gto::AOBasis &basis,
                                        const MolecularOrbitals &mo) {
  constexpr auto R = SpinorbitalKind::Restricted;
  TwoParticleDensity t;
  t.unrestricted = mo.kind == SpinorbitalKind::Unrestricted;
  if (t.unrestricted) {
    t.Pa = 2.0 * block::a(mo.D);
    t.Pb = 2.0 * block::b(mo.D);
    t.Pt = t.Pa + t.Pb;
  } else {
    t.Pa = mo.D;
    t.Pt = 2.0 * mo.D;
  }
  t.Nt = shellblock_norm<R, kind>(basis, t.Pt);
  t.Ns = shellblock_norm<R, kind>(basis, t.Pa);
  if (t.unrestricted)
    t.Ns = t.Ns.cwiseMax(shellblock_norm<R, kind>(basis, t.Pb));
  return t;
}

/// Schwarz screen of quartet (pq|rs) for an energy-like contraction with
/// the two-particle density (no screening without Schwarz).
inline bool screened_out(const TwoParticleDensity &t, const Mat &Schwarz,
                         int p, int q, int r, int s, double jf, double kf,
                         double precision) {
  if (Schwarz.size() == 0)
    return false;
  double bound = 0.0;
  if (jf != 0.0)
    bound = std::abs(jf) * t.Nt(p, q) * t.Nt(r, s);
  if (kf != 0.0)
    bound = std::max(bound, std::abs(kf) * std::max(t.Ns(p, r) * t.Ns(q, s),
                                                    t.Ns(p, s) * t.Ns(q, r)));
  return bound * Schwarz(p, q) * Schwarz(r, s) < precision;
}

/// Gamma for a unique quartet in canonical (p q r s) layout, f_p fastest,
/// degeneracy deg included: E2 = sum_{unique} Gamma (pq|rs) with
/// Gamma = deg [jf/2 Pt Pt - kf/4 sum_s (Ps_pr Ps_qs + Ps_ps Ps_qr)].
template <bool unrestricted>
inline void fill_gamma(double *G, const TwoParticleDensity &t, int bp, int bq,
                       int br, int bs, const int nn[4], double deg, double jf,
                       double kf) {
  const double cj = 0.5 * jf * deg, ck = 0.25 * kf * deg;
  const bool do_k = kf != 0.0;
  for (int d = 0, n = 0; d < nn[3]; d++)
    for (int c = 0; c < nn[2]; c++)
      for (int b = 0; b < nn[1]; b++)
        for (int a = 0; a < nn[0]; a++, n++) {
          const int mu = bp + a, nu = bq + b, la = br + c, si = bs + d;
          double g = cj * t.Pt(mu, nu) * t.Pt(la, si);
          if (do_k) {
            double x =
                t.Pa(mu, la) * t.Pa(nu, si) + t.Pa(mu, si) * t.Pa(nu, la);
            if constexpr (unrestricted)
              x += t.Pb(mu, la) * t.Pb(nu, si) + t.Pb(mu, si) * t.Pb(nu, la);
            else
              x *= 2.0;
            g -= ck * x;
          }
          G[n] = g;
        }
}

/// The atoms of a quartet's four centres, and which to differentiate: the
/// atom holding the most of the four shells is omitted (its derivatives
/// follow by translational invariance), `needed` lists the other centres.
struct QuartetCentres {
  int atoms[4];
  int omit;
  int needed[4];
  int nneeded{0};
  bool trivial{false}; ///< all four on one atom: contributes nothing
};

inline QuartetCentres quartet_centres(int a0, int a1, int a2, int a3) {
  QuartetCentres qc{{a0, a1, a2, a3}, a0, {0, 0, 0, 0}, 0, false};
  if (a0 == a1 && a0 == a2 && a0 == a3) {
    qc.trivial = true;
    return qc;
  }
  int best = 0;
  for (int i = 0; i < 4; i++) {
    int count = 0;
    for (int j = 0; j < 4; j++)
      count += qc.atoms[j] == qc.atoms[i];
    if (count > best) {
      best = count;
      qc.omit = qc.atoms[i];
    }
  }
  for (int c = 0; c < 4; c++)
    if (qc.atoms[c] != qc.omit)
      qc.needed[qc.nneeded++] = c;
  return qc;
}

/// The shell order that puts centre c (0..3 of the canonical quartet) first,
/// keeping its bra/ket partner: libcint's ip1 differentiates the first shell
/// passed. perm[k] = canonical index of the k-th passed shell.
inline void centre_first(int c, int perm[4]) {
  static const int table[4][4] = {
      {0, 1, 2, 3}, {1, 0, 2, 3}, {2, 3, 0, 1}, {3, 2, 0, 1}};
  for (int k = 0; k < 4; k++)
    perm[k] = table[c][k];
}

/// out[c] = sum G * buf[c] for an ncomp-component libcint buffer laid out in
/// the permuted shell order `perm`, G in canonical layout.
template <int ncomp>
inline void contract4(const double *G, const double *buf, const int nn[4],
                      const int perm[4], double out[ncomp]) {
  const size_t cstride[4] = {1, static_cast<size_t>(nn[0]),
                             static_cast<size_t>(nn[0]) * nn[1],
                             static_cast<size_t>(nn[0]) * nn[1] * nn[2]};
  const size_t s0 = cstride[perm[0]], s1 = cstride[perm[1]],
               s2 = cstride[perm[2]], s3 = cstride[perm[3]];
  const int n0 = nn[perm[0]], n1 = nn[perm[1]], n2 = nn[perm[2]],
            n3 = nn[perm[3]];
  const size_t nelem = static_cast<size_t>(n0) * n1 * n2 * n3;
  for (int c = 0; c < ncomp; c++)
    out[c] = 0.0;
  size_t n = 0;
  for (int i3 = 0; i3 < n3; i3++)
    for (int i2 = 0; i2 < n2; i2++)
      for (int i1 = 0; i1 < n1; i1++) {
        const size_t base = i3 * s3 + i2 * s2 + i1 * s1;
        for (int i0 = 0; i0 < n0; i0++, n++) {
          const double g = G[base + i0 * s0];
          for (int c = 0; c < ncomp; c++)
            out[c] += g * buf[n + c * nelem];
        }
      }
}

} // namespace occ::qm::detail
