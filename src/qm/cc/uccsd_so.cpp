#include <algorithm>
#include <chrono>
#include <occ/core/diis.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/qm/cc/thc.h> // mo_eri_general
#include <occ/qm/cc/uccsd_so.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/opmatrix.h> // block::a / block::b
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

// Spin-orbital CCSD(T) -- a direct port of the thc_cct reference ccsd.py
// (Stanton-Gauss-Watts-Bartlett 1991) and triples.py (Crawford project-6).
// Antisymmetrised physicist integrals <pq||rs> are built from the (UHF or RHF)
// MOs, spin orbitals sorted by energy. Canonical reference -> f_ov = 0.

namespace occ::qm::cc {

namespace {

using occ::Mat;
using occ::Vec;
using T2 = Eigen::Tensor<double, 2>;
using T4 = Eigen::Tensor<double, 4>;
using T6 = Eigen::Tensor<double, 6>;
using Sh4 = Eigen::array<int, 4>;
using Sh6 = Eigen::array<int, 6>;
using Idx4 = Eigen::array<Eigen::Index, 4>;
template <int N> using IA = Eigen::array<Eigen::IndexPair<int>, N>;
inline Eigen::IndexPair<int> ip(int a, int b) { return {a, b}; }

T4 t1t1(const T2 &t1) { // tt[i,j,a,b] = t1(i,a) t1(j,b)
  const T4 o = t1.contract(t1, IA<0>{});
  return o.shuffle(Sh4{0, 2, 1, 3});
}
T4 make_tau(const T2 &t1, const T4 &t2, double fac) {
  const T4 tt = t1t1(t1);
  return t2 + (tt - tt.shuffle(Sh4{0, 1, 3, 2})) * fac;
}
T4 P_ij(const T4 &x) { return x - x.shuffle(Sh4{1, 0, 2, 3}); }
T4 P_ab(const T4 &x) { return x - x.shuffle(Sh4{0, 1, 3, 2}); }
T6 P_vir(const T6 &x) {
  return x - x.shuffle(Sh6{0, 1, 2, 4, 3, 5}) - x.shuffle(Sh6{0, 1, 2, 5, 4, 3});
}
T6 P_occ(const T6 &x) {
  return x - x.shuffle(Sh6{1, 0, 2, 3, 4, 5}) - x.shuffle(Sh6{2, 1, 0, 3, 4, 5});
}

struct SOEris {
  int nocc{0}, nvir{0};
  Vec mo_energy; // active spin-orbital energies
  T4 oooo, ooov, oovo, oovv, ovov, ovvo, ovoo, ovvv, vvvv;
  // Per active spin-orbital (length nocc+nvir): spin (0=alpha,1=beta) and the
  // index into the spatial spin-block amplitude arrays. Used to map spin-adapted
  // amplitudes into this (energy-sorted) spin-orbital basis for the (T).
  std::vector<int> spin, amp;
};

// Build antisymmetrised spin-orbital integral blocks from the MOs.
SOEris build_so_eris(const AOBasis &basis, const MolecularOrbitals &mo,
                     int n_frozen) {
  occ::timing::start(occ::timing::category::cc_ao2mo);
  const int nbf = static_cast<int>(mo.n_ao);
  Mat Ca, Cb;
  Vec ea, eb;
  int na, nb;
  if (mo.kind == SpinorbitalKind::Unrestricted) {
    Ca = occ::qm::block::a(mo.C);
    Cb = occ::qm::block::b(mo.C);
    ea = mo.energies.head(nbf);
    eb = mo.energies.segment(nbf, nbf);
    na = static_cast<int>(mo.n_alpha);
    nb = static_cast<int>(mo.n_beta);
  } else {
    Ca = mo.C;
    Cb = mo.C;
    ea = mo.energies.head(nbf);
    eb = ea;
    na = static_cast<int>(mo.n_alpha);
    nb = na;
  }
  const int nmo = static_cast<int>(Ca.cols()); // spatial orbitals per spin

  // Spin-orbital list sorted by energy; the lowest (na+nb) are occupied.
  struct SO {
    int spat, spin;
    double e;
  };
  std::vector<SO> sos;
  sos.reserve(2 * nmo);
  for (int p = 0; p < nmo; ++p)
    sos.push_back({p, 0, ea(p)});
  for (int p = 0; p < nmo; ++p)
    sos.push_back({p, 1, eb(p)});
  std::sort(sos.begin(), sos.end(),
            [](const SO &x, const SO &y) { return x.e < y.e; });

  const int nso = 2 * nmo;
  const int nocc = na + nb;
  const int nfso = 2 * n_frozen;             // frozen spin orbitals
  const int N = nso - nfso;                  // active spin orbitals
  const int no = nocc - nfso, nv = nso - nocc;

  // Spatial chemist integrals for the three spin pairings.
  IntegralEngine engine(basis);
  const T4 Gaa = mo_eri_general(engine, Ca, Ca, Ca, Ca);
  const T4 Gbb = mo_eri_general(engine, Cb, Cb, Cb, Cb);
  const T4 Gab = mo_eri_general(engine, Ca, Ca, Cb, Cb);
  auto chem = [&](int x, int y, int a, int b, int c, int d) -> double {
    if (x == 0 && y == 0)
      return Gaa(a, b, c, d);
    if (x == 1 && y == 1)
      return Gbb(a, b, c, d);
    if (x == 0 && y == 1)
      return Gab(a, b, c, d);
    return Gab(c, d, a, b); // (p_β r_β | q_α s_α) = (q_α s_α | p_β r_β)
  };

  // Antisymmetrised <PQ||RS> = (PR|QS) - (PS|QR) over active spin orbitals.
  T4 gmo(N, N, N, N);
  for (int P = 0; P < N; ++P)
    for (int Q = 0; Q < N; ++Q)
      for (int R = 0; R < N; ++R)
        for (int S = 0; S < N; ++S) {
          const SO &sP = sos[nfso + P], &sQ = sos[nfso + Q];
          const SO &sR = sos[nfso + R], &sS = sos[nfso + S];
          double t1 = (sP.spin == sR.spin && sQ.spin == sS.spin)
                          ? chem(sP.spin, sQ.spin, sP.spat, sR.spat, sQ.spat,
                                 sS.spat)
                          : 0.0;
          double t2 = (sP.spin == sS.spin && sQ.spin == sR.spin)
                          ? chem(sP.spin, sQ.spin, sP.spat, sS.spat, sQ.spat,
                                 sR.spat)
                          : 0.0;
          gmo(P, Q, R, S) = t1 - t2;
        }

  SOEris e;
  e.nocc = no;
  e.nvir = nv;
  e.mo_energy = Vec(N);
  e.spin.resize(N);
  e.amp.resize(N);
  for (int i = 0; i < N; ++i) {
    const SO &so = sos[nfso + i];
    e.mo_energy(i) = so.e;
    e.spin[i] = so.spin;
    // map the full per-spin orbital index to the active spin-block amp index:
    // occupied -> spat - n_frozen, virtual -> spat - n_electrons(spin).
    const int nocc_spin = (so.spin == 0) ? na : nb;
    e.amp[i] = (so.spat < nocc_spin) ? so.spat - n_frozen : so.spat - nocc_spin;
  }

  auto sl = [&](int p0, int pn, int q0, int qn, int r0, int rn, int s0,
                int sn) -> T4 {
    return gmo.slice(Idx4{p0, q0, r0, s0}, Idx4{pn, qn, rn, sn});
  };
  const int v0 = no;
  e.oooo = sl(0, no, 0, no, 0, no, 0, no);
  e.ooov = sl(0, no, 0, no, 0, no, v0, nv);
  e.oovo = sl(0, no, 0, no, v0, nv, 0, no);
  e.oovv = sl(0, no, 0, no, v0, nv, v0, nv);
  e.ovov = sl(0, no, v0, nv, 0, no, v0, nv);
  e.ovvo = sl(0, no, v0, nv, v0, nv, 0, no);
  e.ovoo = sl(0, no, v0, nv, 0, no, 0, no);
  e.ovvv = sl(0, no, v0, nv, v0, nv, v0, nv);
  e.vvvv = sl(v0, nv, v0, nv, v0, nv, v0, nv);
  occ::timing::stop(occ::timing::category::cc_ao2mo);
  return e;
}

double so_energy(const T2 &t1, const T4 &t2, const SOEris &e) {
  const Eigen::Tensor<double, 0> a = (e.oovv * t2).sum();
  const Eigen::Tensor<double, 0> b = (e.oovv * t1t1(t1)).sum();
  return 0.25 * a(0) + 0.5 * b(0);
}

std::pair<T2, T4> so_update_amps(const T2 &t1, const T4 &t2, const SOEris &e) {
  const int o = e.nocc, v = e.nvir;
  const T4 &oooo = e.oooo, &ooov = e.ooov, &oovo = e.oovo, &oovv = e.oovv;
  const T4 &ovov = e.ovov, &ovvo = e.ovvo, &ovoo = e.ovoo, &ovvv = e.ovvv;

  const T4 tau = make_tau(t1, t2, 1.0);
  const T4 taut = make_tau(t1, t2, 0.5);

  // --- F intermediates (f_ov = 0) ---------------------------------------
  T2 Fvv = ovvv.contract(t1, IA<2>{ip(0, 0), ip(2, 1)});          // mf,mafe->ae
  Fvv -= 0.5 * taut.contract(oovv, IA<3>{ip(0, 0), ip(1, 1), ip(3, 3)});
  T2 Foo = ooov.contract(t1, IA<2>{ip(1, 0), ip(3, 1)});          // ne,mnie->mi
  Foo += 0.5 * oovv.contract(taut, IA<3>{ip(1, 1), ip(2, 2), ip(3, 3)});
  T2 Fov = oovv.contract(t1, IA<2>{ip(1, 0), ip(3, 1)});          // nf,mnef->me

  // --- W intermediates --------------------------------------------------
  T4 Woooo = oooo;
  {
    const T4 tmp = ooov.contract(t1, IA<1>{ip(3, 1)}); // je,mnie->mnij
    Woooo += tmp - tmp.shuffle(Sh4{0, 1, 3, 2});
    Woooo += 0.25 * oovv.contract(tau, IA<2>{ip(2, 2), ip(3, 3)});
  }
  T4 Wovvo = ovvo;
  Wovvo += ovvv.contract(t1, IA<1>{ip(3, 1)});                       // jf,mbef->mbej
  Wovvo -= oovo.contract(t1, IA<1>{ip(1, 0)}).shuffle(Sh4{0, 3, 1, 2}); // nb,mnej
  {
    const T4 tt = t1.contract(t1, IA<0>{}).shuffle(Sh4{0, 2, 1, 3}); // (j,n,f,b)
    const T4 Z = 0.5 * t2 + tt;
    Wovvo -= Z.contract(oovv, IA<2>{ip(1, 1), ip(2, 3)}).shuffle(Sh4{2, 1, 3, 0});
  }

  // --- T1 residual ------------------------------------------------------
  T2 r1 = t1.contract(Fvv, IA<1>{ip(1, 1)});            // ie,ae
  r1 -= Foo.contract(t1, IA<1>{ip(0, 0)});              // ma,mi
  r1 += t2.contract(Fov, IA<2>{ip(1, 0), ip(3, 1)});   // imae,me
  r1 -= ovov.contract(t1, IA<2>{ip(0, 0), ip(3, 1)}).shuffle(Eigen::array<int, 2>{1, 0}); // nf,naif
  r1 -= 0.5 * t2.contract(ovvv, IA<3>{ip(1, 0), ip(2, 2), ip(3, 3)}); // imef,maef
  r1 -= 0.5 * t2.contract(oovo, IA<3>{ip(0, 1), ip(1, 0), ip(3, 2)})
                  .shuffle(Eigen::array<int, 2>{1, 0}); // mnae,nmei

  // --- T2 residual ------------------------------------------------------
  T4 r2 = oovv;
  {
    const T2 Fvv_eff = Fvv - 0.5 * t1.contract(Fov, IA<1>{ip(0, 0)});
    r2 += P_ab(t2.contract(Fvv_eff, IA<1>{ip(3, 1)})); // ijae,be
  }
  {
    const T2 Foo_eff = Foo + 0.5 * Fov.contract(t1, IA<1>{ip(1, 1)});
    const T4 tmp = t2.contract(Foo_eff, IA<1>{ip(1, 0)}).shuffle(Sh4{0, 3, 1, 2});
    r2 -= P_ij(tmp); // imab,mj
  }
  r2 += 0.5 * tau.contract(Woooo, IA<2>{ip(0, 0), ip(1, 1)})
                  .shuffle(Sh4{2, 3, 0, 1}); // mnab,mnij
  // vvvv ladder: 0.5 sum_ef tau_ij^ef <ab||ef>
  r2 += 0.5 * tau.contract(e.vvvv, IA<2>{ip(2, 2), ip(3, 3)});
  {
    const T4 Y = (-tau.contract(ovvv, IA<2>{ip(2, 2), ip(3, 3)}))
                     .shuffle(Sh4{0, 1, 3, 2}); // ijef,maef -> ijam
    r2 += -0.5 * P_ab(Y.contract(t1, IA<1>{ip(3, 0)}));
  }
  {
    const T4 Zmn = tau.contract(oovv, IA<2>{ip(2, 2), ip(3, 3)});
    r2 += 0.125 * Zmn.contract(tau, IA<2>{ip(2, 0), ip(3, 1)});
  }
  {
    T4 ring = t2.contract(Wovvo, IA<2>{ip(1, 0), ip(3, 2)})
                  .shuffle(Sh4{0, 3, 1, 2}); // imae,mbej
    ring -= t1.contract(ovvo, IA<1>{ip(1, 2)})
                .contract(t1, IA<1>{ip(1, 0)})
                .shuffle(Sh4{0, 2, 3, 1}); // ie,ma,mbej
    r2 += P_ij(P_ab(ring));
  }
  r2 += P_ij(-ovvv.contract(t1, IA<1>{ip(1, 1)}).shuffle(Sh4{3, 0, 1, 2})); // ie,jeab
  r2 -= P_ab(ovoo.contract(t1, IA<1>{ip(0, 0)}).shuffle(Sh4{1, 2, 3, 0}));  // ma,mbij

  // --- denominators -----------------------------------------------------
  const Vec &mo = e.mo_energy;
  T2 t1n(o, v);
  for (int i = 0; i < o; ++i)
    for (int a = 0; a < v; ++a)
      t1n(i, a) = r1(i, a) / (mo(i) - mo(o + a));
  T4 t2n(o, o, v, v);
  for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
      for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b)
          t2n(i, j, a, b) =
              r2(i, j, a, b) / (mo(i) + mo(j) - mo(o + a) - mo(o + b));
  return {t1n, t2n};
}

double so_triples(const T2 &t1, const T4 &t2, const SOEris &e) {
  occ::timing::start(occ::timing::category::ccsd_triples);
  const int o = e.nocc, v = e.nvir;
  const Vec &mo = e.mo_energy;
  const T4 vovv = -e.ovvv.shuffle(Sh4{1, 0, 2, 3}); // <ei||bc> = -<ie||bc>

  // connected
  T6 w = t2.contract(vovv, IA<1>{ip(3, 0)}).shuffle(Sh6{3, 0, 1, 2, 4, 5});
  w -= t2.contract(e.ovoo, IA<1>{ip(1, 0)}).shuffle(Sh6{0, 4, 5, 3, 1, 2});
  w = P_occ(P_vir(w));
  // disconnected
  T6 vd = t1.contract(e.oovv, IA<0>{}).shuffle(Sh6{0, 2, 3, 1, 4, 5});
  vd = P_occ(P_vir(vd));

  T6 denom(o, o, o, v, v, v);
  for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
      for (int k = 0; k < o; ++k)
        for (int a = 0; a < v; ++a)
          for (int b = 0; b < v; ++b)
            for (int c = 0; c < v; ++c)
              denom(i, j, k, a, b, c) = mo(i) + mo(j) + mo(k) - mo(o + a) -
                                        mo(o + b) - mo(o + c);
  const Eigen::Tensor<double, 0> et = (w * (w + vd) / denom).sum();
  occ::timing::stop(occ::timing::category::ccsd_triples);
  return et(0) / 36.0;
}

} // namespace

UCCSDResult uccsd_so(const AOBasis &basis, const MolecularOrbitals &mo,
                     int n_frozen, bool with_triples, int max_cycle,
                     double tol) {
  const SOEris e = build_so_eris(basis, mo, n_frozen);
  const int o = e.nocc, v = e.nvir;
  const Vec &mo_e = e.mo_energy;
  occ::log::info("Spin-orbital CCSD: {} active occ, {} virtual spin orbitals", o,
                 v);

  occ::timing::start(occ::timing::category::ccsd);
  T2 t1(o, v);
  t1.setZero();
  T4 t2(o, o, v, v);
  for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
      for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b)
          t2(i, j, a, b) =
              e.oovv(i, j, a, b) / (mo_e(i) + mo_e(j) - mo_e(o + a) - mo_e(o + b));

  const Eigen::Index n1 = static_cast<Eigen::Index>(o) * v;
  const Eigen::Index n2 = static_cast<Eigen::Index>(o) * o * v * v;
  occ::core::diis::DIIS diis;

  UCCSDResult res;
  occ::log::info("starting spin-orbital CCSD iterations");
  double e_old = so_energy(t1, t2, e);
  for (int it = 0; it < max_cycle; ++it) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    auto [t1n, t2n] = so_update_amps(t1, t2, e);
    Mat x(n1 + n2, 1), err(n1 + n2, 1);
    std::copy(t1n.data(), t1n.data() + n1, x.data());
    std::copy(t2n.data(), t2n.data() + n2, x.data() + n1);
    for (Eigen::Index k = 0; k < n1; ++k)
      err(k, 0) = t1n.data()[k] - t1.data()[k];
    for (Eigen::Index k = 0; k < n2; ++k)
      err(n1 + k, 0) = t2n.data()[k] - t2.data()[k];
    const double rnorm = err.norm();
    diis.extrapolate(x, err);
    std::copy(x.data(), x.data() + n1, t1n.data());
    std::copy(x.data() + n1, x.data() + n1 + n2, t2n.data());
    t1 = t1n;
    t2 = t2n;
    const double e_new = so_energy(t1, t2, e);
    const double de = e_new - e_old;
    const double secs = std::chrono::duration<double>(
                            std::chrono::high_resolution_clock::now() - t0)
                            .count();
    if (it == 0)
      occ::log::info("{:>4s} {: >20s} {: >12s} {: >12s}  {: >8s}", "#",
                     "E_corr (Ha)", "|dE|", "|dT|", "T (s)");
    occ::log::info("{:>4d} {:>20.12f} {:>12.5e} {:>12.5e}  {:>8.2e}", it + 1,
                   e_new, std::abs(de), rnorm, secs);
    occ::log::flush();
    res.iterations = it + 1;
    e_old = e_new;
    if (std::abs(de) < tol) {
      res.converged = true;
      break;
    }
  }
  occ::timing::stop(occ::timing::category::ccsd);
  res.e_corr = e_old;
  if (!res.converged)
    occ::log::warn("spin-orbital CCSD did not converge in {} iterations",
                   res.iterations);
  if (with_triples)
    res.e_triples = so_triples(t1, t2, e);
  return res;
}

double uccsd_t_via_so(const AOBasis &basis, const MolecularOrbitals &mo,
                      int n_frozen, const T2 &t1a, const T2 &t1b,
                      const T4 &t2aa, const T4 &t2ab, const T4 &t2bb) {
  const SOEris e = build_so_eris(basis, mo, n_frozen);
  const int no = e.nocc, nv = e.nvir;

  // Map the spatial spin-block amplitudes into the spin-orbital basis.
  auto t1val = [&](int P, int A) -> double { // P occ-global, A vir-global
    if (e.spin[P] != e.spin[A])
      return 0.0;
    return e.spin[P] == 0 ? t1a(e.amp[P], e.amp[A]) : t1b(e.amp[P], e.amp[A]);
  };
  auto t2val = [&](int P, int Q, int A, int B) -> double {
    const int sP = e.spin[P], sQ = e.spin[Q], sA = e.spin[A], sB = e.spin[B];
    if (sP + sQ != sA + sB) // spin conservation (#beta occ == #beta vir)
      return 0.0;
    if (sP == 0 && sQ == 0 && sA == 0 && sB == 0)
      return t2aa(e.amp[P], e.amp[Q], e.amp[A], e.amp[B]);
    if (sP == 1 && sQ == 1 && sA == 1 && sB == 1)
      return t2bb(e.amp[P], e.amp[Q], e.amp[A], e.amp[B]);
    // mixed: one alpha + one beta in occ and in vir -> canonical t2ab + sign
    int ia, Jb;
    double socc;
    if (sP == 0) { ia = e.amp[P]; Jb = e.amp[Q]; socc = 1.0; }
    else         { ia = e.amp[Q]; Jb = e.amp[P]; socc = -1.0; }
    int av, Bv;
    double svir;
    if (sA == 0) { av = e.amp[A]; Bv = e.amp[B]; svir = 1.0; }
    else         { av = e.amp[B]; Bv = e.amp[A]; svir = -1.0; }
    return socc * svir * t2ab(ia, Jb, av, Bv);
  };

  T2 t1(no, nv);
  for (int i = 0; i < no; ++i)
    for (int a = 0; a < nv; ++a)
      t1(i, a) = t1val(i, no + a);
  T4 t2(no, no, nv, nv);
  for (int i = 0; i < no; ++i)
    for (int j = 0; j < no; ++j)
      for (int a = 0; a < nv; ++a)
        for (int b = 0; b < nv; ++b)
          t2(i, j, a, b) = t2val(i, j, no + a, no + b);

  // Sanity check: the mapped amplitudes must reproduce the correlation energy.
  occ::log::debug("(T) via spin-orbital: mapped E_corr = {:.10f}",
                  so_energy(t1, t2, e));
  return so_triples(t1, t2, e);
}

} // namespace occ::qm::cc
