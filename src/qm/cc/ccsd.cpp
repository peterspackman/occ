#include <chrono>
#include <occ/core/diis.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/qm/cc/ccsd.h>

// Restricted closed-shell CCSD, a direct port of the thc_cct reference
// rccsd.py (PySCF/Hirata RCCSD equations). The O(V^4) vvvv ladder term goes
// through eris.ladder (exact / DF / THC); everything else uses the cheap
// blocks plus ovvv. A canonical reference is assumed (Fock diagonal, f_ov = 0),
// so the bare orbital energies live only in the amplitude denominators and the
// F/L intermediates carry the correlation parts only.

namespace occ::qm::cc {

namespace {

using T2 = Eigen::Tensor<double, 2>;
using T4 = Eigen::Tensor<double, 4>;
using Sh2 = Eigen::array<int, 2>;
using Sh4 = Eigen::array<int, 4>;

inline Eigen::IndexPair<int> ip(int a, int b) { return {a, b}; }
template <int N> using IA = Eigen::array<Eigen::IndexPair<int>, N>;

// t1(i,a) t1(j,b) -> (i,j,a,b)
T4 t1t1_outer(const T2 &t1) {
  const T4 o = t1.contract(t1, IA<0>{}); // (i,a,j,b)
  return o.shuffle(Sh4{0, 2, 1, 3});     // (i,j,a,b)
}

// P(ij,ab): x + x.transpose(1,0,3,2)
T4 sym_ijab(const T4 &x) { return x + x.shuffle(Sh4{1, 0, 3, 2}); }

} // namespace

double ccsd_energy(const T2 &t1, const T4 &t2, const CCIntegrals &eris) {
  const T4 &ovov = eris.ovov;
  const T4 tau = t2 + t1t1_outer(t1);
  // 2 (ijab,iajb) - (ijab,ibja); reindex ovov to (i,j,a,b)
  const T4 g_iajb = ovov.shuffle(Sh4{0, 2, 1, 3}); // (ia|jb)
  const T4 g_ibja = ovov.shuffle(Sh4{0, 2, 3, 1}); // (ib|ja)
  Eigen::Tensor<double, 0> s1 = (tau * g_iajb).sum();
  Eigen::Tensor<double, 0> s2 = (tau * g_ibja).sum();
  return 2.0 * s1(0) - s2(0);
}

namespace {

// One CCSD amplitude update. Returns (t1new, t2new) already divided by the
// orbital-energy denominators (matching rccsd.update_amps).
std::pair<T2, T4> update_amps(const T2 &t1, const T4 &t2,
                              const CCIntegrals &e) {
  const int o = e.nocc, v = e.nvir;
  const T4 &oooo = e.oooo, &ooov = e.ooov, &oovv = e.oovv, &ovoo = e.ovoo;
  const T4 &ovov = e.ovov, &ovvo = e.ovvo, &ovvv = e.ovvv;

  const T4 tt = t1t1_outer(t1);    // t1(i,a) t1(j,b) -> (i,j,a,b)
  const T4 tau = t2 + tt;          // make_tau
  const T4 z4 = 0.5 * t2 + tt;     // 0.5 t2 + t1t1 (Wvoov/Wvovo)

  // --- F intermediates (correlation parts; canonical -> no bare Fock) -----
  // Fov(k,c) = 2 (kc|ld) t1(ld) - (kd|lc) t1(ld)
  T2 Fov = 2.0 * ovov.contract(t1, IA<2>{ip(2, 0), ip(3, 1)});
  Fov -= ovov.contract(t1, IA<2>{ip(2, 0), ip(1, 1)});

  // Foo(k,i) = 2 (kc|ld) tau(ilcd) - (kd|lc) tau(ilcd)
  T2 Foo = 2.0 * ovov.contract(tau, IA<3>{ip(1, 2), ip(2, 1), ip(3, 3)});
  Foo -= ovov.contract(tau, IA<3>{ip(1, 3), ip(2, 1), ip(3, 2)});

  // Fvv(a,c) = -2 (kc|ld) tau(klad) + (kd|lc) tau(klad)
  T2 Fvv = -2.0 * tau.contract(ovov, IA<3>{ip(0, 0), ip(1, 2), ip(3, 3)});
  Fvv += tau.contract(ovov, IA<3>{ip(0, 0), ip(1, 2), ip(3, 1)});

  // --- L intermediates ----------------------------------------------------
  // Loo(k,i) = Foo + 2 (lc|ki) t1(lc) - (kc|li) t1(lc)
  T2 Loo = Foo;
  Loo += 2.0 * ovoo.contract(t1, IA<2>{ip(0, 0), ip(1, 1)});
  Loo -= ovoo.contract(t1, IA<2>{ip(2, 0), ip(1, 1)});

  // Lvv(a,c) = Fvv + 2 (kd|ac) t1(kd) - (kc|ad) t1(kd)
  T2 Lvv = Fvv;
  Lvv += 2.0 * ovvv.contract(t1, IA<2>{ip(0, 0), ip(1, 1)});
  Lvv -= ovvv.contract(t1, IA<2>{ip(0, 0), ip(3, 1)}).shuffle(Sh2{1, 0});

  // --- W intermediates ----------------------------------------------------
  // Woooo(k,l,i,j)
  const T4 X_ovoo_t1 = ovoo.contract(t1, IA<1>{ip(1, 1)}); // (l/k, k/l, i/j, j/i)
  T4 Woooo = X_ovoo_t1.shuffle(Sh4{1, 0, 2, 3});           // "lcki,jc"
  Woooo += X_ovoo_t1.shuffle(Sh4{0, 1, 3, 2});             // "kclj,ic"
  Woooo += ovov.contract(tau, IA<2>{ip(1, 2), ip(3, 3)});  // kcld,ijcd
  Woooo += oooo.shuffle(Sh4{0, 2, 1, 3});                  // (ki|lj)

  // Wvoov(a,k,i,c)
  T4 Wvoov = ovvv.contract(t1, IA<1>{ip(3, 1)}).shuffle(Sh4{2, 0, 3, 1}); // kcad,id
  Wvoov -= ovoo.contract(t1, IA<1>{ip(2, 0)}).shuffle(Sh4{3, 0, 2, 1});   // kcli,la
  Wvoov += ovvo.shuffle(Sh4{2, 0, 3, 1});                                 // (kc|ai)
  Wvoov -= ovov.contract(z4, IA<2>{ip(0, 1), ip(1, 2)}).shuffle(Sh4{3, 0, 2, 1});       // ldkc,ilda (0.5 t2 + t1t1)
  Wvoov -= 0.5 * ovov.contract(t2, IA<2>{ip(0, 1), ip(3, 3)}).shuffle(Sh4{3, 1, 2, 0}); // lckd,ilad
  Wvoov += ovov.contract(t2, IA<2>{ip(0, 1), ip(1, 3)}).shuffle(Sh4{3, 0, 2, 1});       // ldkc,ilad

  // Wvovo(a,k,c,i)
  T4 Wvovo = ovvv.contract(t1, IA<1>{ip(1, 1)}).shuffle(Sh4{1, 0, 2, 3}); // kdac,id
  Wvovo -= ovoo.contract(t1, IA<1>{ip(0, 0)}).shuffle(Sh4{3, 1, 0, 2});   // lcki,la
  Wvovo += oovv.shuffle(Sh4{2, 0, 3, 1});                                 // (ki|ac)
  Wvovo -= ovov.contract(z4, IA<2>{ip(0, 1), ip(3, 2)}).shuffle(Sh4{3, 1, 0, 2}); // lckd,ilda (0.5 t2 + t1t1)

  // --- T1 residual --------------------------------------------------------
  T2 r1 = t1.contract(Fvv, IA<1>{ip(1, 1)});         // ac,ic
  r1 -= Foo.contract(t1, IA<1>{ip(0, 0)});           // ki,ka
  r1 += 2.0 * Fov.contract(t2, IA<2>{ip(0, 0), ip(1, 2)}); // kc,kica
  r1 -= Fov.contract(t2, IA<2>{ip(0, 1), ip(1, 2)});       // kc,ikca
  {
    const T2 y = Fov.contract(t1, IA<1>{ip(0, 0)});  // (c,a)
    r1 += t1.contract(y, IA<1>{ip(1, 0)});           // kc,ic,ka
  }
  r1 += 2.0 * ovvo.contract(t1, IA<2>{ip(0, 0), ip(1, 1)}).shuffle(Sh2{1, 0}); // kcai,kc
  r1 -= oovv.contract(t1, IA<2>{ip(0, 0), ip(3, 1)});                          // kiac,kc
  r1 += 2.0 * ovvv.contract(t2, IA<3>{ip(0, 1), ip(1, 3), ip(3, 2)}).shuffle(Sh2{1, 0}); // kdac,ikcd
  r1 -= ovvv.contract(t2, IA<3>{ip(0, 1), ip(1, 2), ip(3, 3)}).shuffle(Sh2{1, 0});       // kcad,ikcd
  {
    const T2 y = ovvv.contract(t1, IA<2>{ip(0, 0), ip(1, 1)}); // (a,c)
    r1 += 2.0 * t1.contract(y, IA<1>{ip(1, 1)});               // kdac,kd,ic
    const T2 y2 = ovvv.contract(t1, IA<2>{ip(0, 0), ip(3, 1)}); // (c,a)
    r1 -= t1.contract(y2, IA<1>{ip(1, 0)});                     // kcad,kd,ic
  }
  r1 -= 2.0 * ooov.contract(t2, IA<3>{ip(0, 0), ip(2, 1), ip(3, 3)}); // kilc,klac
  r1 += ooov.contract(t2, IA<3>{ip(0, 1), ip(2, 0), ip(3, 3)});       // likc,klac
  {
    const T2 y = ooov.contract(t1, IA<2>{ip(2, 0), ip(3, 1)}); // (k,i)
    r1 -= 2.0 * t1.contract(y, IA<1>{ip(0, 0)}).shuffle(Sh2{1, 0}); // kilc,lc,ka
    const T2 y2 = ooov.contract(t1, IA<2>{ip(0, 0), ip(3, 1)}); // (i,k)
    r1 += y2.contract(t1, IA<1>{ip(1, 0)});                     // likc,lc,ka
  }

  // --- T2 residual --------------------------------------------------------
  T4 r2 = ovov.shuffle(Sh4{0, 2, 1, 3}); // (ia|jb) -> (i,j,a,b)
  r2 += Woooo.contract(tau, IA<2>{ip(0, 0), ip(1, 1)}); // klij,klab
  occ::timing::start(occ::timing::category::ccsd_ladder);
  r2 += e.ladder(tau); // vvvv ladder
  occ::timing::stop(occ::timing::category::ccsd_ladder);
  {
    const T4 b1 = ovvv.contract(tau, IA<2>{ip(1, 3), ip(3, 2)})
                      .shuffle(Sh4{2, 3, 1, 0}); // kdac,ijcd -> (i,j,a,k)
    r2 -= b1.contract(t1, IA<1>{ip(3, 0)});      // ijak,kb
    const T4 b2 = ovvv.contract(tau, IA<2>{ip(1, 2), ip(3, 3)})
                      .shuffle(Sh4{2, 3, 1, 0}); // kcbd,ijcd -> (i,j,b,k)
    r2 -= b2.contract(t1, IA<1>{ip(3, 0)}).shuffle(Sh4{0, 1, 3, 2}); // ijbk,ka
  }
  {
    const T4 tmp = Lvv.contract(t2, IA<1>{ip(1, 2)}).shuffle(Sh4{1, 2, 0, 3}); // ac,ijcb
    r2 += sym_ijab(tmp);
  }
  {
    const T4 tmp = Loo.contract(t2, IA<1>{ip(0, 0)}); // ki,kjab -> (i,j,a,b)
    r2 -= sym_ijab(tmp);
  }
  {
    T4 tmp = 2.0 * Wvoov.contract(t2, IA<2>{ip(1, 0), ip(3, 2)})
                       .shuffle(Sh4{1, 2, 0, 3}); // akic,kjcb
    tmp -= Wvovo.contract(t2, IA<2>{ip(1, 0), ip(2, 2)})
               .shuffle(Sh4{1, 2, 0, 3}); // akci,kjcb
    r2 += sym_ijab(tmp);
  }
  {
    const T4 tmp = Wvoov.contract(t2, IA<2>{ip(1, 0), ip(3, 3)})
                       .shuffle(Sh4{1, 2, 0, 3}); // akic,kjbc
    r2 -= sym_ijab(tmp);
  }
  {
    const T4 tmp = Wvovo.contract(t2, IA<2>{ip(1, 0), ip(2, 3)})
                       .shuffle(Sh4{1, 2, 3, 0}); // bkci,kjac
    r2 -= sym_ijab(tmp);
  }
  {
    // tmp2(a,b,i,c) = -(ki|bc) t1(ka) + (ia|cb form) ovvv[i,a,c,b]
    T4 tmp2 = -1.0 * oovv.contract(t1, IA<1>{ip(0, 0)}).shuffle(Sh4{3, 1, 0, 2});
    tmp2 += ovvv.shuffle(Sh4{1, 3, 0, 2}); // ovvv.transpose(1,3,0,2)
    const T4 tmp = tmp2.contract(t1, IA<1>{ip(3, 1)}).shuffle(Sh4{2, 3, 0, 1}); // abic,jc
    r2 += sym_ijab(tmp);
  }
  {
    // tmp2(a,k,i,j) = (kc|ai) t1(jc) + ooov.transpose(3,1,2,0)
    T4 tmp2 = ovvo.contract(t1, IA<1>{ip(1, 1)}).shuffle(Sh4{1, 0, 2, 3});
    tmp2 += ooov.shuffle(Sh4{3, 1, 2, 0});
    const T4 tmp = tmp2.contract(t1, IA<1>{ip(1, 0)}).shuffle(Sh4{1, 2, 0, 3}); // akij,kb
    r2 -= sym_ijab(tmp);
  }

  // --- divide by denominators --------------------------------------------
  const Vec &mo_e = e.mo_energy;
  T2 t1new(o, v);
  for (int i = 0; i < o; ++i)
    for (int a = 0; a < v; ++a)
      t1new(i, a) = r1(i, a) / (mo_e(i) - mo_e(o + a));
  T4 t2new(o, o, v, v);
  for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
      for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b)
          t2new(i, j, a, b) =
              r2(i, j, a, b) /
              (mo_e(i) + mo_e(j) - mo_e(o + a) - mo_e(o + b));
  return {t1new, t2new};
}

} // namespace

CCSDResult ccsd(const CCIntegrals &eris, const CCSDOptions &opts) {
  occ::timing::start(occ::timing::category::ccsd);
  const int o = eris.nocc, v = eris.nvir;
  const Vec &mo_e = eris.mo_energy;

  // MP1 guess: t1 = 0, t2 = (ia|jb) / Dijab
  T2 t1(o, v);
  t1.setZero();
  T4 t2(o, o, v, v);
  const T4 g_iajb = eris.ovov.shuffle(Sh4{0, 2, 1, 3});
  for (int i = 0; i < o; ++i)
    for (int j = 0; j < o; ++j)
      for (int a = 0; a < v; ++a)
        for (int b = 0; b < v; ++b)
          t2(i, j, a, b) =
              g_iajb(i, j, a, b) /
              (mo_e(i) + mo_e(j) - mo_e(o + a) - mo_e(o + b));

  const Eigen::Index n1 = static_cast<Eigen::Index>(o) * v;
  const Eigen::Index n2 = static_cast<Eigen::Index>(o) * o * v * v;
  occ::core::diis::DIIS diis;

  CCSDResult result;
  occ::log::info("starting CCSD iterations ({} occ, {} virt)", o, v);
  double e_old = ccsd_energy(t1, t2, eris);
  double total_time = 0.0;
  for (int it = 0; it < opts.max_cycle; ++it) {
    const auto tstart = std::chrono::high_resolution_clock::now();
    auto [t1n, t2n] = update_amps(t1, t2, eris);

    // Amplitude-change residual ||t_new - t_old|| (the DIIS error vector).
    Mat x(n1 + n2, 1), err(n1 + n2, 1);
    std::copy(t1n.data(), t1n.data() + n1, x.data());
    std::copy(t2n.data(), t2n.data() + n2, x.data() + n1);
    for (Eigen::Index k = 0; k < n1; ++k)
      err(k, 0) = t1n.data()[k] - t1.data()[k];
    for (Eigen::Index k = 0; k < n2; ++k)
      err(n1 + k, 0) = t2n.data()[k] - t2.data()[k];
    const double rnorm = err.norm();

    if (opts.diis) {
      diis.extrapolate(x, err);
      std::copy(x.data(), x.data() + n1, t1n.data());
      std::copy(x.data() + n1, x.data() + n1 + n2, t2n.data());
    }

    t1 = t1n;
    t2 = t2n;
    const double e_new = ccsd_energy(t1, t2, eris);
    const double de = e_new - e_old;
    const auto tstop = std::chrono::high_resolution_clock::now();
    const double secs = std::chrono::duration<double>(tstop - tstart).count();
    total_time += secs;

    if (it == 0)
      occ::log::info("{:>4s} {: >20s} {: >12s} {: >12s}  {: >8s}", "#",
                     "E_corr (Ha)", "|dE|", "|dT|", "T (s)");
    occ::log::info("{:>4d} {:>20.12f} {:>12.5e} {:>12.5e}  {:>8.2e}", it + 1,
                   e_new, std::abs(de), rnorm, secs);
    occ::log::flush();

    result.iterations = it + 1;
    e_old = e_new;
    if (std::abs(de) < opts.tol) {
      result.converged = true;
      break;
    }
  }
  occ::log::info("CCSD {} after {} iterations ({:.3f} s)",
                 result.converged ? "converged" : "NOT converged",
                 result.iterations, total_time);

  result.e_corr = e_old;
  result.t1 = t1;
  result.t2 = t2;
  occ::timing::stop(occ::timing::category::ccsd);
  return result;
}

} // namespace occ::qm::cc
