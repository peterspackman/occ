#include <chrono>
#include <occ/core/diis.h>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/qm/cc/gemm.h> // pcon2 (parallel 2-index contraction)
#include <occ/qm/cc/uccsd.h>
#include <occ/qm/cc/uintegrals.h>
#include <occ/qm/cc/utriples.h> // native spin-adapted (T)
#include <unsupported/Eigen/CXX11/Tensor>

// Spin-adapted unrestricted CCSD -- a faithful port of PySCF cc/uccsd.py
// (update_amps + energy). Amplitudes are spin-blocked spatial tensors
// (t1a,t1b; t2aa,t2ab,t2bb); integrals come from UCCIntegrals. A canonical UHF
// reference is assumed (f_ov = 0, Fock diagonal), so the bare-Fock pieces drop
// out and the orbital energies live only in the denominators -- exactly the
// simplification the restricted and spin-orbital ports already use.
//
// numpy transpose(perm) == Eigen shuffle(perm): result.axis[i] = in.axis[perm[i]].

namespace occ::qm::cc {

namespace {

using occ::Mat;
using occ::Vec;
using T2 = Eigen::Tensor<double, 2>;
using T4 = Eigen::Tensor<double, 4>;
using Sh2 = Eigen::array<int, 2>;
using Sh4 = Eigen::array<int, 4>;
template <int N> using IA = Eigen::array<Eigen::IndexPair<int>, N>;
inline Eigen::IndexPair<int> ip(int a, int b) { return {a, b}; }
inline double sum0(const T4 &x) {
  Eigen::Tensor<double, 0> s = x.sum();
  return s(0);
}

struct Amps {
  T2 t1a, t1b;
  T4 t2aa, t2ab, t2bb;
};

// outer(i,j,a,b) = x(i,a) y(j,b)
T4 outer(const T2 &x, const T2 &y) {
  return x.contract(y, IA<0>{}).shuffle(Sh4{0, 2, 1, 3});
}

// make_tau_aa: t2aa + 0.5*fac * P(ij)P(ab)[ t1a(i,a) t1a(j,b) ]
T4 tau_aa(const T4 &t2aa, const T2 &t1a, double fac) {
  T4 A = outer(t1a, t1a);
  T4 t = A - A.shuffle(Sh4{1, 0, 2, 3});
  // fresh tensor: `t = t - t.shuffle(...)` would self-alias under Eigen's
  // in-place element-wise assignment and corrupt the highest-index slice.
  T4 ta = t - t.shuffle(Sh4{0, 1, 3, 2});
  return t2aa + ta * (0.5 * fac);
}
// make_tau_ab: t2ab + fac * t1a(i,a) t1b(j,b)
T4 tau_ab(const T4 &t2ab, const T2 &t1a, const T2 &t1b, double fac) {
  return t2ab + outer(t1a, t1b) * fac;
}

double uccsd_energy(const Amps &a, const UCCIntegrals &e) {
  const T4 &ovov = e.ovov, &OVOV = e.OVOV, &ovOV = e.ovOV;
  double en = 0.0;
  en += 0.25 * sum0(a.t2aa * ovov.shuffle(Sh4{0, 2, 1, 3}));
  en -= 0.25 * sum0(a.t2aa * ovov.shuffle(Sh4{0, 2, 3, 1}));
  en += 0.25 * sum0(a.t2bb * OVOV.shuffle(Sh4{0, 2, 1, 3}));
  en -= 0.25 * sum0(a.t2bb * OVOV.shuffle(Sh4{0, 2, 3, 1}));
  en += sum0(a.t2ab * ovOV.shuffle(Sh4{0, 2, 1, 3}));
  const T4 taa = outer(a.t1a, a.t1a);
  const T4 tbb = outer(a.t1b, a.t1b);
  const T4 tab = outer(a.t1a, a.t1b);
  en += 0.5 * sum0(taa * ovov.shuffle(Sh4{0, 2, 1, 3}));
  en -= 0.5 * sum0(taa * ovov.shuffle(Sh4{0, 2, 3, 1}));
  en += 0.5 * sum0(tbb * OVOV.shuffle(Sh4{0, 2, 1, 3}));
  en -= 0.5 * sum0(tbb * OVOV.shuffle(Sh4{0, 2, 3, 1}));
  en += sum0(tab * ovOV.shuffle(Sh4{0, 2, 1, 3}));
  return en;
}

// One spin-adapted UCCSD amplitude update, returning the new amplitudes already
// divided by the orbital-energy denominators (matching uccsd.update_amps).
Amps update_amps(const Amps &a, const UCCIntegrals &e) {
  const T2 &t1a = a.t1a, &t1b = a.t1b;
  const T4 &t2aa = a.t2aa, &t2ab = a.t2ab, &t2bb = a.t2bb;
  const int oa = e.nocca, ob = e.noccb, va = e.nvira, vb = e.nvirb;

  // tau (fac=1) and ladders ------------------------------------------------
  const T4 tauaa = tau_aa(t2aa, t1a, 1.0);
  const T4 taubb = tau_aa(t2bb, t1b, 1.0);
  const T4 tauab = tau_ab(t2ab, t1a, t1b, 1.0);
  T4 u2aa = e.ladder_aa(tauaa);
  T4 u2bb = e.ladder_bb(taubb);
  T4 u2ab = e.ladder_ab(tauab);
  u2aa = u2aa * 0.5;
  u2bb = u2bb * 0.5;

  T2 u1a(oa, va);
  u1a.setZero();
  T2 u1b(ob, vb);
  u1b.setZero();

  // F intermediates (canonical -> start from zero) -------------------------
  T2 Fooa(oa, oa), Foob(ob, ob), Fvva(va, va), Fvvb(vb, vb);
  Fooa.setZero();
  Foob.setZero();
  Fvva.setZero();
  Fvvb.setZero();
  T2 Fova(oa, va), Fovb(ob, vb);
  Fova.setZero();
  Fovb.setZero();

  T4 wovvo(oa, va, va, oa);
  wovvo.setZero();
  T4 wOVVO(ob, vb, vb, ob);
  wOVVO.setZero();
  T4 woVvO(oa, vb, va, ob);
  woVvO.setZero();
  T4 woVVo(oa, vb, vb, oa);
  woVVo.setZero();
  T4 wOvVo(ob, va, vb, oa);
  wOvVo.setZero();
  T4 wOvvO(ob, va, va, ob);
  wOvvO.setZero();

  // ovvv (alpha) -----------------------------------------------------------
  {
    const T4 ovvv = e.ovvv - e.ovvv.shuffle(Sh4{0, 3, 2, 1}); // (m,e,a,f)
    Fvva += t1a.contract(ovvv, IA<2>{ip(0, 0), ip(1, 1)});    // mf,mfae->ae
    wovvo += ovvv.contract(t1a, IA<1>{ip(3, 1)}).shuffle(Sh4{0, 2, 1, 3}); // jf,mebf->mbej
    u1a += 0.5 * t2aa.contract(ovvv, IA<3>{ip(0, 0), ip(2, 1), ip(3, 3)}); // mief,meaf->ia
    u2aa += t1a.contract(ovvv, IA<1>{ip(1, 2)}).shuffle(Sh4{0, 1, 3, 2});  // ie,mbea->imab
    const T4 tmp = pcon2(tauaa, 2, 3, ovvv, 1, 3);        // ijef,mebf->ijmb
    u2aa -= 0.5 * tmp.contract(t1a, IA<1>{ip(2, 0)}).shuffle(Sh4{0, 1, 3, 2}); // ijmb,ma->ijab
  }
  // OVVV (beta) ------------------------------------------------------------
  {
    const T4 OVVV = e.OVVV - e.OVVV.shuffle(Sh4{0, 3, 2, 1});
    Fvvb += t1b.contract(OVVV, IA<2>{ip(0, 0), ip(1, 1)});
    wOVVO += OVVV.contract(t1b, IA<1>{ip(3, 1)}).shuffle(Sh4{0, 2, 1, 3});
    u1b += 0.5 * t2bb.contract(OVVV, IA<3>{ip(0, 0), ip(2, 1), ip(3, 3)});
    u2bb += t1b.contract(OVVV, IA<1>{ip(1, 2)}).shuffle(Sh4{0, 1, 3, 2});
    const T4 tmp = pcon2(taubb, 2, 3, OVVV, 1, 3);
    u2bb -= 0.5 * tmp.contract(t1b, IA<1>{ip(2, 0)}).shuffle(Sh4{0, 1, 3, 2});
  }
  // ovVV (alpha occ/vir, beta vir) ----------------------------------------
  {
    const T4 &ovVV = e.ovVV; // (m,f,A,E) = (mf|AE)
    Fvvb += t1a.contract(ovVV, IA<2>{ip(0, 0), ip(1, 1)});                 // mf,mfAE->AE
    woVvO += ovVV.contract(t1b, IA<1>{ip(3, 1)}).shuffle(Sh4{0, 2, 1, 3}); // JF,meBF->mBeJ
    woVVo += (-t1a).contract(ovVV, IA<1>{ip(1, 1)}).shuffle(Sh4{1, 2, 3, 0}); // jf,mfBE->mBEj
    u1b += t2ab.contract(ovVV, IA<3>{ip(0, 0), ip(2, 1), ip(3, 3)});       // mIeF,meAF->IA
    u2ab += t1b.contract(ovVV, IA<1>{ip(1, 2)}).shuffle(Sh4{1, 0, 2, 3}); // IE,maEB->mIaB
    const T4 tmp = pcon2(tauab, 2, 3, ovVV, 1, 3);       // iJeF,meBF->iJmB
    u2ab -= tmp.contract(t1a, IA<1>{ip(2, 0)}).shuffle(Sh4{0, 1, 3, 2});  // iJmB,ma->iJaB
  }
  // OVvv (beta occ/vir, alpha vir) ----------------------------------------
  {
    const T4 &OVvv = e.OVvv; // (M,F,a,e) = (MF|ae)
    Fvva += t1b.contract(OVvv, IA<2>{ip(0, 0), ip(1, 1)});                 // MF,MFae->ae
    wOvVo += OVvv.contract(t1a, IA<1>{ip(3, 1)}).shuffle(Sh4{0, 2, 1, 3}); // jf,MEbf->MbEj
    wOvvO += (-t1b).contract(OVvv, IA<1>{ip(1, 1)}).shuffle(Sh4{1, 2, 3, 0}); // JF,MFbe->MbeJ
    u1a += t2ab.contract(OVvv, IA<3>{ip(1, 0), ip(3, 1), ip(2, 3)});      // iMfE,MEaf->ia
    u2ab += t1a.contract(OVvv, IA<1>{ip(1, 2)}).shuffle(Sh4{0, 1, 3, 2}); // ie,MBea->iMaB
    const T4 tmp = pcon2(tauab, 2, 3, OVvv, 3, 1);       // iJeF,MFbe->iJMb
    u2ab -= tmp.contract(t1b, IA<1>{ip(2, 0)});                           // iJMb,MA->iJbA
  }

  // Woooo (alpha) ----------------------------------------------------------
  {
    const T4 &ovoo = e.ovoo, &ovov = e.ovov;
    T4 W0 = ovoo.contract(t1a, IA<1>{ip(1, 1)}).shuffle(Sh4{1, 0, 2, 3}); // je,nemi->mnij
    T4 Woooo = W0 - W0.shuffle(Sh4{0, 1, 3, 2});
    Woooo += e.oooo.shuffle(Sh4{0, 2, 1, 3});
    Woooo += 0.5 * pcon2(ovov, 1, 3, tauaa, 2, 3); // ijef,menf->mnij
    u2aa += 0.5 * pcon2(Woooo, 0, 1, tauaa, 0, 1); // mnab,mnij->ijab
  }
  // ovoo (alpha) -----------------------------------------------------------
  {
    const T4 ovoo = e.ovoo - e.ovoo.shuffle(Sh4{2, 1, 0, 3}); // (m,e,n,i)
    Fooa += t1a.contract(ovoo, IA<2>{ip(0, 0), ip(1, 1)});    // ne,nemi->mi
    u1a += 0.5 * t2aa.contract(ovoo, IA<3>{ip(0, 0), ip(1, 2), ip(3, 1)})
                     .shuffle(Sh2{1, 0});                       // mnae,meni->ia
    wovvo += t1a.contract(ovoo, IA<1>{ip(0, 0)}).shuffle(Sh4{2, 0, 1, 3}); // nb,nemj->mbej
  }
  // tilaa / ovov_as (alpha) ------------------------------------------------
  {
    const T4 tilaa = tau_aa(t2aa, t1a, 0.5);
    const T4 ovov = e.ovov - e.ovov.shuffle(Sh4{0, 3, 2, 1}); // (m,e,n,f)
    Fvva -= 0.5 * tilaa.contract(ovov, IA<3>{ip(0, 0), ip(1, 2), ip(3, 3)}); // mnaf,menf->ae
    Fooa += 0.5 * ovov.contract(tilaa, IA<3>{ip(1, 2), ip(2, 1), ip(3, 3)}); // inef,menf->mi
    Fova += ovov.contract(t1a, IA<2>{ip(2, 0), ip(3, 1)});                   // nf,menf->me
    u2aa += 0.5 * ovov.shuffle(Sh4{0, 2, 1, 3});
    wovvo -= 0.5 * pcon2(ovov, 2, 3, t2aa, 1, 2)
                       .shuffle(Sh4{0, 3, 1, 2}); // jnfb,menf->mbej
    woVvO += 0.5 * pcon2(ovov, 2, 3, t2ab, 0, 2)
                       .shuffle(Sh4{0, 3, 1, 2}); // nJfB,menf->mBeJ
    const T4 tmpaa =
        ovov.contract(t1a, IA<1>{ip(3, 1)}).shuffle(Sh4{0, 2, 1, 3}); // jf,menf->mnej
    wovvo -= t1a.contract(tmpaa, IA<1>{ip(0, 1)}).shuffle(Sh4{1, 0, 2, 3}); // nb,mnej->mbej
  }
  // WOOOO (beta) -----------------------------------------------------------
  {
    const T4 &OVOO = e.OVOO, &OVOV = e.OVOV;
    T4 W0 = OVOO.contract(t1b, IA<1>{ip(1, 1)}).shuffle(Sh4{1, 0, 2, 3});
    T4 WOOOO = W0 - W0.shuffle(Sh4{0, 1, 3, 2});
    WOOOO += e.OOOO.shuffle(Sh4{0, 2, 1, 3});
    WOOOO += 0.5 * pcon2(OVOV, 1, 3, taubb, 2, 3);
    u2bb += 0.5 * pcon2(WOOOO, 0, 1, taubb, 0, 1);
  }
  // OVOO (beta) ------------------------------------------------------------
  {
    const T4 OVOO = e.OVOO - e.OVOO.shuffle(Sh4{2, 1, 0, 3});
    Foob += t1b.contract(OVOO, IA<2>{ip(0, 0), ip(1, 1)});
    u1b += 0.5 * t2bb.contract(OVOO, IA<3>{ip(0, 0), ip(1, 2), ip(3, 1)})
                     .shuffle(Sh2{1, 0});
    wOVVO += t1b.contract(OVOO, IA<1>{ip(0, 0)}).shuffle(Sh4{2, 0, 1, 3});
  }
  // tilbb / OVOV_as (beta) -------------------------------------------------
  {
    const T4 tilbb = tau_aa(t2bb, t1b, 0.5);
    const T4 OVOV = e.OVOV - e.OVOV.shuffle(Sh4{0, 3, 2, 1});
    Fvvb -= 0.5 * tilbb.contract(OVOV, IA<3>{ip(0, 0), ip(1, 2), ip(3, 3)});
    Foob += 0.5 * OVOV.contract(tilbb, IA<3>{ip(1, 2), ip(2, 1), ip(3, 3)});
    Fovb += OVOV.contract(t1b, IA<2>{ip(2, 0), ip(3, 1)});
    u2bb += 0.5 * OVOV.shuffle(Sh4{0, 2, 1, 3});
    wOVVO -= 0.5 * pcon2(OVOV, 2, 3, t2bb, 1, 2)
                       .shuffle(Sh4{0, 3, 1, 2}); // jnfb,menf->mbej
    wOvVo += 0.5 * pcon2(OVOV, 2, 3, t2ab, 1, 3)
                       .shuffle(Sh4{0, 3, 1, 2}); // jNbF,MENF->MbEj
    const T4 tmpbb =
        OVOV.contract(t1b, IA<1>{ip(3, 1)}).shuffle(Sh4{0, 2, 1, 3});
    wOVVO -= t1b.contract(tmpbb, IA<1>{ip(0, 1)}).shuffle(Sh4{1, 0, 2, 3});
  }
  // OVoo / ovOO (mixed) ----------------------------------------------------
  T4 WoOoO(oa, ob, oa, ob);
  {
    const T4 &OVoo = e.OVoo, &ovOO = e.ovOO;
    Fooa += t1b.contract(OVoo, IA<2>{ip(0, 0), ip(1, 1)});                 // NE,NEmi->mi
    u1a -= t2ab.contract(OVoo, IA<3>{ip(0, 2), ip(1, 0), ip(3, 1)})
               .shuffle(Sh2{1, 0});                                        // nMaE,MEni->ia
    wOvVo -= t1a.contract(OVoo, IA<1>{ip(0, 2)}).shuffle(Sh4{1, 0, 2, 3}); // nb,MEnj->MbEj
    woVVo += t1b.contract(OVoo, IA<1>{ip(0, 0)}).shuffle(Sh4{2, 0, 1, 3}); // NB,NEmj->mBEj
    Foob += t1a.contract(ovOO, IA<2>{ip(0, 0), ip(1, 1)});                 // ne,neMI->MI
    u1b -= t2ab.contract(ovOO, IA<3>{ip(0, 0), ip(1, 2), ip(2, 1)})
               .shuffle(Sh2{1, 0});                                        // mNeA,meNI->IA
    woVvO -= t1b.contract(ovOO, IA<1>{ip(0, 2)}).shuffle(Sh4{1, 0, 2, 3}); // NB,meNJ->mBeJ
    wOvvO += t1a.contract(ovOO, IA<1>{ip(0, 0)}).shuffle(Sh4{2, 0, 1, 3}); // nb,neMJ->MbeJ
    WoOoO = OVoo.contract(t1b, IA<1>{ip(1, 1)}).shuffle(Sh4{1, 0, 2, 3});  // JE,NEmi->mNiJ
    WoOoO += ovOO.contract(t1a, IA<1>{ip(1, 1)}).shuffle(Sh4{0, 1, 3, 2}); // je,neMI->nMjI
    WoOoO += e.ooOO.shuffle(Sh4{0, 2, 1, 3});
  }
  // ovOV (mixed) -----------------------------------------------------------
  {
    const T4 &ovOV = e.ovOV;
    WoOoO += pcon2(ovOV, 1, 3, tauab, 2, 3); // iJeF,meNF->mNiJ
    u2ab += pcon2(WoOoO, 0, 1, tauab, 0, 1); // mNaB,mNiJ->iJaB
    const T4 tilab = tau_ab(t2ab, t1a, t1b, 0.5);
    Fvva -= tilab.contract(ovOV, IA<3>{ip(0, 0), ip(1, 2), ip(3, 3)});     // mNaF,meNF->ae
    Fvvb -= tilab.contract(ovOV, IA<3>{ip(0, 0), ip(2, 1), ip(1, 2)});     // nMfA,nfME->AE
    Fooa += ovOV.contract(tilab, IA<3>{ip(1, 2), ip(2, 1), ip(3, 3)});     // iNeF,meNF->mi
    Foob += ovOV.contract(tilab, IA<3>{ip(0, 0), ip(1, 2), ip(3, 3)});     // nIfE,nfME->MI
    Fova += ovOV.contract(t1b, IA<2>{ip(2, 0), ip(3, 1)});                 // NF,meNF->me
    Fovb += t1a.contract(ovOV, IA<2>{ip(0, 0), ip(1, 1)});                 // nf,nfME->ME
    u2ab += ovOV.shuffle(Sh4{0, 2, 1, 3});
    wovvo += 0.5 * pcon2(ovOV, 2, 3, t2ab, 1, 3)
                       .shuffle(Sh4{0, 3, 1, 2}); // jNbF,meNF->mbej
    wOVVO += 0.5 * pcon2(ovOV, 0, 1, t2ab, 0, 2)
                       .shuffle(Sh4{0, 3, 1, 2}); // nJfB,nfME->MBEJ
    wOvVo -= 0.5 * pcon2(ovOV, 0, 1, t2aa, 1, 2)
                       .shuffle(Sh4{0, 3, 1, 2}); // jnfb,nfME->MbEj
    woVvO -= 0.5 * pcon2(ovOV, 2, 3, t2bb, 1, 2)
                       .shuffle(Sh4{0, 3, 1, 2}); // JNFB,meNF->mBeJ
    woVVo += 0.5 * pcon2(ovOV, 1, 2, t2ab, 2, 1)
                       .shuffle(Sh4{0, 3, 1, 2}); // jNfB,mfNE->mBEj
    wOvvO += 0.5 * pcon2(ovOV, 0, 3, t2ab, 0, 3)
                       .shuffle(Sh4{1, 3, 0, 2}); // nJbF,neMF->MbeJ
    const T4 tmpabab =
        ovOV.contract(t1b, IA<1>{ip(3, 1)}).shuffle(Sh4{0, 2, 1, 3}); // JF,meNF->mNeJ
    const T4 tmpbaba =
        ovOV.contract(t1a, IA<1>{ip(1, 1)}).shuffle(Sh4{1, 0, 2, 3}); // jf,nfME->MnEj
    woVvO -= t1b.contract(tmpabab, IA<1>{ip(0, 1)}).shuffle(Sh4{1, 0, 2, 3}); // NB,mNeJ->mBeJ
    wOvVo -= tmpbaba.contract(t1a, IA<1>{ip(1, 0)}).shuffle(Sh4{0, 3, 1, 2}); // nb,MnEj->MbEj
    woVVo += t1b.contract(tmpbaba, IA<1>{ip(0, 0)}).shuffle(Sh4{1, 0, 2, 3}); // NB,NmEj->mBEj
    wOvvO += t1a.contract(tmpabab, IA<1>{ip(0, 0)}).shuffle(Sh4{1, 0, 2, 3}); // nb,nMeJ->MbeJ
  }

  // u1 Fock-like contractions (canonical: fova/fovb are integral-built) -----
  u1a += t1a.contract(Fvva, IA<1>{ip(1, 1)});            // ie,ae->ia
  u1a -= Fooa.contract(t1a, IA<1>{ip(0, 0)});            // ma,mi->ia
  u1a -= t2aa.contract(Fova, IA<2>{ip(1, 0), ip(2, 1)}); // imea,me->ia
  u1a += t2ab.contract(Fovb, IA<2>{ip(1, 0), ip(3, 1)}); // iMaE,ME->ia
  u1b += t1b.contract(Fvvb, IA<1>{ip(1, 1)});
  u1b -= Foob.contract(t1b, IA<1>{ip(0, 0)});
  u1b -= t2bb.contract(Fovb, IA<2>{ip(1, 0), ip(2, 1)});
  u1b += t2ab.contract(Fova, IA<2>{ip(0, 0), ip(2, 1)}); // mIeA,me->IA

  // oovv / ovvo (alpha) ----------------------------------------------------
  {
    wovvo -= e.oovv.shuffle(Sh4{0, 2, 3, 1});
    wovvo += e.ovvo.shuffle(Sh4{0, 2, 1, 3});
    const T4 oovv = e.oovv - e.ovvo.shuffle(Sh4{0, 3, 2, 1}); // (n,i,a,f)
    u1a -= t1a.contract(oovv, IA<2>{ip(0, 0), ip(1, 3)});     // nf,niaf->ia
    const T4 tmp = oovv.contract(t1a, IA<1>{ip(3, 1)}).shuffle(Sh4{0, 2, 3, 1}); // ie,mjbe->mbij
    u2aa += 2.0 * t1a.contract(tmp, IA<1>{ip(0, 0)}).shuffle(Sh4{2, 3, 0, 1});   // ma,mbij->ijab
  }
  // OOVV / OVVO (beta) -----------------------------------------------------
  {
    wOVVO -= e.OOVV.shuffle(Sh4{0, 2, 3, 1});
    wOVVO += e.OVVO.shuffle(Sh4{0, 2, 1, 3});
    const T4 OOVV = e.OOVV - e.OVVO.shuffle(Sh4{0, 3, 2, 1});
    u1b -= t1b.contract(OOVV, IA<2>{ip(0, 0), ip(1, 3)});
    const T4 tmp = OOVV.contract(t1b, IA<1>{ip(3, 1)}).shuffle(Sh4{0, 2, 3, 1});
    u2bb += 2.0 * t1b.contract(tmp, IA<1>{ip(0, 0)}).shuffle(Sh4{2, 3, 0, 1});
  }
  // ooVV / ovVO (mixed) ----------------------------------------------------
  {
    woVVo -= e.ooVV.shuffle(Sh4{0, 2, 3, 1});
    woVvO += e.ovVO.shuffle(Sh4{0, 2, 1, 3});
    u1b += t1a.contract(e.ovVO, IA<2>{ip(0, 0), ip(1, 1)}).shuffle(Sh2{1, 0}); // nf,nfAI->IA
    T4 tmp1ab = e.ovVO.contract(t1a, IA<1>{ip(1, 1)}).shuffle(Sh4{0, 1, 3, 2}); // ie,meBJ->mBiJ
    tmp1ab += e.ooVV.contract(t1b, IA<1>{ip(3, 1)}).shuffle(Sh4{0, 2, 1, 3});   // IE,mjBE->mBjI
    u2ab -= t1a.contract(tmp1ab, IA<1>{ip(0, 0)}).shuffle(Sh4{2, 3, 0, 1});     // ma,mBiJ->iJaB
  }
  // OOvv / OVvo (mixed) ----------------------------------------------------
  {
    wOvvO -= e.OOvv.shuffle(Sh4{0, 2, 3, 1});
    wOvVo += e.OVvo.shuffle(Sh4{0, 2, 1, 3});
    u1a += t1b.contract(e.OVvo, IA<2>{ip(0, 0), ip(1, 1)}).shuffle(Sh2{1, 0}); // NF,NFai->ia
    T4 tmp1ba = e.OVvo.contract(t1b, IA<1>{ip(1, 1)}).shuffle(Sh4{0, 1, 3, 2}); // IE,MEbj->MbIj
    tmp1ba += e.OOvv.contract(t1a, IA<1>{ip(3, 1)}).shuffle(Sh4{0, 2, 1, 3});   // ie,MJbe->MbJi
    u2ab -= t1b.contract(tmp1ba, IA<1>{ip(0, 0)}).shuffle(Sh4{3, 2, 1, 0});     // MA,MbIj->jIbA
  }

  // ring terms -------------------------------------------------------------
  u2aa += 2.0 * pcon2(t2aa, 1, 3, wovvo, 0, 2)
                    .shuffle(Sh4{0, 3, 1, 2}); // imae,mbej->ijab
  u2aa += 2.0 * pcon2(t2ab, 1, 3, wOvVo, 0, 2)
                    .shuffle(Sh4{0, 3, 1, 2}); // iMaE,MbEj->ijab
  u2bb += 2.0 * pcon2(t2bb, 1, 3, wOVVO, 0, 2)
                    .shuffle(Sh4{0, 3, 1, 2});
  u2bb += 2.0 * pcon2(t2ab, 0, 2, woVvO, 0, 2)
                    .shuffle(Sh4{0, 3, 1, 2}); // mIeA,mBeJ->IJAB
  u2ab += pcon2(t2aa, 1, 3, woVvO, 0, 2)
              .shuffle(Sh4{0, 3, 1, 2}); // imae,mBeJ->iJaB
  u2ab += pcon2(t2ab, 1, 3, wOVVO, 0, 2)
              .shuffle(Sh4{0, 3, 1, 2}); // iMaE,MBEJ->iJaB
  u2ab += pcon2(t2ab, 1, 2, wOvvO, 0, 2)
              .shuffle(Sh4{0, 3, 2, 1}); // iMeA,MbeJ->iJbA
  u2ab += pcon2(t2bb, 1, 3, wOvVo, 0, 2)
              .shuffle(Sh4{3, 0, 2, 1}); // IMAE,MbEj->jIbA
  u2ab += pcon2(t2ab, 0, 2, wovvo, 0, 2)
              .shuffle(Sh4{3, 0, 2, 1}); // mIeA,mbej->jIbA
  u2ab += pcon2(t2ab, 0, 3, woVVo, 0, 2)
              .shuffle(Sh4{3, 0, 1, 2}); // mIaE,mBEj->jIaB
  // Ftmp terms -------------------------------------------------------------
  {
    const T2 Ftmpa = Fvva - 0.5 * t1a.contract(Fova, IA<1>{ip(0, 0)}); // mb,me->be
    const T2 Ftmpb = Fvvb - 0.5 * t1b.contract(Fovb, IA<1>{ip(0, 0)});
    u2aa += t2aa.contract(Ftmpa, IA<1>{ip(3, 1)}); // ijae,be->ijab
    u2bb += t2bb.contract(Ftmpb, IA<1>{ip(3, 1)});
    u2ab += t2ab.contract(Ftmpb, IA<1>{ip(3, 1)});                       // iJaE,BE->iJaB
    u2ab += t2ab.contract(Ftmpa, IA<1>{ip(2, 1)}).shuffle(Sh4{0, 1, 3, 2}); // iJeA,be->iJbA
  }
  {
    const T2 Gooa = Fooa + 0.5 * Fova.contract(t1a, IA<1>{ip(1, 1)}); // je,me->mj
    const T2 Goob = Foob + 0.5 * Fovb.contract(t1b, IA<1>{ip(1, 1)});
    u2aa -= t2aa.contract(Gooa, IA<1>{ip(1, 0)}).shuffle(Sh4{0, 3, 1, 2}); // imab,mj->ijab
    u2bb -= t2bb.contract(Goob, IA<1>{ip(1, 0)}).shuffle(Sh4{0, 3, 1, 2});
    u2ab -= t2ab.contract(Goob, IA<1>{ip(1, 0)}).shuffle(Sh4{0, 3, 1, 2}); // iMaB,MJ->iJaB
    u2ab -= t2ab.contract(Gooa, IA<1>{ip(0, 0)}).shuffle(Sh4{3, 0, 1, 2}); // mIaB,mj->jIaB
  }

  // ovoo final -------------------------------------------------------------
  {
    const T4 ovoo = e.ovoo - e.ovoo.shuffle(Sh4{2, 1, 0, 3});
    const T4 OVOO = e.OVOO - e.OVOO.shuffle(Sh4{2, 1, 0, 3});
    u2aa -= t1a.contract(ovoo, IA<1>{ip(0, 3)}).shuffle(Sh4{3, 1, 0, 2}); // ma,jbim->ijab
    u2bb -= t1b.contract(OVOO, IA<1>{ip(0, 3)}).shuffle(Sh4{3, 1, 0, 2});
    u2ab -= t1a.contract(e.OVoo, IA<1>{ip(0, 3)}).shuffle(Sh4{3, 1, 0, 2}); // ma,JBim->iJaB
    u2ab -= t1b.contract(e.ovOO, IA<1>{ip(0, 3)}).shuffle(Sh4{1, 3, 2, 0}); // MA,ibJM->iJbA
  }

  // finalise u2aa/u2bb: *0.5 then antisymmetrise. Each `X = X - X.shuffle`
  // must go through a fresh tensor -- in-place self-aliasing corrupts the
  // highest-index slice under Eigen's element-wise assignment.
  u2aa = u2aa * 0.5;
  u2bb = u2bb * 0.5;
  {
    T4 a1 = u2aa - u2aa.shuffle(Sh4{0, 1, 3, 2});
    u2aa = a1 - a1.shuffle(Sh4{1, 0, 2, 3});
    T4 b1 = u2bb - u2bb.shuffle(Sh4{0, 1, 3, 2});
    u2bb = b1 - b1.shuffle(Sh4{1, 0, 2, 3});
  }

  // denominators -----------------------------------------------------------
  const Vec &ea = e.mo_energy_a, &eb = e.mo_energy_b;
  Amps out;
  out.t1a = T2(oa, va);
  for (int i = 0; i < oa; ++i)
    for (int x = 0; x < va; ++x)
      out.t1a(i, x) = u1a(i, x) / (ea(i) - ea(oa + x));
  out.t1b = T2(ob, vb);
  for (int i = 0; i < ob; ++i)
    for (int x = 0; x < vb; ++x)
      out.t1b(i, x) = u1b(i, x) / (eb(i) - eb(ob + x));
  out.t2aa = T4(oa, oa, va, va);
  for (int i = 0; i < oa; ++i)
    for (int j = 0; j < oa; ++j)
      for (int x = 0; x < va; ++x)
        for (int y = 0; y < va; ++y)
          out.t2aa(i, j, x, y) =
              u2aa(i, j, x, y) / (ea(i) + ea(j) - ea(oa + x) - ea(oa + y));
  out.t2bb = T4(ob, ob, vb, vb);
  for (int i = 0; i < ob; ++i)
    for (int j = 0; j < ob; ++j)
      for (int x = 0; x < vb; ++x)
        for (int y = 0; y < vb; ++y)
          out.t2bb(i, j, x, y) =
              u2bb(i, j, x, y) / (eb(i) + eb(j) - eb(ob + x) - eb(ob + y));
  out.t2ab = T4(oa, ob, va, vb);
  for (int i = 0; i < oa; ++i)
    for (int j = 0; j < ob; ++j)
      for (int x = 0; x < va; ++x)
        for (int y = 0; y < vb; ++y)
          out.t2ab(i, j, x, y) =
              u2ab(i, j, x, y) / (ea(i) + eb(j) - ea(oa + x) - eb(ob + y));
  return out;
}

// MP1 guess: t1 = 0, t2 from the corresponding oovv-type block / denominators.
Amps mp1_guess(const UCCIntegrals &e) {
  const int oa = e.nocca, ob = e.noccb, va = e.nvira, vb = e.nvirb;
  const Vec &ea = e.mo_energy_a, &eb = e.mo_energy_b;
  Amps a;
  a.t1a = T2(oa, va);
  a.t1a.setZero();
  a.t1b = T2(ob, vb);
  a.t1b.setZero();
  // physicist <ij||ab> for aa: (ia|jb)-(ib|ja) -> ovov.shuffle
  const T4 g_aa = e.ovov.shuffle(Sh4{0, 2, 1, 3}) - e.ovov.shuffle(Sh4{0, 2, 3, 1});
  const T4 g_bb = e.OVOV.shuffle(Sh4{0, 2, 1, 3}) - e.OVOV.shuffle(Sh4{0, 2, 3, 1});
  const T4 g_ab = e.ovOV.shuffle(Sh4{0, 2, 1, 3}); // <iJ|aB> = (ia|JB)
  a.t2aa = T4(oa, oa, va, va);
  for (int i = 0; i < oa; ++i)
    for (int j = 0; j < oa; ++j)
      for (int x = 0; x < va; ++x)
        for (int y = 0; y < va; ++y)
          a.t2aa(i, j, x, y) =
              g_aa(i, j, x, y) / (ea(i) + ea(j) - ea(oa + x) - ea(oa + y));
  a.t2bb = T4(ob, ob, vb, vb);
  for (int i = 0; i < ob; ++i)
    for (int j = 0; j < ob; ++j)
      for (int x = 0; x < vb; ++x)
        for (int y = 0; y < vb; ++y)
          a.t2bb(i, j, x, y) =
              g_bb(i, j, x, y) / (eb(i) + eb(j) - eb(ob + x) - eb(ob + y));
  a.t2ab = T4(oa, ob, va, vb);
  for (int i = 0; i < oa; ++i)
    for (int j = 0; j < ob; ++j)
      for (int x = 0; x < va; ++x)
        for (int y = 0; y < vb; ++y)
          a.t2ab(i, j, x, y) =
              g_ab(i, j, x, y) / (ea(i) + eb(j) - ea(oa + x) - eb(ob + y));
  return a;
}

// Run the spin-adapted CCSD iterations on a prepared integral set. The
// converged amplitudes are returned via `out` (for the (T) correction).
UCCSDResult run_uccsd(const UCCIntegrals &e, const UCCSDOptions &opts,
                      Amps &out) {
  const int oa = e.nocca, ob = e.noccb, va = e.nvira, vb = e.nvirb;
  occ::log::info("Unrestricted CCSD: occ (a,b)=({},{}) vir (a,b)=({},{})", oa,
                 ob, va, vb);

  occ::timing::start(occ::timing::category::ccsd);
  Amps t = mp1_guess(e);

  const Eigen::Index n1 = static_cast<Eigen::Index>(oa) * va +
                          static_cast<Eigen::Index>(ob) * vb;
  const Eigen::Index n2 = static_cast<Eigen::Index>(oa) * oa * va * va +
                          static_cast<Eigen::Index>(oa) * ob * va * vb +
                          static_cast<Eigen::Index>(ob) * ob * vb * vb;
  const Eigen::Index ntot = n1 + n2;
  auto pack = [&](const Amps &a, Mat &v) {
    Eigen::Index k = 0;
    auto put = [&](const auto &t) {
      const Eigen::Index n = t.size();
      std::copy(t.data(), t.data() + n, v.data() + k);
      k += n;
    };
    put(a.t1a);
    put(a.t1b);
    put(a.t2aa);
    put(a.t2ab);
    put(a.t2bb);
  };
  auto unpack = [&](const Mat &v, Amps &a) {
    Eigen::Index k = 0;
    auto get = [&](auto &t) {
      const Eigen::Index n = t.size();
      std::copy(v.data() + k, v.data() + k + n, t.data());
      k += n;
    };
    get(a.t1a);
    get(a.t1b);
    get(a.t2aa);
    get(a.t2ab);
    get(a.t2bb);
  };

  occ::core::diis::DIIS diis;
  UCCSDResult res;
  double e_old = uccsd_energy(t, e);
  occ::log::info("starting UCCSD iterations");
  double total_time = 0.0;
  for (int it = 0; it < opts.max_cycle; ++it) {
    const auto t0 = std::chrono::high_resolution_clock::now();
    Amps tn = update_amps(t, e);

    Mat x(ntot, 1), err(ntot, 1);
    pack(tn, x);
    Mat xold(ntot, 1);
    pack(t, xold);
    err = x - xold;
    const double rnorm = err.norm();
    diis.extrapolate(x, err);
    unpack(x, tn);
    t = tn;

    const double e_new = uccsd_energy(t, e);
    const double de = e_new - e_old;
    const double secs = std::chrono::duration<double>(
                            std::chrono::high_resolution_clock::now() - t0)
                            .count();
    total_time += secs;
    if (it == 0)
      occ::log::info("{:>4s} {: >20s} {: >12s} {: >12s}  {: >8s}", "#",
                     "E_corr (Ha)", "|dE|", "|dT|", "T (s)");
    occ::log::info("{:>4d} {:>20.12f} {:>12.5e} {:>12.5e}  {:>8.2e}", it + 1,
                   e_new, std::abs(de), rnorm, secs);
    occ::log::flush();
    res.iterations = it + 1;
    e_old = e_new;
    if (std::abs(de) < opts.tol) {
      res.converged = true;
      break;
    }
  }
  occ::timing::stop(occ::timing::category::ccsd);
  occ::log::info("UCCSD {} after {} iterations ({:.3f} s)",
                 res.converged ? "converged" : "NOT converged", res.iterations,
                 total_time);
  res.e_corr = e_old;
  if (!res.converged)
    occ::log::warn("UCCSD did not converge in {} iterations", res.iterations);
  out = t;
  return res;
}

// Build the backend integrals for the requested backend.
UCCIntegrals build_eris(const AOBasis &basis, const AOBasis *aux,
                        const MolecularOrbitals &mo, const UCCSDOptions &opts) {
  const std::string backend = opts.backend.empty() ? "exact" : opts.backend;
  if (backend == "exact")
    return u_exact_eris(basis, mo, opts.n_frozen, opts.memory_budget);
  if (!aux)
    throw std::runtime_error("spin-adapted uccsd: the '" + backend +
                             "' backend requires an auxiliary basis");
  if (backend == "df")
    return u_df_eris(basis, *aux, mo, opts.n_frozen, opts.memory_budget);
  if (backend == "thc")
    return u_thc_eris(basis, *aux, mo, opts.thc, opts.n_frozen,
                      opts.memory_budget);
  throw std::runtime_error("spin-adapted uccsd: unknown backend '" + backend +
                           "'");
}

// Run CCSD and, if requested, the (T) correction (via the spin-orbital kernel).
UCCSDResult run(const AOBasis &basis, const AOBasis *aux,
                const MolecularOrbitals &mo, const UCCSDOptions &opts) {
  Amps amps;
  const UCCIntegrals e = build_eris(basis, aux, mo, opts);
  UCCSDResult res = run_uccsd(e, opts, amps);
  if (opts.with_triples) {
    res.e_triples = uccsd_t(e, amps.t1a, amps.t1b, amps.t2aa, amps.t2ab,
                            amps.t2bb);
    occ::log::info("UCCSD(T) correction: {:.12f}", res.e_triples);
  }
  return res;
}

} // namespace

UCCSDResult uccsd(const AOBasis &basis, const MolecularOrbitals &mo,
                  const UCCSDOptions &opts) {
  return run(basis, nullptr, mo, opts);
}

UCCSDResult uccsd(const AOBasis &basis, const AOBasis &aux_basis,
                  const MolecularOrbitals &mo, const UCCSDOptions &opts) {
  return run(basis, &aux_basis, mo, opts);
}

} // namespace occ::qm::cc
