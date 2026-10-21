// Cluster-in-molecule CCSD / CCSD(T) density (Li & Piecuch): one cluster per
// Pipek-Mezey LMO, MP2 pair selection, local natural orbitals, nested clusters
// merged, all on one DF store.
#include "detail/density.h"
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/qm/correlation/df_integrals.h>
#include <occ/qm/fitting_basis.h>
#include <occ/qm/integral_engine.h>
#include <occ/qm/integral_engine_df.h>

namespace occ::qm::cc {

namespace {
using namespace detail;
using bvec = std::vector<bool>;
using ivec2 = std::vector<ivec>;

// Pipek-Mezey localisation with Mulliken charges by Jacobi sweeps (Pipek &
// Mezey, JCP 90, 4916 (1989)): rotates C's columns among themselves to maximise
// sum_k sum_A (Q^A_kk)^2
occ::Mat pipek_mezey(occ::Mat C, const occ::Mat &S, const ivec &bf_atom,
                     const int natoms) {
  const int nbf = static_cast<int>(C.rows()), n = static_cast<int>(C.cols());
  occ::Mat SC = S * C;
  vec qss(natoms), qtt(natoms), qst(natoms);
  double max_rot = 1.0;
  int sweep = 0;
  for (; sweep < 500 && max_rot > 1e-8; sweep++) {
    max_rot = 0.0;
    for (int a = 0; a < n; a++)
      for (int b = a + 1; b < n; b++) {
        std::fill(qss.begin(), qss.end(), 0.0);
        std::fill(qtt.begin(), qtt.end(), 0.0);
        std::fill(qst.begin(), qst.end(), 0.0);
        for (int mu = 0; mu < nbf; mu++) {
          const int A = bf_atom[mu];
          qss[A] += C(mu, a) * SC(mu, a);
          qtt[A] += C(mu, b) * SC(mu, b);
          qst[A] += 0.5 * (C(mu, a) * SC(mu, b) + C(mu, b) * SC(mu, a));
        }
        double x = 0.0, y = 0.0;
        for (int A = 0; A < natoms; A++) {
          const double d = qss[A] - qtt[A];
          x += qst[A] * qst[A] - 0.25 * d * d;
          y += qst[A] * d;
        }
        if (x * x + y * y < 1e-30)
          continue;
        // cos 4g = -x / r, sin 4g = y / r maximises the functional for this
        // pair
        const double g = 0.25 * std::atan2(y, -x);
        if (std::abs(g) < 1e-14)
          continue;
        max_rot = std::max(max_rot, std::abs(g));
        const double c = std::cos(g), sn = std::sin(g);
        const occ::Vec ca = C.col(a), cb = C.col(b), sa = SC.col(a),
                       sb = SC.col(b);
        C.col(a) = c * ca + sn * cb;
        C.col(b) = c * cb - sn * ca;
        SC.col(a) = c * sa + sn * sb;
        SC.col(b) = c * sb - sn * sa;
      }
  }
  if (max_rot > 1e-8)
    occ::log::warn("Pipek-Mezey: largest rotation still {:.3e} after {} sweeps",
                   max_rot, sweep);
  return C;
}

// one cluster's half-density H (H + H^T: the correlation part of the 1-RDM) in
// [cluster LMOs | virtuals], R(k, i) = <LMO k | semicanonical i>; a term
// belongs to the cluster of its distinguished occupied index (Li & Piecuch):
// the free one for oo and ov, Lambda's first for vv, the triples block's first
// through proj
occ::Mat cluster_half_density(const CcSolution &s, const occ::Mat &R,
                              const ivec &central, const bool with_triples) {
  const int o = s.t1.d[0], v = s.t1.d[1];
  const Tn Rt = as_tn(R);
  auto occ1 = [&](const Tn &x) { return einsum("ki,ia->ka", Rt, x); };
  auto occ2 = [&](const Tn &x) {
    return einsum("lj,kjab->klab", Rt, einsum("ki,ijab->kjab", Rt, x));
  };
  const Tn t1 = occ1(s.t1), l1 = occ1(s.l1), t2 = occ2(s.t2), l2 = occ2(s.l2);
  bvec is_c(o, false);
  for (int c = 0; c < static_cast<int>(central.size()); c++)
    is_c[central[c]] = true;
  const Gamma1 g = gamma1(t1, t2, l1, l2);
  Tn l1c = l1, l2c = l2;
  const size_t ovv = size_t(o) * v * v;
  for (int k = 0; k < o; k++)
    if (!is_c[k]) {
      std::fill_n(l1c.v.begin() + size_t(k) * v, v, 0.0);
      std::fill_n(l2c.v.begin() + size_t(k) * ovv, ovv, 0.0);
    }
  const Tn theta = t2 * 2.0 - T(t2, {0, 1, 3, 2});
  Tn dvv = einsum("ia,ib->ab", t1, l1c) + einsum("jica,jicb->ab", theta, l2c);
  occ::Mat doo(o, o), dvo(v, o);
  for (int k = 0; k < o; k++) {
    for (int j = 0; j < o; j++)
      doo(k, j) = g.doo.at(k, j);
    for (int a = 0; a < v; a++)
      dvo(a, k) = g.dvo.at(a, k);
  }
  if (with_triples) {
    occ::Mat goo(o, o), dvo_t(v, o);
    for (int i = 0; i < o; i++) {
      for (int j = 0; j < o; j++)
        goo(i, j) = s.tr.goo.at(i, j);
      for (int a = 0; a < v; a++)
        dvo_t(a, i) = s.tr.dvo.at(a, i);
    }
    doo -= 0.5 * R * goo * R.transpose();
    dvo += dvo_t * R.transpose();
    dvv += s.tr.gvv_c * 0.5;
  }
  occ::Mat H = occ::Mat::Zero(o + v, o + v);
  for (int c = 0; c < static_cast<int>(central.size()); c++) {
    const int k = central[c];
    H.block(k, 0, 1, o) = doo.row(k);
    for (int a = 0; a < v; a++)
      H(k, o + a) = l1.at(k, a) + dvo(a, k);
  }
  for (int a = 0; a < v; a++)
    for (int b = 0; b < v; b++)
      H(o + a, o + b) = dvv.at(a, b);
  return H;
}

// the cluster's share of E(CCSD corr), sum_jab [2 (Ia|jb) - (Ib|ja)] tau_Ij^ab
// with LMO I central
double cluster_energy(const CcSolution &s, const occ::Mat &R, const Tn &ovov,
                      const int I) {
  const int o = s.t1.d[0], v = s.t1.d[1];
  const Tn Rt = as_tn(R);
  const Tn t1 = einsum("ki,ia->ka", Rt, s.t1);
  const Tn t2 = einsum("lj,kjab->klab", Rt, einsum("ki,ijab->kjab", Rt, s.t2));
  const Tn g = einsum("lj,kajb->kalb", Rt, einsum("ki,iajb->kajb", Rt, ovov));
  double e = 0.0;
  for (int j = 0; j < o; j++)
    for (int a = 0; a < v; a++)
      for (int b = 0; b < v; b++) {
        const double tau = t2.v[((size_t(I) * o + j) * v + a) * v + b] +
                           t1.at(I, a) * t1.at(j, b);
        e += (2.0 * g.v[((size_t(I) * v + a) * o + j) * v + b] -
              g.v[((size_t(I) * v + b) * o + j) * v + a]) *
             tau;
      }
  return e;
}
} // namespace

CorrelatedDensity local_ccsd_density(const occ::qm::Wavefunction &scf,
                                     const CorrelatedDensityOptions &opts) {
  check(scf.mo.kind == occ::qm::SpinorbitalKind::Restricted &&
            scf.mo.n_alpha == scf.mo.n_beta,
        "The local CCSD density needs a closed-shell (restricted) reference");
  CorrelatedDensity result;
  auto clock = std::chrono::steady_clock::now();
  auto mark = [&clock](const char *stage) {
    const auto now = std::chrono::steady_clock::now();
    occ::log::info("local CCSD density: {} took {:.3f} s", stage,
                   std::chrono::duration<double>(now - clock).count());
    clock = now;
  };
  const auto quiet = [](const char *) {};
  const auto &basis = scf.basis;
  const occ::Mat &C = scf.mo.C;
  const int nbf = static_cast<int>(C.rows()), nmo = static_cast<int>(C.cols()),
            nocc = static_cast<int>(scf.mo.n_alpha);
  result.n_frozen = std::max(
      0, std::min(opts.n_frozen < 0 ? occ::qm::cc::num_frozen_core(basis)
                                    : opts.n_frozen,
                  nocc - 1));
  const int nf = result.n_frozen;
  const occ::qm::IntegralEngine engine(basis);
  const occ::Mat S =
      engine.one_electron_operator(occ::qm::cint::Operator::overlap);
  // the converged Fock matrix in the AO basis, exact on the MO span: F = S C e
  // C^T S
  const occ::Mat F =
      S * C * scf.mo.energies.head(nmo).asDiagonal() * C.transpose() * S;
  const occ::Mat Cocc = C.leftCols(nocc);
  // projected atomic orbitals: column mu is AO mu with every occupied orbital
  // (core included) projected out
  const occ::Mat Q =
      occ::Mat::Identity(nbf, nbf) - Cocc * (Cocc.transpose() * S);
  const auto &bf_atom = basis.bf_to_atom();
  const int natoms = static_cast<int>(basis.atoms().size());
  const occ::Mat L =
      pipek_mezey(C.middleCols(nf, nocc - nf), S, bf_atom, natoms);
  const int n = static_cast<int>(L.cols());
  const occ::Mat SL = S * L;
  occ::Mat pop =
      occ::Mat::Zero(natoms, n); // Mulliken population of LMO k on atom A
  for (int mu = 0; mu < nbf; mu++)
    for (int k = 0; k < n; k++)
      pop(bf_atom[mu], k) += L(mu, k) * SL(mu, k);
  occ::Mat centre = occ::Mat::Zero(3, n);
  for (int k = 0; k < n; k++) {
    for (int A = 0; A < natoms; A++)
      centre.col(k) +=
          pop(A, k) *
          occ::Vec3(basis.atoms()[A].x, basis.atoms()[A].y, basis.atoms()[A].z);
    centre.col(k) /= pop.col(k).sum();
  }
  mark("localisation");
  const double r_cluster = opts.cluster_radius * occ::units::ANGSTROM_TO_BOHR;
  occ::Mat Dhalf = occ::Mat::Zero(nbf, nbf);
  double e_corr = 0.0, e_t = 0.0;
  long long sum_o = 0, sum_v = 0, sum_cand = 0, sum_dom = 0;
  int max_o = 0, max_v = 0;
  std::string backend = occ::util::to_lower_copy(opts.backend), aux_name;
  const bool use_df = backend == "df" || backend == "ri";
  // one DF store for the molecule: every cluster's MP2 screening and, with the
  // df backend, its CC integrals
  const std::string fit =
      opts.aux_basis.empty()
          ? occ::qm::resolve_fitting_basis(basis.name(),
                                           occ::qm::FittingKind::Correlation)
          : opts.aux_basis;
  occ::gto::AOBasis aux = occ::gto::AOBasis::load(basis.atoms(), fit);
  aux.set_kind(basis.kind());
  occ::qm::IntegralEngineDF df_engine(basis.atoms(), basis.shells(),
                                      aux.shells());
  occ::qm::DFIntegrals df(
      df_engine,
      static_cast<size_t>(opts.max_memory_gb * 1024.0 * 1024.0 * 1024.0));
  const occ::Mat FL = F * L;
  occ::Vec fdiag(n);
  for (int k = 0; k < n; k++)
    fdiag(k) = L.col(k).dot(FL.col(k));
  mark("DF store");
  // per LMO I: MP2 pairs (I, J) over the candidates within r_cluster, in their
  // PAO domain; pairs above pair_threshold select I's LMOs, and their MP2
  // virtual density is kept with the domain's virtuals
  ivec2 sel(n), doms(n);
  std::vector<occ::Mat> Cdom(n), Dvir(n);
  for (int I = 0; I < n; I++) {
    ivec cand{I};
    for (int J = 0; J < n; J++)
      if (J != I && (centre.col(I) - centre.col(J)).norm() < r_cluster)
        cand.push_back(J);
    const int nc = static_cast<int>(cand.size());
    ivec dom;
    for (int mu = 0; mu < nbf; mu++)
      for (int c = 0; c < nc; c++)
        if (pop(bf_atom[mu], cand[c]) > opts.domain_threshold) {
          dom.push_back(mu);
          break;
        }
    occ::Mat P(nbf, dom.size());
    for (size_t k = 0; k < dom.size(); k++)
      P.col(k) = Q.col(dom[k]);
    // orthonormal virtuals from the domain's PAOs, the occupied directions
    // (overlap eigenvalue 0) dropped
    Eigen::SelfAdjointEigenSolver<occ::Mat> sp(P.transpose() * S * P);
    ivec keep;
    for (int k = 0; k < sp.eigenvalues().size(); k++)
      if (sp.eigenvalues()(k) > 1e-6)
        keep.push_back(k);
    const int nd = static_cast<int>(keep.size());
    occ::Mat Cd(nbf, nd);
    for (int k = 0; k < nd; k++)
      Cd.col(k) = P * sp.eigenvectors().col(keep[k]) /
                  std::sqrt(sp.eigenvalues()(keep[k]));
    Eigen::SelfAdjointEigenSolver<occ::Mat> fd(Cd.transpose() * F * Cd);
    const occ::Mat Cds = Cd * fd.eigenvectors();
    const occ::Vec &ed = fd.eigenvalues();
    // semicanonical MP2 pairs (I, J) with F_II + F_JJ as the occupied energies,
    // the prescreening of DLPNO
    occ::Mat Lc(nbf, nc);
    for (int k = 0; k < nc; k++)
      Lc.col(k) = L.col(cand[k]);
    const occ::Mat B = df.build_b_tilde(Lc, Cds); // row k*nd + a: (J_k a|P)
    ivec cl{I};
    occ::Mat Dv = occ::Mat::Zero(nd, nd);
    for (int k = 0; k < nc; k++) {
      const occ::Mat K =
          B.topRows(nd) * B.middleRows(static_cast<Eigen::Index>(k) * nd, nd)
                              .transpose(); //(Ia|Jb)
      occ::Mat t(nd, nd);
      double e_pair = 0.0;
      for (int b = 0; b < nd; b++)
        for (int a = 0; a < nd; a++) {
          t(a, b) = K(a, b) / (fdiag(I) + fdiag(cand[k]) - ed(a) - ed(b));
          e_pair += t(a, b) * (2.0 * K(a, b) - K(b, a));
        }
      if (k > 0 && !(std::abs(e_pair) > opts.pair_threshold))
        continue;
      if (k > 0)
        cl.push_back(cand[k]);
      // the MP2 virtual density of pairs (I, J) and (J, I): sum_c t_ac (2 t_bc
      // - t_cb), t_JI = t_IJ^T
      Dv += t * (2.0 * t - t.transpose()).transpose();
      if (k > 0)
        Dv += t.transpose() * (2.0 * t.transpose() - t).transpose();
    }
    sel[I] = cl;
    doms[I] = dom;
    Cdom[I] = Cds;
    Dvir[I] = 0.5 * (Dv + Dv.transpose());
    sum_cand += nc;
    sum_dom += nd;
  }
  mark("MP2 pair screening");
  // merging: an LMO whose selection lies inside a larger one's is one more
  // central LMO of that cluster
  ivec order(n), owner(n, -1);
  for (int I = 0; I < n; I++)
    order[I] = I;
  std::stable_sort(order.begin(), order.end(), [&](const int a, const int b) {
    return sel[a].size() > sel[b].size();
  });
  ivec2 groups;
  for (int x = 0; x < n; x++) {
    const int I = order[x];
    if (owner[I] >= 0)
      continue;
    bvec in(n, false);
    for (int k = 0; k < static_cast<int>(sel[I].size()); k++)
      in[sel[I][k]] = true;
    ivec g{I};
    owner[I] = static_cast<int>(groups.size());
    for (int y = x + 1; y < n; y++) {
      const int J = order[y];
      if (owner[J] >= 0)
        continue;
      bool inside = true;
      for (int k = 0; k < static_cast<int>(sel[J].size()) && inside; k++)
        inside = in[sel[J][k]];
      if (!inside)
        continue;
      g.push_back(J);
      owner[J] = static_cast<int>(groups.size());
    }
    groups.push_back(g);
  }
  // per cluster: the leader's LMOs, and the natural orbitals above
  // lno_threshold of the members' summed MP2 virtual densities in their joint
  // PAO domain (local natural orbitals)
  for (int gi = 0; gi < static_cast<int>(groups.size()); gi++) {
    const ivec &g = groups[gi], &cl = sel[g[0]];
    bvec use(nbf, false);
    for (int m = 0; m < static_cast<int>(g.size()); m++)
      for (int k = 0; k < static_cast<int>(doms[g[m]].size()); k++)
        use[doms[g[m]][k]] = true;
    ivec dom;
    for (int mu = 0; mu < nbf; mu++)
      if (use[mu])
        dom.push_back(mu);
    occ::Mat P(nbf, dom.size());
    for (size_t k = 0; k < dom.size(); k++)
      P.col(k) = Q.col(dom[k]);
    Eigen::SelfAdjointEigenSolver<occ::Mat> sp(P.transpose() * S * P);
    ivec keep;
    for (int k = 0; k < sp.eigenvalues().size(); k++)
      if (sp.eigenvalues()(k) > 1e-6)
        keep.push_back(k);
    const int nu = static_cast<int>(keep.size());
    occ::Mat Cu(nbf, nu);
    for (int k = 0; k < nu; k++)
      Cu.col(k) = P * sp.eigenvectors().col(keep[k]) /
                  std::sqrt(sp.eigenvalues()(keep[k]));
    const occ::Mat CuS = Cu.transpose() * S;
    occ::Mat Dv = occ::Mat::Zero(nu, nu);
    for (int m = 0; m < static_cast<int>(g.size()); m++) {
      const occ::Mat X = CuS * Cdom[g[m]];
      Dv += X * Dvir[g[m]] * X.transpose();
    }
    Eigen::SelfAdjointEigenSolver<occ::Mat> no(Dv);
    ivec lno;
    for (int k = 0; k < nu; k++)
      if (no.eigenvalues()(k) > opts.lno_threshold)
        lno.push_back(k);
    const int v = static_cast<int>(lno.size());
    occ::Mat Cv(nbf, v);
    for (int k = 0; k < v; k++)
      Cv.col(k) = Cu * no.eigenvectors().col(lno[k]);
    const int o = static_cast<int>(cl.size());
    occ::Mat Co(nbf, o);
    for (int k = 0; k < o; k++)
      Co.col(k) = L.col(cl[k]);
    // semicanonical within the cluster's occupied and within its virtual space,
    // so occ's canonical CCSD applies; F_ov vanishes because the virtuals are
    // orthogonal to every occupied orbital of the converged SCF
    Eigen::SelfAdjointEigenSolver<occ::Mat> fo(Co.transpose() * F * Co),
        fv(Cv.transpose() * F * Cv);
    const occ::Mat &R = fo.eigenvectors();
    const occ::Mat Cvs = Cv * fv.eigenvectors();
    occ::qm::MolecularOrbitals mo;
    mo.kind = occ::qm::SpinorbitalKind::Restricted;
    mo.n_alpha = mo.n_beta = o;
    mo.n_ao = nbf;
    mo.C.resize(nbf, o + v);
    mo.C << Co * R, Cvs;
    mo.energies.resize(o + v);
    mo.energies << fo.eigenvalues(), fv.eigenvalues();
    const occ::qm::cc::CCIntegrals eris =
        use_df ? occ::qm::cc::df_eris(df, mo, 0)
               : build_eris(basis, mo, 0, opts, backend, aux_name);
    ivec central(g.size());
    occ::Mat proj(o, g.size()); // the central LMOs in the semicanonical
                                // orbitals
    for (int m = 0; m < static_cast<int>(g.size()); m++) {
      central[m] =
          static_cast<int>(std::find(cl.begin(), cl.end(), g[m]) - cl.begin());
      proj.col(m) = R.row(central[m]).transpose();
    }
    const CcSolution s = solve_cc(eris, opts, quiet, &proj);
    const occ::Mat H = cluster_half_density(s, R, central, opts.triples);
    occ::Mat Ccl(nbf, o + v);
    Ccl << Co, Cvs;
    Dhalf += Ccl * H * Ccl.transpose();
    const Tn ovov = from_tensor(eris.ovov);
    for (int m = 0; m < static_cast<int>(g.size()); m++)
      e_corr += cluster_energy(s, R, ovov, central[m]);
    e_t += s.tr.e_t * g.size() /
           o; // a log figure: (T) shared evenly by the cluster's LMOs
    sum_o += o;
    sum_v += v;
    max_o = std::max(max_o, o);
    max_v = std::max(max_v, v);
  }
  mark("cluster CCSD densities");
  const occ::Mat D = 2.0 * Cocc * Cocc.transpose() + Dhalf + Dhalf.transpose();
  const occ::Mat dm = C.transpose() * S * D * S * C;
  const std::string name = opts.triples ? "local CCSD(T)" : "local CCSD";
  result.e_corr = e_corr;
  result.e_triples = opts.triples ? e_t : 0.0;
  natural_orbitals(scf, dm, name, scf.energy.total + e_corr + result.e_triples,
                   result);
  mark("density and natural orbitals");
  result.backend = use_df ? "df" : "exact";
  result.aux_basis = fit;
  result.n_lmo = n;
  result.n_clusters = static_cast<int>(groups.size());
  result.mean_candidates = double(sum_cand) / n;
  result.mean_domain = double(sum_dom) / n;
  result.mean_cluster_occ = double(sum_o) / groups.size();
  result.mean_cluster_vir = double(sum_v) / groups.size();
  result.max_cluster_occ = max_o;
  result.max_cluster_vir = max_v;
  return result;
}

} // namespace occ::qm::cc
