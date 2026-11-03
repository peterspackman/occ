// Native effective core potential integrals. See include/occ/ints/ecp.h for
// the method; the special functions and tables live in ecp_math.cpp.
#include "ecp_internal.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <numbers>
#include <occ/gto/shell.h>
#include <occ/ints/ecoeffs.h>
#include <stdexcept>

namespace occ::ints {

using namespace ecp_detail;

namespace {

constexpr double kFourPi = 4.0 * std::numbers::pi;
constexpr double kFourPiSq = kFourPi * kFourPi;

// Cartesian components of shell l, in occ's default order.
template <typename F> inline void for_each_cart(int l, F &&f) {
  int idx = 0;
  for (int i = l; i >= 0; i--)
    for (int j = l - i; j >= 0; j--)
      f(idx++, i, j, l - i - j);
}

} // namespace

// ---------------------------------------------------------------------------
// Construction from occ's basis
// ---------------------------------------------------------------------------

std::vector<EcpCentre> ecp_centres_from_basis(const gto::AOBasis &basis) {
  std::vector<EcpCentre> result;
  const auto &ecp = basis.ecp_shells();
  const auto &ecp_atom = basis.ecp_shell_to_atom();
  if (ecp.empty())
    return result;
  std::map<int, size_t> atom_to_centre;
  std::vector<std::vector<std::vector<EcpPrimitive>>> channels;
  for (size_t s = 0; s < ecp.size(); s++) {
    const auto &sh = ecp[s];
    const int atom = static_cast<int>(ecp_atom[s]);
    auto it = atom_to_centre.find(atom);
    size_t c;
    if (it == atom_to_centre.end()) {
      c = result.size();
      atom_to_centre[atom] = c;
      EcpCentre centre;
      centre.atom = atom;
      centre.origin = sh.origin;
      centre.L = 0;
      result.push_back(centre);
      channels.emplace_back();
    } else {
      c = it->second;
    }
    auto &centre = result[c];
    const int l = sh.l;
    centre.L = std::max(centre.L, l);
    if (static_cast<int>(channels[c].size()) <= l)
      channels[c].resize(l + 1);
    for (int i = 0; i < sh.num_primitives(); i++) {
      EcpPrimitive prim;
      prim.n = sh.ecp_r_exponents.size() > i ? sh.ecp_r_exponents(i) : 2;
      prim.zeta = sh.exponents(i);
      prim.coeff = sh.contraction_coefficients(i, 0);
      if (prim.coeff == 0.0)
        continue;
      channels[c][l].push_back(prim);
    }
  }
  for (size_t c = 0; c < result.size(); c++) {
    auto &centre = result[c];
    channels[c].resize(centre.L + 1);
    centre.local = channels[c][centre.L];
    centre.semilocal.assign(channels[c].begin(),
                            channels[c].begin() + centre.L);
    double zmin = std::numeric_limits<double>::max();
    for (const auto &ch : channels[c])
      for (const auto &p : ch)
        zmin = std::min(zmin, p.zeta);
    centre.min_zeta = zmin == std::numeric_limits<double>::max() ? 1.0 : zmin;
  }
  return result;
}

std::vector<EcpBasisShell> ecp_shells_from_basis(const gto::AOBasis &basis) {
  std::vector<EcpBasisShell> result;
  const auto &shell_atom = basis.shell_to_atom();
  for (size_t s = 0; s < basis.size(); s++) {
    const auto &sh = basis[s];
    EcpBasisShell e;
    e.l = sh.l;
    e.atom = static_cast<int>(shell_atom[s]);
    e.origin = sh.origin;
    // libcint folds these into its s and p integrals (CINTcommon_fac_sp).
    double fac = 1.0;
    if (sh.l == 0)
      fac = 0.28209479177387814; // 1 / (2 sqrt(pi))
    if (sh.l == 1)
      fac = 0.4886025119029199; // sqrt(3) / (2 sqrt(pi))
    for (int i = 0; i < sh.num_primitives(); i++) {
      e.exponents.push_back(sh.exponents(i));
      e.coeffs.push_back(sh.contraction_coefficients(i, 0) * fac);
    }
    result.push_back(std::move(e));
  }
  return result;
}

// ---------------------------------------------------------------------------
// Per-centre data and workspace
// ---------------------------------------------------------------------------

struct EcpEvaluator::CentreData {
  int centre{0};
  int deriv{0};
  int num_channels{0}; // L (semilocal channels)
  std::vector<Vec3> rel;         // shell origin - C
  std::vector<double> dist;      // |rel|
  std::vector<double> log_bound; // ln of a bound on the shell's reach to C
  std::vector<char> active;
  // ang[s][(dl + deriv) * num_channels + l]: ncart(l') x (2l+1)(l'+1)(l'+l+1)
  std::vector<std::vector<MatRM>> ang;
  double max_log_bound{-1e300};

  const MatRM &angular(int s, int dl, int l) const {
    return ang[s][(dl + deriv) * num_channels + l];
  }
};

struct EcpEvaluator::Workspace {
  std::vector<double> node_r, node_g;
  Mat besA, besB, X, Yb, I;
  std::vector<Mat> rad;
  std::vector<double> Ex, Ey, Ez, R, src, hermite_work;
  MatRM Rt, tmp;
  std::vector<MatRM> vout;
  std::vector<int> widx;
  std::vector<std::pair<int, int>> weights;
};

EcpEvaluator::EcpEvaluator(std::vector<EcpBasisShell> shells,
                           std::vector<EcpCentre> centres, int max_deriv)
    : m_shells(std::move(shells)), m_centres(std::move(centres)),
      m_max_deriv(max_deriv) {
  for (const auto &sh : m_shells)
    m_max_basis_l = std::max(m_max_basis_l, sh.l);
  for (const auto &c : m_centres) {
    m_max_semilocal_l = std::max(m_max_semilocal_l, c.L - 1);
    for (const auto &p : c.local)
      if (p.n < 0 || p.n > 2)
        throw std::runtime_error(
            "ECP: local channel terms with r^(n-2), n outside 0..2, are not "
            "supported");
    for (const auto &ch : c.semilocal)
      for (const auto &p : ch)
        if (p.n < 0)
          throw std::runtime_error("ECP: negative r exponent");
  }
  m_tables = std::make_shared<const Tables>(m_max_basis_l + m_max_deriv,
                                            m_max_semilocal_l);
  for (const auto &sh : m_shells) {
    double m = -1e300;
    for (double c : sh.coeffs)
      m = std::max(m, std::log(std::abs(c) + 1e-300));
    m_shell_log_cmax.push_back(m);
  }
}

EcpEvaluator::EcpEvaluator(const gto::AOBasis &basis, int max_deriv)
    : EcpEvaluator(ecp_shells_from_basis(basis), ecp_centres_from_basis(basis),
                   max_deriv) {}

EcpEvaluator::~EcpEvaluator() = default;
EcpEvaluator::EcpEvaluator(EcpEvaluator &&) noexcept = default;
EcpEvaluator &EcpEvaluator::operator=(EcpEvaluator &&) noexcept = default;

std::shared_ptr<EcpEvaluator::Workspace> EcpEvaluator::make_workspace() const {
  return std::make_shared<Workspace>();
}

namespace {

// T[(mono_index(ijk) * (Lam+1) + lambda) * (2l+1) + m]
//   = sum_mu Y_{lambda mu}(Ahat) int x^i y^j z^k Y_{lambda mu} Y_{l m} dOmega
std::vector<double> direction_table(const Tables &tab, const Vec3 &dir, int l,
                                    int dmax) {
  const int Lam = dmax + l;
  const int nm = nmono_cumulative(dmax);
  const int nml = 2 * l + 1;
  std::vector<double> T(static_cast<size_t>(nm) * (Lam + 1) * nml, 0.0);
  // a_lambda[p] = sum_mu Y_{lambda mu}(dir) y_{lambda mu, p}
  std::vector<std::vector<double>> a(Lam + 1);
  for (int lam = 0; lam <= Lam; lam++) {
    const Mat &Y = tab.Y[lam];
    std::vector<double> mono(nmono(lam));
    for_each_cart(lam, [&](int idx, int i, int j, int k) {
      mono[idx] = std::pow(dir(0), i) * std::pow(dir(1), j) *
                  std::pow(dir(2), k);
    });
    a[lam].assign(nmono(lam), 0.0);
    for (int mu = 0; mu < 2 * lam + 1; mu++) {
      double y = 0.0;
      for (int p = 0; p < nmono(lam); p++)
        y += Y(mu, p) * mono[p];
      for (int p = 0; p < nmono(lam); p++)
        a[lam][p] += y * Y(mu, p);
    }
  }
  const Mat &S = tab.S[l];
  for (int n = 0; n <= dmax; n++) {
    for_each_cart(n, [&](int, int i, int j, int k) {
      const int row = mono_index(i, j, k);
      for (int lam = (n + l) & 1; lam <= n + l; lam += 2) {
        double *t = &T[(static_cast<size_t>(row) * (Lam + 1) + lam) * nml];
        for_each_cart(lam, [&](int p, int pi, int pj, int pk) {
          const double ap = a[lam][p];
          if (ap == 0.0)
            return;
          const int col = mono_index(i + pi, j + pj, k + pk);
          for (int m = 0; m < nml; m++)
            t[m] += ap * S(m, col);
        });
      }
    });
  }
  return T;
}

// Angular table for one shell (angular momentum ls, centre `a` relative to C)
// and channel l: G(ca, (m*(ls+1) + n)*(ls+l+1) + lambda).
MatRM shell_angular(const Tables &tab, const std::vector<double> &T, int dmax,
                    int ls, int l, const Vec3 &a) {
  const int Lam = dmax + l;
  const int nml = 2 * l + 1;
  const int nN = ls + 1, nL = ls + l + 1;
  MatRM G = MatRM::Zero(ncart(ls), nml * nN * nL);
  std::vector<double> pa[3];
  for (int d = 0; d < 3; d++) {
    pa[d].resize(ls + 1);
    pa[d][0] = 1.0;
    for (int e = 1; e <= ls; e++)
      pa[d][e] = pa[d][e - 1] * (-a(d));
  }
  for_each_cart(ls, [&](int ca, int ax, int ay, int az) {
    for (int i = 0; i <= ax; i++) {
      const double ci = tab.binom[ax][i] * pa[0][ax - i];
      if (ci == 0.0)
        continue;
      for (int j = 0; j <= ay; j++) {
        const double cj = ci * tab.binom[ay][j] * pa[1][ay - j];
        if (cj == 0.0)
          continue;
        for (int k = 0; k <= az; k++) {
          const double c = cj * tab.binom[az][k] * pa[2][az - k];
          if (c == 0.0)
            continue;
          const int n = i + j + k;
          const int row = mono_index(i, j, k);
          for (int lam = (n + l) & 1; lam <= n + l; lam += 2) {
            const double *t =
                &T[(static_cast<size_t>(row) * (Lam + 1) + lam) * nml];
            for (int m = 0; m < nml; m++)
              G(ca, (m * nN + n) * nL + lam) += c * t[m];
          }
        }
      }
    }
  });
  return G;
}

} // namespace

std::shared_ptr<const EcpEvaluator::CentreData>
EcpEvaluator::prepare(int centre, int deriv) const {
  if (deriv > m_max_deriv)
    throw std::runtime_error("ECP: derivative order above evaluator maximum");
  auto cd = std::make_shared<CentreData>();
  const auto &C = m_centres[centre];
  cd->centre = centre;
  cd->deriv = deriv;
  cd->num_channels = C.L;
  const size_t ns = m_shells.size();
  cd->rel.resize(ns);
  cd->dist.resize(ns);
  cd->log_bound.resize(ns);
  cd->active.assign(ns, 0);
  cd->ang.resize(ns);
  const double zh = 0.5 * C.min_zeta;
  for (size_t s = 0; s < ns; s++) {
    const auto &sh = m_shells[s];
    cd->rel[s] = sh.origin - C.origin;
    const double d = cd->rel[s].norm();
    cd->dist[s] = d;
    double b = -1e300;
    for (size_t i = 0; i < sh.exponents.size(); i++) {
      const double a = sh.exponents[i];
      const double v = std::log(std::abs(sh.coeffs[i]) + 1e-300) +
                       deriv * std::log(std::max(a, 1.0)) -
                       a * zh * d * d / (a + zh);
      b = std::max(b, v);
    }
    b += (sh.l + deriv) * std::log(2.0 + d);
    cd->log_bound[s] = b;
    cd->max_log_bound = std::max(cd->max_log_bound, b);
  }
  double dmax_coeff = 0.0;
  for (const auto &p : C.local)
    dmax_coeff = std::max(dmax_coeff, std::abs(p.coeff));
  for (const auto &ch : C.semilocal)
    for (const auto &p : ch)
      dmax_coeff = std::max(dmax_coeff, std::abs(p.coeff));
  const double log_thr = std::log(m_settings.screening_threshold) -
                         std::log(dmax_coeff + 1e-300) - std::log(1e4);

  const auto &tab = *m_tables;
  const int dmax = m_max_basis_l + deriv;
  // Direction tables depend only on where the shell sits relative to C;
  // shells sharing an origin (an atom) share them.
  std::map<std::array<double, 3>, std::vector<std::vector<double>>>
      dir_tables;
  for (size_t s = 0; s < ns; s++) {
    if (cd->log_bound[s] + cd->max_log_bound < log_thr)
      continue;
    cd->active[s] = 1;
    if (C.L == 0)
      continue;
    const auto &sh = m_shells[s];
    const double d = cd->dist[s];
    Vec3 dir(0, 0, 1);
    if (d > 0.0)
      dir = cd->rel[s] / d;
    const std::array<double, 3> key{sh.origin(0), sh.origin(1), sh.origin(2)};
    auto it = dir_tables.find(key);
    if (it == dir_tables.end()) {
      std::vector<std::vector<double>> tabs(C.L);
      for (int l = 0; l < C.L; l++)
        if (!C.semilocal[l].empty())
          tabs[l] = direction_table(tab, dir, l, dmax);
      it = dir_tables.emplace(key, std::move(tabs)).first;
    }
    const auto *Tp = &it->second;
    cd->ang[s].resize((2 * deriv + 1) * C.L);
    for (int dl = -deriv; dl <= deriv; dl++) {
      const int ls = sh.l + dl;
      if (ls < 0)
        continue;
      for (int l = 0; l < C.L; l++) {
        if (C.semilocal[l].empty())
          continue;
        cd->ang[s][(dl + deriv) * C.L + l] =
            shell_angular(tab, (*Tp)[l], dmax, ls, l, cd->rel[s]);
      }
    }
  }
  return cd;
}

bool EcpEvaluator::significant(const CentreData &cd, int sa, int sb) const {
  return cd.active[sa] && cd.active[sb];
}

// ---------------------------------------------------------------------------
// Core integral
// ---------------------------------------------------------------------------

namespace {

// 1D Hermite expansion coefficients E(i, j, t), i<=la, j<=lb, t<=i+j, for a
// Gaussian of total exponent p centred at P: (x-A)^i (x-B)^j = sum_t E Lambda_t
void hermite_e(int la, int lb, double XPA, double XPB, double inv2p,
               double *E) {
  const int nt = la + lb + 1;
  auto at = [&](int i, int j, int t) -> double & {
    return E[(i * (lb + 1) + j) * nt + t];
  };
  std::fill(E, E + (la + 1) * (lb + 1) * nt, 0.0);
  at(0, 0, 0) = 1.0;
  for (int i = 0; i < la; i++) {
    for (int t = 0; t <= i + 1; t++) {
      double v = 0.0;
      if (t > 0)
        v += inv2p * at(i, 0, t - 1);
      if (t <= i)
        v += XPA * at(i, 0, t);
      if (t + 1 <= i)
        v += (t + 1) * at(i, 0, t + 1);
      at(i + 1, 0, t) = v;
    }
  }
  for (int i = 0; i <= la; i++) {
    for (int j = 0; j < lb; j++) {
      for (int t = 0; t <= i + j + 1; t++) {
        double v = 0.0;
        if (t > 0)
          v += inv2p * at(i, j, t - 1);
        if (t <= i + j)
          v += XPB * at(i, j, t);
        if (t + 1 <= i + j)
          v += (t + 1) * at(i, j, t + 1);
        at(i, j + 1, t) = v;
      }
    }
  }
}

// Hermite R integrals R_{tuv}, t+u+v <= L, from R^n_000 = src[n].
// Output R[(t*(L+1) + u)*(L+1) + v] (n = 0 level).
void hermite_r(int L, const Vec3 &PC, const double *src,
               std::vector<double> &work, double *R) {
  const int n1 = L + 1;
  const size_t lvl = static_cast<size_t>(n1) * n1 * n1;
  work.assign(lvl * (L + 1), 0.0);
  auto at = [&](int n, int t, int u, int v) -> double & {
    return work[n * lvl + (t * n1 + u) * n1 + v];
  };
  for (int n = L; n >= 0; n--) {
    const int maxo = L - n;
    at(n, 0, 0, 0) = src[n];
    for (int t = 0; t <= maxo; t++) {
      for (int u = 0; u + t <= maxo; u++) {
        for (int v = 0; v + u + t <= maxo; v++) {
          if (t + u + v == 0)
            continue;
          double val;
          if (t > 0) {
            val = PC(0) * at(n + 1, t - 1, u, v);
            if (t > 1)
              val += (t - 1) * at(n + 1, t - 2, u, v);
          } else if (u > 0) {
            val = PC(1) * at(n + 1, t, u - 1, v);
            if (u > 1)
              val += (u - 1) * at(n + 1, t, u - 2, v);
          } else {
            val = PC(2) * at(n + 1, t, u, v - 1);
            if (v > 1)
              val += (v - 1) * at(n + 1, t, u, v - 2);
          }
          at(n, t, u, v) = val;
        }
      }
    }
  }
  std::copy(work.begin(), work.begin() + lvl, R);
}

} // namespace

void EcpEvaluator::compute_variants(const CentreData &cd, int sa, int sb,
                                    const std::vector<Variant> &variants,
                                    Workspace &ws,
                                    std::vector<MatRM> &out) const {
  const auto &shA = m_shells[sa];
  const auto &shB = m_shells[sb];
  const auto &C = m_centres[cd.centre];
  const auto &tab = *m_tables;
  const int la = shA.l, lb = shB.l;
  const int nv = static_cast<int>(variants.size());
  out.resize(nv);
  int lamax = -1, lbmax = -1;
  ws.weights.clear();
  ws.widx.assign(nv, -1);
  for (int v = 0; v < nv; v++) {
    const int lav = la + variants[v].dla, lbv = lb + variants[v].dlb;
    if (lav < 0 || lbv < 0) {
      out[v].resize(0, 0);
      continue;
    }
    out[v].setZero(ncart(lav), ncart(lbv));
    lamax = std::max(lamax, lav);
    lbmax = std::max(lbmax, lbv);
    const std::pair<int, int> w{variants[v].ka, variants[v].kb};
    auto it = std::find(ws.weights.begin(), ws.weights.end(), w);
    if (it == ws.weights.end()) {
      ws.widx[v] = static_cast<int>(ws.weights.size());
      ws.weights.push_back(w);
    } else {
      ws.widx[v] = static_cast<int>(it - ws.weights.begin());
    }
  }
  if (lamax < 0 || !cd.active[sa] || !cd.active[sb])
    return;

  const Vec3 &a = cd.rel[sa];
  const Vec3 &b = cd.rel[sb];
  const double ra = cd.dist[sa], rb = cd.dist[sb];
  const double log_thr = std::log(m_settings.screening_threshold);
  const int nw = static_cast<int>(ws.weights.size());

  auto weight = [&](int w, double alpha, double beta) {
    const auto [ka, kb] = ws.weights[w];
    double f = 1.0;
    for (int k = 0; k < ka; k++)
      f *= alpha;
    for (int k = 0; k < kb; k++)
      f *= beta;
    return f;
  };
  double max_wlog_a = 0, max_wlog_b = 0;
  for (const auto &[ka, kb] : ws.weights) {
    max_wlog_a = std::max(max_wlog_a, double(ka));
    max_wlog_b = std::max(max_wlog_b, double(kb));
  }

  // ---------------- type 1: local channel, analytic -----------------------
  if (!C.local.empty()) {
    const int Lt = lamax + lbmax;
    const int ne = (lamax + 1) * (lbmax + 1) * (Lt + 1);
    ws.Ex.resize(ne);
    ws.Ey.resize(ne);
    ws.Ez.resize(ne);
    const double ab2 = (a - b).squaredNorm();
    const double a2 = ra * ra, b2 = rb * rb;
    auto E = [&](const std::vector<double> &Ed, int i, int j, int t) {
      return Ed[(i * (lbmax + 1) + j) * (Lt + 1) + t];
    };
    const size_t nR = static_cast<size_t>(Lt + 1) * (Lt + 1) * (Lt + 1);
    for (size_t i = 0; i < shA.exponents.size(); i++) {
      const double alpha = shA.exponents[i], ca = shA.coeffs[i];
      for (size_t j = 0; j < shB.exponents.size(); j++) {
        const double beta = shB.exponents[j], cb = shB.coeffs[j];
        for (const auto &prim : C.local) {
          const double zeta = prim.zeta;
          const double p = alpha + beta + zeta;
          const double inv_p = 1.0 / p;
          const double kexp =
              (alpha * beta * ab2 + alpha * zeta * a2 + beta * zeta * b2) *
              inv_p;
          const double base = ca * cb * prim.coeff;
          const double lb_est =
              std::log(std::abs(base) + 1e-300) - kexp +
              max_wlog_a * std::log(std::max(alpha, 1.0)) +
              max_wlog_b * std::log(std::max(beta, 1.0)) +
              lamax * std::log(1.0 + ra) + lbmax * std::log(1.0 + rb) +
              1.5 * std::log(std::numbers::pi * inv_p + 1.0) + 3.0;
          if (lb_est < log_thr)
            continue;
          const Vec3 P = (alpha * a + beta * b) * inv_p;
          const double inv2p = 0.5 * inv_p;
          hermite_e(lamax, lbmax, P(0) - a(0), P(0) - b(0), inv2p,
                    ws.Ex.data());
          hermite_e(lamax, lbmax, P(1) - a(1), P(1) - b(1), inv2p,
                    ws.Ey.data());
          hermite_e(lamax, lbmax, P(2) - a(2), P(2) - b(2), inv2p,
                    ws.Ez.data());
          const double K = std::exp(-kexp);
          if (prim.n == 2) {
            const double pref = K * std::pow(std::numbers::pi * inv_p, 1.5);
            for (int v = 0; v < nv; v++) {
              if (ws.widx[v] < 0)
                continue;
              const int lav = la + variants[v].dla, lbv = lb + variants[v].dlb;
              const double w = base * pref * weight(ws.widx[v], alpha, beta);
              auto &blk = out[v];
              for_each_cart(lav, [&](int ia, int ax, int ay, int az) {
                for_each_cart(lbv, [&](int ib, int bx, int by, int bz) {
                  blk(ia, ib) += w * E(ws.Ex, ax, bx, 0) *
                                 E(ws.Ey, ay, by, 0) * E(ws.Ez, az, bz, 0);
                });
              });
            }
          } else {
            const Vec3 PC = P; // C at the origin
            const double T = p * PC.squaredNorm();
            ws.src.resize(Lt + 1);
            double pref;
            if (prim.n == 1) {
              boys(Lt, T, ws.src.data());
              pref = K * 2.0 * std::numbers::pi * inv_p;
            } else {
              gfunc(Lt, T, ws.src.data());
              pref = K * 2.0 * std::pow(std::numbers::pi, 1.5) *
                     std::sqrt(inv_p);
            }
            double f = 1.0;
            for (int n = 0; n <= Lt; n++) {
              ws.src[n] *= f;
              f *= -2.0 * p;
            }
            ws.R.resize(nR);
            hermite_r(Lt, PC, ws.src.data(), ws.hermite_work, ws.R.data());
            auto Rv = [&](int t, int u, int v) {
              return ws.R[(t * (Lt + 1) + u) * (Lt + 1) + v];
            };
            for (int v = 0; v < nv; v++) {
              if (ws.widx[v] < 0)
                continue;
              const int lav = la + variants[v].dla, lbv = lb + variants[v].dlb;
              const double w = base * pref * weight(ws.widx[v], alpha, beta);
              auto &blk = out[v];
              for_each_cart(lav, [&](int ia, int ax, int ay, int az) {
                for_each_cart(lbv, [&](int ib, int bx, int by, int bz) {
                  double s = 0.0;
                  for (int t = 0; t <= ax + bx; t++) {
                    const double ex = E(ws.Ex, ax, bx, t);
                    if (ex == 0.0)
                      continue;
                    for (int u = 0; u <= ay + by; u++) {
                      const double exy = ex * E(ws.Ey, ay, by, u);
                      if (exy == 0.0)
                        continue;
                      for (int q = 0; q <= az + bz; q++)
                        s += exy * E(ws.Ez, az, bz, q) * Rv(t, u, q);
                    }
                  }
                  blk(ia, ib) += w * s;
                });
              });
            }
          }
        }
      }
    }
  }

  // ---------------- type 2: semilocal channels, half-numerical ------------
  const int np = static_cast<int>(tab.leg_x.size());
  for (int l = 0; l < C.L; l++) {
    const auto &prims = C.semilocal[l];
    if (prims.empty())
      continue;
    const int La = lamax + l, Lb = lbmax + l, Nmax = lamax + lbmax;
    const int nA = (Nmax + 1) * (La + 1);
    ws.rad.resize(nw);
    for (int w = 0; w < nw; w++)
      ws.rad[w].setZero(nA, Lb + 1);
    bool any = false;
    for (size_t i = 0; i < shA.exponents.size(); i++) {
      const double alpha = shA.exponents[i], ca = shA.coeffs[i];
      for (size_t j = 0; j < shB.exponents.size(); j++) {
        const double beta = shB.exponents[j], cb = shB.coeffs[j];
        ws.node_r.clear();
        ws.node_g.clear();
        for (const auto &prim : prims) {
          const double zeta = prim.zeta;
          const double p = alpha + beta + zeta;
          const double r0 = (alpha * ra + beta * rb) / p;
          const double E = -(alpha * (ra - r0) * (ra - r0) +
                             beta * (rb - r0) * (rb - r0) + zeta * r0 * r0);
          const int n = prim.n;
          const double Mhi = Nmax + n + La + Lb;
          const double Mlo = n - 2;
          auto peak = [&](double M) {
            const double disc = r0 * r0 + 2.0 * M / p;
            return disc <= 0.0 ? 0.0 : 0.5 * (r0 + std::sqrt(disc));
          };
          const double delta = std::sqrt(window_drop / p);
          const double hi = peak(Mhi) + delta;
          const double lo = std::max(0.0, peak(Mlo) - delta);
          const double bound =
              std::log(std::abs(ca * cb * prim.coeff) + 1e-300) + E +
              max_wlog_a * std::log(std::max(alpha, 1.0)) +
              max_wlog_b * std::log(std::max(beta, 1.0)) +
              lamax * std::log(ra + hi) + lbmax * std::log(rb + hi) +
              n * std::log(std::max(hi, 1.0)) + std::log(hi - lo) +
              std::log(kFourPiSq * (2 * l + 1)) + 3.0;
          if (bound < log_thr)
            continue;
          const double half = 0.5 * (hi - lo), mid = 0.5 * (hi + lo);
          for (int q = 0; q < np; q++) {
            const double r = mid + half * tab.leg_x[q];
            const double dr = r - r0;
            double g = half * tab.leg_w[q] * prim.coeff *
                       std::exp(-p * dr * dr + E);
            for (int e = 0; e < n; e++)
              g *= r;
            ws.node_r.push_back(r);
            ws.node_g.push_back(g);
          }
        }
        const int nn = static_cast<int>(ws.node_r.size());
        if (nn == 0)
          continue;
        any = true;
        ws.besA.resize(La + 1, nn);
        ws.besB.resize(Lb + 1, nn);
        const double ka = 2.0 * alpha * ra, kb = 2.0 * beta * rb;
        for (int q = 0; q < nn; q++) {
          scaled_bessel_i(La, ka * ws.node_r[q], ws.besA.col(q).data());
          scaled_bessel_i(Lb, kb * ws.node_r[q], ws.besB.col(q).data());
        }
        ws.X.resize(nn, nA);
        ws.Yb.resize(nn, Lb + 1);
        for (int q = 0; q < nn; q++) {
          const double r = ws.node_r[q];
          double rn = 1.0;
          for (int N = 0; N <= Nmax; N++) {
            for (int lam = 0; lam <= La; lam++)
              ws.X(q, N * (La + 1) + lam) = rn * ws.besA(lam, q);
            rn *= r;
          }
          for (int lam = 0; lam <= Lb; lam++)
            ws.Yb(q, lam) = ws.node_g[q] * ws.besB(lam, q);
        }
        ws.I.noalias() = ws.X.transpose() * ws.Yb;
        for (int w = 0; w < nw; w++)
          ws.rad[w] += (ca * cb * weight(w, alpha, beta)) * ws.I;
      }
    }
    if (!any)
      continue;
    const int nml = 2 * l + 1;
    for (int v = 0; v < nv; v++) {
      if (ws.widx[v] < 0)
        continue;
      const int lav = la + variants[v].dla, lbv = lb + variants[v].dlb;
      const MatRM &Ga = cd.angular(sa, variants[v].dla, l);
      const MatRM &Gb = cd.angular(sb, variants[v].dlb, l);
      const int nLa = lav + l + 1, nLb = lbv + l + 1;
      const int Ka = (lav + 1) * nLa, Kb = (lbv + 1) * nLb;
      const Mat &rad = ws.rad[ws.widx[v]];
      ws.Rt.resize(Ka, Kb);
      for (int na = 0; na <= lav; na++)
        for (int lam_a = 0; lam_a < nLa; lam_a++)
          for (int nb = 0; nb <= lbv; nb++)
            for (int lam_b = 0; lam_b < nLb; lam_b++)
              ws.Rt(na * nLa + lam_a, nb * nLb + lam_b) =
                  rad((na + nb) * (La + 1) + lam_a, lam_b);
      for (int m = 0; m < nml; m++) {
        ws.tmp.noalias() = Ga.middleCols(m * Ka, Ka) * ws.Rt;
        out[v].noalias() +=
            kFourPiSq * ws.tmp * Gb.middleCols(m * Kb, Kb).transpose();
      }
    }
  }
}

// ---------------------------------------------------------------------------
// V and derivatives
// ---------------------------------------------------------------------------

void EcpEvaluator::compute(const CentreData &cd, int sa, int sb, Workspace &ws,
                           Eigen::Ref<MatRM> out) const {
  static const std::vector<Variant> v0{{0, 0, 0, 0}};
  compute_variants(cd, sa, sb, v0, ws, ws.vout);
  out = ws.vout[0];
}

void EcpEvaluator::compute_first_derivatives(const CentreData &cd, int sa,
                                             int sb, bool want_a, bool want_b,
                                             Workspace &ws,
                                             std::array<MatRM, 6> &out) const {
  if (cd.deriv < 1)
    throw std::runtime_error("ECP: centre data prepared without derivatives");
  const int la = m_shells[sa].l, lb = m_shells[sb].l;
  std::vector<Variant> vars;
  int iap = -1, iam = -1, ibp = -1, ibm = -1;
  if (want_a) {
    iap = vars.size();
    vars.push_back({1, 0, 1, 0});
    if (la > 0) {
      iam = vars.size();
      vars.push_back({-1, 0, 0, 0});
    }
  }
  if (want_b) {
    ibp = vars.size();
    vars.push_back({0, 1, 0, 1});
    if (lb > 0) {
      ibm = vars.size();
      vars.push_back({0, -1, 0, 0});
    }
  }
  if (vars.empty())
    return;
  compute_variants(cd, sa, sb, vars, ws, ws.vout);
  const auto &V = ws.vout;
  const int na = ncart(la), nb = ncart(lb);
  if (want_a) {
    for (int d = 0; d < 3; d++)
      out[d].setZero(na, nb);
    for_each_cart(la, [&](int ia, int ax, int ay, int az) {
      const int pw[3] = {ax, ay, az};
      for (int d = 0; d < 3; d++) {
        int up[3] = {ax, ay, az};
        up[d]++;
        const int iu = ecp_cart_index(up[0], up[1], up[2]);
        out[d].row(ia) = 2.0 * V[iap].row(iu);
        if (pw[d] > 0) {
          int dn[3] = {ax, ay, az};
          dn[d]--;
          const int id = ecp_cart_index(dn[0], dn[1], dn[2]);
          out[d].row(ia) -= pw[d] * V[iam].row(id);
        }
      }
    });
  }
  if (want_b) {
    for (int d = 0; d < 3; d++)
      out[3 + d].setZero(na, nb);
    for_each_cart(lb, [&](int ib, int bx, int by, int bz) {
      const int pw[3] = {bx, by, bz};
      for (int d = 0; d < 3; d++) {
        int up[3] = {bx, by, bz};
        up[d]++;
        const int iu = ecp_cart_index(up[0], up[1], up[2]);
        out[3 + d].col(ib) = 2.0 * V[ibp].col(iu);
        if (pw[d] > 0) {
          int dn[3] = {bx, by, bz};
          dn[d]--;
          const int id = ecp_cart_index(dn[0], dn[1], dn[2]);
          out[3 + d].col(ib) -= pw[d] * V[ibm].col(id);
        }
      }
    });
  }
}

void EcpEvaluator::compute_second_derivatives(
    const CentreData &cd, int sa, int sb, bool want_aa, bool want_ab,
    bool want_bb, Workspace &ws, std::array<MatRM, 6> &aa,
    std::array<MatRM, 9> &ab, std::array<MatRM, 6> &bb) const {
  if (cd.deriv < 2)
    throw std::runtime_error(
        "ECP: centre data prepared without second derivatives");
  const int la = m_shells[sa].l, lb = m_shells[sb].l;
  std::vector<Variant> vars;
  int a_m2 = -1, a_0 = -1, a_p2 = -1, b_m2 = -1, b_0 = -1, b_p2 = -1;
  int mm = -1, mp = -1, pm = -1, pp = -1;
  auto add = [&](Variant v) {
    vars.push_back(v);
    return static_cast<int>(vars.size()) - 1;
  };
  if (want_aa) {
    a_p2 = add({2, 0, 2, 0});
    a_0 = add({0, 0, 1, 0});
    if (la >= 2)
      a_m2 = add({-2, 0, 0, 0});
  }
  if (want_bb) {
    b_p2 = add({0, 2, 0, 2});
    b_0 = add({0, 0, 0, 1});
    if (lb >= 2)
      b_m2 = add({0, -2, 0, 0});
  }
  if (want_ab) {
    pp = add({1, 1, 1, 1});
    if (lb > 0)
      pm = add({1, -1, 1, 0});
    if (la > 0)
      mp = add({-1, 1, 0, 1});
    if (la > 0 && lb > 0)
      mm = add({-1, -1, 0, 0});
  }
  if (vars.empty())
    return;
  compute_variants(cd, sa, sb, vars, ws, ws.vout);
  const auto &V = ws.vout;
  const int na = ncart(la), nb = ncart(lb);

  // one-sided second derivative assembly; `side` 0 = rows (A), 1 = cols (B)
  auto second = [&](int lside, int ip2, int i0, int im2, int side,
                    std::array<MatRM, 6> &res) {
    for (auto &m : res)
      m.setZero(na, nb);
    for_each_cart(lside, [&](int ic, int x, int y, int z) {
      const int pw[3] = {x, y, z};
      for (int i = 0; i < 3; i++) {
        for (int j = i; j < 3; j++) {
          MatRM &dst = res[ecp_sym_index(i, j)];
          auto acc = [&](int vi, int shift_i, int shift_j, double f) {
            int e[3] = {x, y, z};
            e[i] += shift_i;
            e[j] += shift_j;
            if (e[0] < 0 || e[1] < 0 || e[2] < 0 || f == 0.0)
              return;
            const int idx = ecp_cart_index(e[0], e[1], e[2]);
            if (side == 0)
              dst.row(ic) += f * V[vi].row(idx);
            else
              dst.col(ic) += f * V[vi].col(idx);
          };
          if (i == j) {
            acc(ip2, 1, 1, 4.0);
            acc(i0, 0, 0, -2.0 * (2 * pw[i] + 1));
            if (pw[i] >= 2)
              acc(im2, -1, -1, pw[i] * (pw[i] - 1.0));
          } else {
            acc(ip2, 1, 1, 4.0);
            if (pw[i] > 0)
              acc(i0, -1, 1, -2.0 * pw[i]);
            if (pw[j] > 0)
              acc(i0, 1, -1, -2.0 * pw[j]);
            if (pw[i] > 0 && pw[j] > 0)
              acc(im2, -1, -1, double(pw[i]) * pw[j]);
          }
        }
      }
    });
  };
  if (want_aa)
    second(la, a_p2, a_0, a_m2, 0, aa);
  if (want_bb)
    second(lb, b_p2, b_0, b_m2, 1, bb);
  if (want_ab) {
    for (auto &m : ab)
      m.setZero(na, nb);
    for_each_cart(la, [&](int ia, int ax, int ay, int az) {
      const int pa[3] = {ax, ay, az};
      for_each_cart(lb, [&](int ib, int bx, int by, int bz) {
        const int pb[3] = {bx, by, bz};
        for (int i = 0; i < 3; i++) {
          int au[3] = {ax, ay, az}, ad[3] = {ax, ay, az};
          au[i]++;
          ad[i]--;
          for (int j = 0; j < 3; j++) {
            int bu[3] = {bx, by, bz}, bd[3] = {bx, by, bz};
            bu[j]++;
            bd[j]--;
            double v = 4.0 * V[pp](ecp_cart_index(au[0], au[1], au[2]),
                                   ecp_cart_index(bu[0], bu[1], bu[2]));
            if (pb[j] > 0)
              v -= 2.0 * pb[j] *
                   V[pm](ecp_cart_index(au[0], au[1], au[2]),
                         ecp_cart_index(bd[0], bd[1], bd[2]));
            if (pa[i] > 0)
              v -= 2.0 * pa[i] *
                   V[mp](ecp_cart_index(ad[0], ad[1], ad[2]),
                         ecp_cart_index(bu[0], bu[1], bu[2]));
            if (pa[i] > 0 && pb[j] > 0)
              v += double(pa[i]) * pb[j] *
                   V[mm](ecp_cart_index(ad[0], ad[1], ad[2]),
                         ecp_cart_index(bd[0], bd[1], bd[2]));
            ab[3 * i + j](ia, ib) = v;
          }
        }
      });
    });
  }
}

MatRM EcpEvaluator::shell_pair(int centre, int sa, int sb) const {
  auto cd = prepare(centre, 0);
  auto ws = make_workspace();
  MatRM out(ncart(m_shells[sa].l), ncart(m_shells[sb].l));
  compute(*cd, sa, sb, *ws, out);
  return out;
}

std::array<MatRM, 9> EcpEvaluator::shell_pair_first_derivatives(int centre,
                                                                int sa,
                                                                int sb) const {
  auto cd = prepare(centre, 1);
  auto ws = make_workspace();
  std::array<MatRM, 6> d;
  compute_first_derivatives(*cd, sa, sb, true, true, *ws, d);
  std::array<MatRM, 9> out;
  for (int i = 0; i < 6; i++)
    out[i] = d[i];
  for (int i = 0; i < 3; i++)
    out[6 + i] = -(d[i] + d[3 + i]);
  return out;
}

void EcpEvaluator::shell_pair_second_derivatives(
    int centre, int sa, int sb, std::array<MatRM, 6> &aa,
    std::array<MatRM, 9> &ab, std::array<MatRM, 6> &bb) const {
  auto cd = prepare(centre, 2);
  auto ws = make_workspace();
  compute_second_derivatives(*cd, sa, sb, true, true, true, *ws, aa, ab, bb);
}

} // namespace occ::ints
