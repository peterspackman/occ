// XC contributions to the analytic DFT Hessian.
//
// compute_xc_derivative_fock: dV_xc/dR for every nuclear coordinate at fixed
// AO density P (restricted, LDA/GGA/meta-GGA), with the full grid response.
// With
//
//   V = K + K^T,  K = phi^T (w o A),  A = 1/2 v_rho phi + 2 v_sigma grad rho . grad phi
//
// (the occ form of V_xc; a meta-GGA adds the symmetric
// 1/2 sum_i d_i phi^T (w v_tau d_i phi), with tau = 1/2 sum_i d_i phi P d_i phi),
// moving atom B in direction x changes
//
//  - the basis functions on B in K directly           (explicit term),
//  - rho and grad rho through those functions          (response term, the
//    fxc kernel applied to rho^B, grad rho^B),
//  - the partition weights                             (weight term),
//  - and, for points of parent atom B, the points themselves; by
//    translational invariance that is minus the sum of the first two terms
//    over all atoms, for that point group.

#include <Eigen/Eigenvalues>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/dft/dft.h>
#include <occ/dft/dft_kernels.h>
#include <occ/gto/density.h>
#include <occ/numint/grid_utils.h>
#include <occ/qm/cphf.h>
#include "detail/xc_kernel.h"

namespace occ::dft {

namespace {

using occ::gto::GTOValues;
using Functionals = std::vector<DensityFunctional>;

// Restricted blocks: the total density P on a batch of points, the
// functional's derivatives there, and X = phi P, Xd_i = d_i phi P.
template <int deriv> struct DensityBlock {
  Mat rho;
  bool negligible{true};
  DensityFunctional::SecondDerivatives f;
  Vec w; ///< quadrature weights, zero where the density is negligible
  Array vr, frr, vs, frs, fss, gr[3], vt, frt, fst, ftt;
  Mat X, Xd[3];

  DensityBlock(const GTOValues &gto, const Mat &P, const Functionals &funcs,
               const Vec &w_in, double density_threshold,
               Vec *exc = nullptr) {
    rho.resize(gto.phi.rows(), occ::density::num_components(deriv));
    occ::density::evaluate_density<deriv, SpinorbitalKind::Restricted>(P, gto,
                                                                       rho);
    if (rho.col(0).maxCoeff() < density_threshold || funcs.empty())
      return;
    negligible = false;
    f = detail::second_derivatives<SpinorbitalKind::Restricted, deriv>(
        funcs, rho, exc);
    w = detail::screened_weights(w_in, rho.col(0).array(), density_threshold)
            .matrix();
    vr = f.vrho.col(0).array();
    frr = f.v2rho2.col(0).array();
    X = gto.phi * P;
    if constexpr (deriv >= 1) {
      vs = f.vsigma.col(0).array();
      frs = f.v2rhosigma.col(0).array();
      fss = f.v2sigma2.col(0).array();
      for (int i = 0; i < 3; i++) {
        gr[i] = rho.col(1 + i).array();
        Xd[i] = gto.first(i) * P;
      }
    }
    if constexpr (deriv == 2) {
      vt = f.vtau.col(0).array();
      frt = f.v2rhotau.col(0).array();
      fst = f.v2sigmatau.col(0).array();
      ftt = f.v2tau2.col(0).array();
    }
  }
};

// Unrestricted blocks: every term is the restricted expression per spin
// channel, with the spin density P^s = C_o^s C_o^s^T in place of the total
// density, (v_rho^s, V^s) in place of (v_rho, 2 v_sigma grad rho), and the
// spin kernel (detail::SpinKernel) wherever a density derivative feeds back
// into the potential. Co holds the channels' occupied coefficients side by
// side (na alpha columns, then beta) on the block's functions.
//
// Spin densities of one block: X^s = phi P^s, Xd^s_i = d_i phi P^s and rho
// (alpha rows over beta rows), with the functional's derivatives there.
template <int deriv> struct SpinBlock {
  Mat C[2], P[2], X[2], Xd[2][3];
  Mat rho;
  Array total;
  bool negligible{true};
  DensityFunctional::SecondDerivatives f;
  Array w; ///< quadrature weights, zero where the density is negligible

  SpinBlock(const GTOValues &gto, const Mat &Co, Eigen::Index na,
            const Functionals &funcs, const Vec &w_in,
            double density_threshold, Vec *exc = nullptr) {
    const Eigen::Index n = gto.phi.rows();
    rho.resize(2 * n, occ::density::num_components(deriv));
    C[0] = Co.leftCols(na);
    C[1] = Co.rightCols(Co.cols() - na);
    for (int s = 0; s < 2; s++) {
      P[s] = C[s] * C[s].transpose();
      X[s] = gto.phi * P[s];
      rho.col(0).segment(s * n, n) =
          (gto.phi.array() * X[s].array()).rowwise().sum();
      if constexpr (deriv == 1)
        for (int i = 0; i < 3; i++) {
          Xd[s][i] = gto.first(i) * P[s];
          rho.col(1 + i).segment(s * n, n) =
              2.0 * (gto.first(i).array() * X[s].array()).rowwise().sum();
        }
    }
    total = rho.col(0).head(n).array() + rho.col(0).tail(n).array();
    if (total.maxCoeff() < density_threshold || funcs.empty())
      return;
    negligible = false;
    f = detail::second_derivatives<SpinorbitalKind::Unrestricted, deriv>(
        funcs, rho, exc);
    w = detail::screened_weights(w_in, total, density_threshold);
  }
};

// Whether any of an atom's functions (the local functions atom_fns[l]) has
// a derivative, to the order the explicit terms need, of at least
// `threshold` over the batch; the others contribute nothing above it.
std::vector<bool> active_atoms(const GTOValues &gto,
                               const std::vector<std::vector<int>> &atom_fns,
                               bool second, double threshold) {
  std::vector<bool> active(atom_fns.size(), false);
  for (size_t l = 0; l < atom_fns.size(); l++)
    for (int mu : atom_fns[l]) {
      double m = 0.0;
      for (int x = 0; x < 3; x++)
        m = std::max(m, gto.first(x).col(mu).cwiseAbs().maxCoeff());
      if (second)
        for (int i = 0; i < 3; i++)
          for (int j = i; j < 3; j++)
            m = std::max(m, gto.second(i, j).col(mu).cwiseAbs().maxCoeff());
      if (m >= threshold) {
        active[l] = true;
        break;
      }
    }
  return active;
}

// The basis functions significant over a chunk of points with one parent
// atom (every function when `extents` is empty), and the atoms they sit on:
// local atom 0 is the parent, the others follow in shell order.
struct LocalBasis {
  std::vector<int> shells;   ///< significant shells
  std::vector<int> rows;     ///< global index of each local function
  std::vector<int> atoms;    ///< global index of each local atom
  std::vector<int> fn_atom;  ///< local atom of each local function
  std::vector<std::vector<int>> atom_fns; ///< local functions of each local atom

  LocalBasis(const occ::gto::AOBasis &basis, const Vec &extents,
             Eigen::Ref<const Mat3N> pts, int parent, int natom)
      : atoms{parent}, atom_fns(1) {
    const auto &shell_list = basis.shells();
    const auto &first_bf = basis.first_bf();
    const auto &shell_atom = basis.shell_to_atom();
    const Vec3 centre = pts.rowwise().mean();
    const double radius = (pts.colwise() - centre).colwise().norm().maxCoeff();
    std::vector<int> local_of(natom, -1);
    local_of[parent] = 0;
    for (size_t sh = 0; sh < shell_list.size(); sh++) {
      if (extents.size() > 0 &&
          (shell_list[sh].origin - centre).norm() - extents(sh) >= radius)
        continue;
      shells.push_back(static_cast<int>(sh));
      const int A = shell_atom[sh];
      if (local_of[A] < 0) {
        local_of[A] = static_cast<int>(atoms.size());
        atoms.push_back(A);
        atom_fns.emplace_back();
      }
      for (size_t k = 0; k < shell_list[sh].size(); k++) {
        atom_fns[local_of[A]].push_back(static_cast<int>(rows.size()));
        fn_atom.push_back(local_of[A]);
        rows.push_back(first_bf[sh] + k);
      }
    }
  }

  /// The rows of M on the local functions.
  Mat rows_of(const Mat &M) const {
    Mat out(rows.size(), M.cols());
    for (size_t i = 0; i < rows.size(); i++)
      out.row(i) = M.row(rows[i]);
    return out;
  }
  /// The block of M on the local functions.
  Mat block_of(const Mat &M) const {
    const Eigen::Index nl = rows.size();
    Mat out(nl, nl);
    for (Eigen::Index i = 0; i < nl; i++)
      for (Eigen::Index j = 0; j < nl; j++)
        out(i, j) = M(rows[i], rows[j]);
    return out;
  }
};

// Grid points in chunks of at most 128 consecutive points of one atom's grid,
// so each chunk has one parent.
struct AtomChunk {
  size_t start, count;
  int atom;
};

std::vector<AtomChunk>
atom_chunks(const occ::numint::MolecularGridPoints &grid) {
  constexpr size_t block = 128;
  const auto &blocks = grid.atom_blocks();
  std::vector<AtomChunk> chunks;
  for (int a = 0; a < static_cast<int>(blocks.size()); a++)
    for (size_t s = 0; s < blocks[a].second; s += block)
      chunks.push_back({blocks[a].first + s,
                        std::min(block, blocks[a].second - s), a});
  return chunks;
}

template <int deriv>
void xc_derivative_fock_block(
    const occ::gto::AOBasis &basis, const Mat &P, const Functionals &funcs,
    Eigen::Ref<const Mat3N> pts, const Vec &w_in, int parent,
    double density_threshold, const occ::numint::BeckeWeightDerivative &dwdr,
    const std::vector<std::vector<int>> &atom_bfs, std::vector<Mat> &out) {
  const Eigen::Index n = pts.cols();
  const int nbf = static_cast<int>(basis.nbf());
  const int natom = static_cast<int>(atom_bfs.size());
  // Basis derivatives one order above the functional's: LDA needs grad phi,
  // GGA and meta-GGA (tau holds first derivatives only) the second ones too.
  const auto gto = occ::gto::evaluate_basis(basis, pts, std::min(deriv, 1) + 1);
  const DensityBlock<deriv> b(gto, P, funcs, w_in, density_threshold);
  if (b.negligible)
    return;
  const Vec &w = b.w;
  const Array &vr = b.vr, &frr = b.frr, &vs = b.vs, &frs = b.frs,
              &fss = b.fss, *gr = b.gr, &vt = b.vt, &frt = b.frt,
              &fst = b.fst, &ftt = b.ftt;
  const Mat &X = b.X, *Xd = b.Xd;

  const Mat &phi = gto.phi;
  const Mat *dphi[3] = {&gto.phi_x, &gto.phi_y, &gto.phi_z};

  // tau part of K: sum_i d_i phi^T (c d_i phi)
  auto add_tau_k = [&](const Array &c, Mat &K) {
    for (int i = 0; i < 3; i++)
      K.noalias() +=
          dphi[i]->transpose() * (dphi[i]->array().colwise() * c).matrix();
  };
  // Unweighted V integrand A (n x nbf) and its weighted form.
  Mat A = 0.5 * (phi.array().colwise() * vr).matrix();
  if constexpr (deriv >= 1) {
    for (int i = 0; i < 3; i++)
      A.array() += (dphi[i]->array().colwise() * (2.0 * vs * gr[i]));
  }

  const Mat dW = dwdr.batch(pts, w, parent);

  // Point motion accumulators (per direction): sum over atoms of the
  // explicit and response contributions, to be subtracted at the parent.
  Mat tsum[3], Ksum[3];
  Array ctsum[3];
  for (int x = 0; x < 3; x++) {
    tsum[x] = Mat::Zero(n, nbf);
    Ksum[x] = Mat::Zero(nbf, nbf);
    if constexpr (deriv == 2)
      ctsum[x] = Array::Zero(n);
  }

  for (int B = 0; B < natom; B++) {
    const auto &bfs = atom_bfs[B];
    for (int x = 0; x < 3; x++) {
      Mat t = Mat::Zero(n, nbf);
      Mat Kx = Mat::Zero(nbf, nbf);
      // coefficient of the tau-type term d_i phi^T (ct d_i phi) in K
      Array ct;
      if constexpr (deriv == 2)
        ct = Array::Zero(n);
      if (!bfs.empty()) {
        // rho^B, grad rho^B, tau^B from the functions on B (P fixed).
        Array rB = Array::Zero(n), tB = Array::Zero(n);
        Array grB[3] = {Array::Zero(n), Array::Zero(n), Array::Zero(n)};
        for (int mu : bfs) {
          const Array y = dphi[x]->col(mu).array();
          rB -= 2.0 * y * X.col(mu).array();
          if constexpr (deriv >= 1) {
            for (int i = 0; i < 3; i++)
              grB[i] -= 2.0 * (gto.second(i, x).col(mu).array() * X.col(mu).array() +
                               y * Xd[i].col(mu).array());
          }
          if constexpr (deriv == 2) {
            for (int i = 0; i < 3; i++)
              tB -= gto.second(i, x).col(mu).array() * Xd[i].col(mu).array();
          }
        }
        // response through fxc
        if constexpr (deriv == 0) {
          t = (phi.array().colwise() * (0.5 * w.array() * frr * rB)).matrix();
        } else {
          const Array sB = 2.0 * (gr[0] * grB[0] + gr[1] * grB[1] + gr[2] * grB[2]);
          Array dvr = w.array() * (frr * rB + frs * sB);
          Array dvs = w.array() * (frs * rB + fss * sB);
          if constexpr (deriv == 2) {
            dvr += w.array() * frt * tB;
            dvs += w.array() * fst * tB;
            ct = 0.25 * w.array() * (frt * rB + fst * sB + ftt * tB);
            ctsum[x] += ct;
          }
          t = (phi.array().colwise() * (0.5 * dvr)).matrix();
          for (int i = 0; i < 3; i++)
            t.array() += dphi[i]->array().colwise() *
                         (2.0 * (dvs * gr[i] + w.array() * vs * grB[i]));
        }
        // explicit: -[Y^T (w o A) + phi^T (w o (1/2 vr Y + 2 vs sum_i gr_i Y_i))]
        Mat wA = (A.array().colwise() * w.array()).matrix();
        for (int mu : bfs) {
          const Vec y = dphi[x]->col(mu);
          Kx.row(mu) -= (y.transpose() * wA);
          Vec c = (0.5 * w.array() * vr * dphi[x]->col(mu).array()).matrix();
          if constexpr (deriv >= 1) {
            for (int i = 0; i < 3; i++)
              c.array() += 2.0 * w.array() * vs * gr[i] *
                           gto.second(i, x).col(mu).array();
          }
          Kx.col(mu) -= phi.transpose() * c;
          // tau term: its row and column derivatives, both in the row, as
          // K + K^T symmetrises
          if constexpr (deriv == 2) {
            for (int i = 0; i < 3; i++)
              Kx.row(mu).noalias() -=
                  (0.5 * w.array() * vt * gto.second(i, x).col(mu).array())
                      .matrix()
                      .transpose() *
                  (*dphi[i]);
          }
        }
        tsum[x] += t;
        Ksum[x] += Kx;
      }
      // weight derivative
      const Array dwx = dW.col(3 * B + x).array();
      Mat tw = (A.array().colwise() * dwx).matrix();
      Mat K = phi.transpose() * (t + tw) + Kx;
      if constexpr (deriv == 2)
        add_tau_k(ct + 0.25 * dwx * vt, K);
      out[3 * B + x] += K + K.transpose();
    }
  }
  // points move with the parent
  for (int x = 0; x < 3; x++) {
    Mat K = phi.transpose() * tsum[x] + Ksum[x];
    if constexpr (deriv == 2)
      add_tau_k(ctsum[x], K);
    out[3 * parent + x] -= K + K.transpose();
  }
}

// Skeleton XC Hessian contribution of one chunk of points with one parent
// atom: the basis-only Hessian of sum_g w_g f(rho, sigma, tau), transformed
// for the points moving with the parent, plus the first-order weight cross
// terms. Adds into H (3N x 3N) and stores the energy density per point for
// the second-order weight term.
//
// The chunk works on the local basis (gto, P, Co restricted to its
// functions) and in the coordinates of the atoms those functions sit on.
// The off-diagonal atom-pair term 2 sum_{mu in D, nu in E} P_mn M_mn is
// evaluated through P = 2 Co Co^T as 4 sum_g sum_i (alpha_D delta_E +
// delta_D beta_E), avoiding nbf x nbf products.
template <int deriv>
void xc_hessian_block(const GTOValues &gto, const Mat &P, const Mat &Co,
                      const Functionals &funcs, Eigen::Ref<const Mat3N> pts,
                      const Vec &w_in, const LocalBasis &lb, int natom,
                      double density_threshold,
                      const occ::numint::BeckeWeightDerivative &dwdr, Mat &H,
                      double *edens) {
  const Eigen::Index n = pts.cols();
  const int n3 = 3 * natom;
  const std::vector<int> &lat = lb.fn_atom, &latoms = lb.atoms;
  const int nal = static_cast<int>(latoms.size());
  const int n3l = 3 * nal;
  const Eigen::Index nl = gto.phi.cols();
  const Eigen::Index o = Co.cols();
  const int parent = latoms[0];
  for (Eigen::Index g = 0; g < n; g++)
    edens[g] = 0.0;
  Vec exc;
  const DensityBlock<deriv> b(gto, P, funcs, w_in, density_threshold, &exc);
  if (b.negligible)
    return;
  const Vec &w = b.w;
  for (Eigen::Index g = 0; g < n; g++)
    edens[g] = (w(g) == 0.0) ? 0.0 : exc(g) * b.rho(g, 0);
  const Array &vr = b.vr, &frr = b.frr, &vs = b.vs, &frs = b.frs,
              &fss = b.fss, *gr = b.gr, &vt = b.vt, &frt = b.frt,
              &fst = b.fst, &ftt = b.ftt;
  const Mat &X = b.X, *Xd = b.Xd;

  const Mat *d1[3] = {&gto.phi_x, &gto.phi_y, &gto.phi_z};
  const Mat *d2m[6] = {&gto.phi_xx, &gto.phi_xy, &gto.phi_xz,
                       &gto.phi_yy, &gto.phi_yz, &gto.phi_zz};
  static const int i2[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
  const Array W = w.array();

  // Sum the columns of an n x nl matrix by local atom into column 3 l + a.
  auto by_atom = [&](const Mat &Mfn, Mat &out, int a, double scale) {
    for (Eigen::Index mu = 0; mu < nl; mu++)
      out.col(3 * lat[mu] + a) += scale * Mfn.col(mu);
  };

  // First derivatives (basis only, point fixed), local columns 3 l + a:
  // rho (R), grad rho (G), sigma (S) and tau (Tt).
  Mat R = Mat::Zero(n, n3l), S = Mat::Zero(n, n3l), G[3], Tt;
  for (int i = 0; i < 3; i++)
    G[i] = Mat::Zero(n, n3l);
  if constexpr (deriv == 2)
    Tt = Mat::Zero(n, n3l);
  for (int a = 0; a < 3; a++) {
    by_atom(d1[a]->cwiseProduct(X), R, a, -2.0);
    if constexpr (deriv >= 1) {
      for (int i = 0; i < 3; i++)
        by_atom(gto.second(i, a).cwiseProduct(X) + d1[a]->cwiseProduct(Xd[i]), G[i],
                a, -2.0);
    }
    if constexpr (deriv == 2) {
      for (int i = 0; i < 3; i++)
        by_atom(gto.second(i, a).cwiseProduct(Xd[i]), Tt, a, -1.0);
    }
  }
  if constexpr (deriv >= 1) {
    for (int i = 0; i < 3; i++)
      S += 2.0 * (G[i].array().colwise() * gr[i]).matrix();
  }

  // Basis-only Hessian in local coordinates.
  Mat Hb = R.transpose() * (R.array().colwise() * (W * frr)).matrix();
  if constexpr (deriv >= 1) {
    const Mat RS = R.transpose() * (S.array().colwise() * (W * frs)).matrix();
    Hb += RS + RS.transpose();
    Hb += S.transpose() * (S.array().colwise() * (W * fss)).matrix();
    for (int i = 0; i < 3; i++)
      Hb += G[i].transpose() *
            (G[i].array().colwise() * (2.0 * W * vs)).matrix();
  }
  if constexpr (deriv == 2) {
    const Mat RST =
        (R.array().colwise() * (W * frt)).matrix().transpose() * Tt +
        (S.array().colwise() * (W * fst)).matrix().transpose() * Tt;
    Hb += RST + RST.transpose();
    Hb += Tt.transpose() * (Tt.array().colwise() * (W * ftt)).matrix();
  }

  // Off-diagonal function pairs: per local atom, occupied projections of
  // A_a, d_a and B_a, as (n o) x nal matrices.
  const auto &atom_fns = lb.atom_fns;
  auto project = [&](const Mat &Mfn) {
    Mat out(n * o, nal);
    for (int l = 0; l < nal; l++) {
      Mat acc = Mat::Zero(n, o);
      for (int mu : atom_fns[l])
        acc.noalias() += Mfn.col(mu) * Co.row(mu);
      out.col(l) = Eigen::Map<const Vec>(acc.data(), n * o);
    }
    return out;
  };
  Mat alpha[3], delta[3], beta[3];
  for (int a = 0; a < 3; a++) {
    Mat Aa = (d1[a]->array().colwise() * (W * vr)).matrix();
    if constexpr (deriv >= 1) {
      Mat Bb = Mat::Zero(n, nl);
      for (int i = 0; i < 3; i++) {
        const Array c = 2.0 * W * vs * gr[i];
        Aa.array() += gto.second(i, a).array().colwise() * c;
        Bb.array() += gto.second(i, a).array().colwise() * c;
      }
      beta[a] = project(Bb);
    }
    alpha[a] = project(Aa);
    delta[a] = project(*d1[a]);
  }
  // tau: 2 sum_i sum_g proj(w v_tau d_i d_a phi)_D . proj(d_i d_b phi)_E,
  // the weighting applied per point to the projections (rows g + n i).
  Mat eta[6], eta_w[6];
  if constexpr (deriv == 2) {
    Array wvt(n * o);
    for (Eigen::Index i = 0; i < o; i++)
      wvt.segment(n * i, n) = W * vt;
    for (int k = 0; k < 6; k++) {
      eta[k] = project(*d2m[k]);
      eta_w[k] = (eta[k].array().colwise() * wvt).matrix();
    }
  }
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++) {
      Mat blocks = 4.0 * alpha[a].transpose() * delta[b];
      if constexpr (deriv >= 1)
        blocks.noalias() += 4.0 * delta[a].transpose() * beta[b];
      if constexpr (deriv == 2)
        for (int i = 0; i < 3; i++)
          blocks.noalias() +=
              2.0 * eta_w[i2[i][a]].transpose() * eta[i2[i][b]];
      for (int D = 0; D < nal; D++)
        for (int E = 0; E < nal; E++)
          Hb(3 * D + a, 3 * E + b) += blocks(D, E);
    }
  // Both derivatives on the same function (D = E):
  // 2 sum_{mu in D} sum_g [w vr d_a d_b phi X + 2 w vs sum_i gr_i
  //                         (d_i d_a d_b phi X + d_a d_b phi X_i)]
  for (int a = 0; a < 3; a++) {
    for (int b = 0; b < 3; b++) {
      Mat c = gto.second(a, b).cwiseProduct(X).array().colwise() * (W * vr);
      if constexpr (deriv >= 1) {
        for (int i = 0; i < 3; i++)
          c.array() += (gto.third(i, a, b).cwiseProduct(X) +
                        gto.second(a, b).cwiseProduct(Xd[i]))
                           .array()
                           .colwise() *
                       (2.0 * W * vs * gr[i]);
      }
      if constexpr (deriv == 2) {
        for (int i = 0; i < 3; i++)
          c.array() += gto.third(i, a, b).cwiseProduct(Xd[i]).array().colwise() *
                       (0.5 * W * vt);
      }
      const Vec colsum = c.colwise().sum().transpose();
      for (Eigen::Index mu = 0; mu < nl; mu++)
        Hb(3 * lat[mu] + a, 3 * lat[mu] + b) += 2.0 * colsum(mu);
    }
  }

  // Local to global coordinates, with the points moving with the parent:
  // d_X = sum_E T_XE d_E (per direction).
  Mat T = Mat::Zero(n3, n3l);
  for (int l = 0; l < nal; l++)
    for (int x = 0; x < 3; x++) {
      T(3 * latoms[l] + x, 3 * l + x) += 1.0;
      T(3 * parent + x, 3 * l + x) -= 1.0;
    }
  H += T * Hb * T.transpose();

  // First-order weight cross terms: dw_X f_Y + dw_Y f_X with f the total
  // first derivative of the energy density.
  Mat Fb = (R.array().colwise() * vr).matrix();
  if constexpr (deriv >= 1)
    Fb += (S.array().colwise() * vs).matrix();
  if constexpr (deriv == 2)
    Fb += (Tt.array().colwise() * vt).matrix();
  const Mat Ft = Fb * T.transpose();
  const Mat cross = dwdr.batch(pts, w, parent).transpose() * Ft;
  H += cross + cross.transpose();
}

// dV_xc/dR contracted with C_o for one chunk of points with one parent, on
// the local basis (gto, P, Co restricted to its functions). Atoms carrying
// significant functions get the explicit and kernel-response terms as local
// nl x nl matrices; the partition-weight term, which every atom has, is
// contracted with C_o directly:
// (phi^T (dw A) + (dw A)^T phi) Co = phi^T (dw (A Co)) + A^T (dw (phi Co)).
template <int deriv>
void xc_derivative_fock_occ_block(
    const GTOValues &gto, const Mat &P, const Mat &Co, const Functionals &funcs,
    Eigen::Ref<const Mat3N> pts, const Vec &w_in, const LocalBasis &lb,
    int natom, double density_threshold,
    const occ::numint::BeckeWeightDerivative &dwdr, double fn_threshold,
    std::vector<Mat> &out) {
  const Eigen::Index n = pts.cols();
  const Eigen::Index nl = gto.phi.cols();
  const Eigen::Index o = Co.cols();
  const auto &atom_fns = lb.atom_fns;
  const auto &latoms = lb.atoms;
  const auto &rows = lb.rows;
  const int nal = static_cast<int>(latoms.size());
  const int parent = latoms[0];
  const DensityBlock<deriv> b(gto, P, funcs, w_in, density_threshold);
  if (b.negligible)
    return;
  const Vec &w = b.w;
  const Array &vr = b.vr, &frr = b.frr, &vs = b.vs, &frs = b.frs,
              &fss = b.fss, *gr = b.gr, &vt = b.vt, &frt = b.frt,
              &fst = b.fst, &ftt = b.ftt;
  const Mat &X = b.X, *Xd = b.Xd;

  const Mat &phi = gto.phi;
  const Mat *dphi[3] = {&gto.phi_x, &gto.phi_y, &gto.phi_z};
  Mat A = 0.5 * (phi.array().colwise() * vr).matrix();
  if constexpr (deriv >= 1) {
    for (int i = 0; i < 3; i++)
      A.array() += (dphi[i]->array().colwise() * (2.0 * vs * gr[i]));
  }
  const Mat wA = (A.array().colwise() * w.array()).matrix();
  const Mat X0 = phi * Co;
  const Mat ACo = A * Co;
  Mat Xg0[3];
  if constexpr (deriv >= 1)
    for (int i = 0; i < 3; i++)
      Xg0[i] = (*dphi[i]) * Co;
  // out[X] is o x nbf (transposed) so that scattering is contiguous; Vt is
  // o x nl.
  auto scatter = [&](int X, const Mat &Vt) {
    for (Eigen::Index i = 0; i < nl; i++)
      out[X].col(rows[i]) += Vt.col(i);
  };

  // Partition-weight term for every atom and direction as one GEMM: per
  // point, Q_g = (A Co)_g^T phi_g + (phi Co)_g^T A_g (o x nl), plus
  // 1/2 v_tau sum_i (d_i phi Co)_g^T (d_i phi)_g for a meta-GGA, is the same
  // for every coordinate, so the term is Q (o nl x n) . dW (n x 3N).
  {
    const int n3 = 3 * natom;
    const Mat dW = dwdr.batch(pts, w, parent);
    const Mat phiT = phi.transpose(), AT = A.transpose(); // nl x n
    const Mat ACoT = ACo.transpose(), X0T = X0.transpose(); // o x n
    Mat XgT[3], dphiT[3];
    if constexpr (deriv == 2)
      for (int i = 0; i < 3; i++) {
        XgT[i] = (Xg0[i].array().colwise() * (0.5 * vt)).matrix().transpose();
        dphiT[i] = dphi[i]->transpose();
      }
    Mat Q(o * nl, n);
    for (Eigen::Index g = 0; g < n; g++) {
      Eigen::Map<Mat> Qg(Q.col(g).data(), o, nl);
      Qg.noalias() = ACoT.col(g) * phiT.col(g).transpose();
      Qg.noalias() += X0T.col(g) * AT.col(g).transpose();
      if constexpr (deriv == 2)
        for (int i = 0; i < 3; i++)
          Qg.noalias() += XgT[i].col(g) * dphiT[i].col(g).transpose();
    }
    const Mat V = Q * dW; // (o nl) x 3N
    for (int X = 0; X < n3; X++)
      scatter(X, Eigen::Map<const Mat>(V.col(X).data(), o, nl));
  }

  // Explicit and kernel-response terms for atoms with functions here; the
  // points move with the parent (minus their sum, per direction).
  const auto active = active_atoms(gto, atom_fns, deriv >= 1, fn_threshold);
  // Contracted directly: with K = phi^T t + Kx (Kx: rows mu in B less
  // y_mu^T wA, columns mu in B less phi^T c_mu),
  //
  //   Co^T (K + K^T) = X0^T t + (t Co - C_B Co_B)^T phi - (Y_B Co_B)^T wA
  //                    - [X0^T C_B + (wA Co)^T Y_B] in the columns of B,
  //
  // with X0 = phi Co, Y_B = d_x phi on B's functions and C_B their c_mu, so
  // nothing nl x nl is formed. A meta-GGA's tau term adds
  //
  //   sum_i Lt_i^T d_i phi - [sum_i (1/2 w v_tau X_i)^T H_i] in the columns of B,
  //   Lt_i = 1/2 dv_tau X_i - 1/2 w v_tau H_i Co_B,
  //
  // with X_i = d_i phi Co and H_i = d_i d_x phi on B's functions.
  const Mat wACoT = (wA * Co).transpose(); // o x n
  const Mat X0T = X0.transpose();          // o x n
  Mat t(n, nl), CB, YB, CoB, Left(n, o), HB[3], Lt(n, o);
  Array rB(n), grB[3] = {Array(n), Array(n), Array(n)}, dvr(n), dvs(n),
      sB(n), c(n), ci[3] = {Array(n), Array(n), Array(n)}, tB(n), dvt(n);
  Mat wvtXg[3];
  if constexpr (deriv == 2)
    for (int i = 0; i < 3; i++)
      wvtXg[i] = (Xg0[i].array().colwise() * (0.5 * w.array() * vt)).matrix();
  for (int x = 0; x < 3; x++) {
    Mat Ssum = Mat::Zero(o, nl);
    for (int l = 0; l < nal; l++) {
      const auto &fns = atom_fns[l];
      if (fns.empty() || !active[l])
        continue;
      const Eigen::Index nb = fns.size();
      rB.setZero();
      tB.setZero();
      for (int i = 0; i < 3; i++)
        grB[i].setZero();
      CB.resize(n, nb);
      YB.resize(n, nb);
      CoB.resize(nb, o);
      if constexpr (deriv == 2)
        for (int i = 0; i < 3; i++)
          HB[i].resize(n, nb);
      for (Eigen::Index k = 0; k < nb; k++) {
        const int mu = fns[k];
        const auto y = dphi[x]->col(mu).array();
        rB -= 2.0 * y * X.col(mu).array();
        YB.col(k) = dphi[x]->col(mu);
        CoB.row(k) = Co.row(mu);
        c = 0.5 * w.array() * vr * y;
        if constexpr (deriv >= 1) {
          for (int i = 0; i < 3; i++) {
            grB[i] -= 2.0 * (gto.second(i, x).col(mu).array() * X.col(mu).array() +
                             y * Xd[i].col(mu).array());
            c += 2.0 * w.array() * vs * gr[i] * gto.second(i, x).col(mu).array();
          }
        }
        if constexpr (deriv == 2) {
          for (int i = 0; i < 3; i++) {
            tB -= gto.second(i, x).col(mu).array() * Xd[i].col(mu).array();
            HB[i].col(k) = gto.second(i, x).col(mu);
          }
        }
        CB.col(k) = c.matrix();
      }
      // kernel response t = 1/2 dvr phi + sum_i c_i d_i phi, and t Co
      if constexpr (deriv == 0) {
        dvr = w.array() * frr * rB;
        t = (phi.array().colwise() * (0.5 * dvr)).matrix();
        Left = (X0.array().colwise() * (0.5 * dvr)).matrix();
      } else {
        sB = 2.0 * (gr[0] * grB[0] + gr[1] * grB[1] + gr[2] * grB[2]);
        dvr = w.array() * (frr * rB + frs * sB);
        dvs = w.array() * (frs * rB + fss * sB);
        if constexpr (deriv == 2) {
          dvr += w.array() * frt * tB;
          dvs += w.array() * fst * tB;
          dvt = w.array() * (frt * rB + fst * sB + ftt * tB);
        }
        t = (phi.array().colwise() * (0.5 * dvr)).matrix();
        Left = (X0.array().colwise() * (0.5 * dvr)).matrix();
        for (int i = 0; i < 3; i++) {
          ci[i] = 2.0 * (dvs * gr[i] + w.array() * vs * grB[i]);
          t.array() += dphi[i]->array().colwise() * ci[i];
          Left.array() += Xg0[i].array().colwise() * ci[i];
        }
      }
      Left.noalias() -= CB * CoB;
      Mat Ct = X0T * t;                        // X0^T t
      Ct.noalias() += Left.transpose() * phi;  // (t Co - C_B Co_B)^T phi
      Ct.noalias() -= (YB * CoB).transpose() * wA;
      Mat block_o = X0T * CB + wACoT * YB; // o x nb
      if constexpr (deriv == 2) {
        for (int i = 0; i < 3; i++) {
          Lt = (Xg0[i].array().colwise() * (0.5 * dvt)).matrix();
          Lt.array() -= (HB[i] * CoB).array().colwise() * (0.5 * w.array() * vt);
          Ct.noalias() += Lt.transpose() * (*dphi[i]);
          block_o.noalias() += wvtXg[i].transpose() * HB[i];
        }
      }
      for (Eigen::Index k = 0; k < nb; k++)
        Ct.col(fns[k]) -= block_o.col(k);
      Ssum += Ct;
      scatter(3 * latoms[l] + x, Ct);
    }
    scatter(3 * parent + x, -Ssum);
  }
}

// dV_xc^s/dR C_o^s (see xc_derivative_fock_occ_block), out[X] (o x nbf,
// transposed) with alpha rows then beta rows.
template <int deriv>
void xc_derivative_fock_occ_block_unrestricted(
    const GTOValues &gto, const Mat &Co, Eigen::Index na,
    const Functionals &funcs, Eigen::Ref<const Mat3N> pts, const Vec &w_in,
    const LocalBasis &lb, int natom, double density_threshold,
    const occ::numint::BeckeWeightDerivative &dwdr, double fn_threshold,
    std::vector<Mat> &out) {
  const Eigen::Index n = pts.cols();
  const Eigen::Index nl = gto.phi.cols();
  const Eigen::Index c0[2] = {0, na};
  const auto &atom_fns = lb.atom_fns;
  const auto &latoms = lb.atoms;
  const auto &rows = lb.rows;
  const int nal = static_cast<int>(latoms.size());
  const int parent = latoms[0];
  const SpinBlock<deriv> sb(gto, Co, na, funcs, w_in, density_threshold);
  if (sb.negligible)
    return;
  const detail::SpinKernel<deriv> kernel(sb.f, sb.rho);
  const Array &w = sb.w;

  const Mat &phi = gto.phi;
  const Mat *dphi[3] = {&gto.phi_x, &gto.phi_y, &gto.phi_z};
  Array vr[2], V[2][3];
  Mat A[2], wA[2], X0[2], Xg0[2][3];
  for (int s = 0; s < 2; s++) {
    vr[s] = kernel.vr(s);
    A[s] = 0.5 * (phi.array().colwise() * vr[s]).matrix();
    if constexpr (deriv == 1)
      for (int i = 0; i < 3; i++) {
        V[s][i] = kernel.V(s, i);
        A[s].array() += dphi[i]->array().colwise() * V[s][i];
        Xg0[s][i] = (*dphi[i]) * sb.C[s];
      }
    wA[s] = (A[s].array().colwise() * w).matrix();
    X0[s] = phi * sb.C[s];
  }
  auto scatter = [&](int X, int s, const Mat &Vt) {
    auto o = out[X].middleRows(c0[s], Vt.rows());
    for (Eigen::Index i = 0; i < nl; i++)
      o.col(rows[i]) += Vt.col(i);
  };

  // Partition-weight term, one GEMM per spin.
  {
    const int n3 = 3 * natom;
    const Mat dW = dwdr.batch(pts, w.matrix(), parent);
    const Mat phiT = phi.transpose();
    for (int s = 0; s < 2; s++) {
      const Eigen::Index o = sb.C[s].cols();
      const Mat AT = A[s].transpose(), ACoT = (A[s] * sb.C[s]).transpose(),
                X0T = X0[s].transpose();
      Mat Q(o * nl, n);
      for (Eigen::Index g = 0; g < n; g++) {
        Eigen::Map<Mat> Qg(Q.col(g).data(), o, nl);
        Qg.noalias() = ACoT.col(g) * phiT.col(g).transpose();
        Qg.noalias() += X0T.col(g) * AT.col(g).transpose();
      }
      const Mat Vw = Q * dW;
      for (int X = 0; X < n3; X++)
        scatter(X, s, Eigen::Map<const Mat>(Vw.col(X).data(), o, nl));
    }
  }

  const auto active = active_atoms(gto, atom_fns, deriv >= 1, fn_threshold);
  for (int x = 0; x < 3; x++) {
    Mat Ssum[2] = {Mat::Zero(sb.C[0].cols(), nl),
                   Mat::Zero(sb.C[1].cols(), nl)};
    for (int l = 0; l < nal; l++) {
      const auto &fns = atom_fns[l];
      if (fns.empty() || !active[l])
        continue;
      const Eigen::Index nb = fns.size();
      Mat rB[2], grB[2][3], dvr[2], dV[2][3], CB[2], CoB[2];
      Mat YB(n, nb);
      for (int s = 0; s < 2; s++) {
        rB[s] = Mat::Zero(n, 1);
        for (int i = 0; i < 3; i++)
          grB[s][i] = Mat::Zero(n, 1);
        CB[s].resize(n, nb);
        CoB[s].resize(nb, sb.C[s].cols());
      }
      for (Eigen::Index k = 0; k < nb; k++) {
        const int mu = fns[k];
        const auto y = dphi[x]->col(mu).array();
        YB.col(k) = dphi[x]->col(mu);
        for (int s = 0; s < 2; s++) {
          rB[s].array() -= 2.0 * y * sb.X[s].col(mu).array();
          CoB[s].row(k) = sb.C[s].row(mu);
          Array c = 0.5 * w * vr[s] * y;
          if constexpr (deriv == 1)
            for (int i = 0; i < 3; i++) {
              grB[s][i].array() -=
                  2.0 * (gto.second(i, x).col(mu).array() * sb.X[s].col(mu).array() +
                         y * sb.Xd[s][i].col(mu).array());
              c += w * V[s][i] * gto.second(i, x).col(mu).array();
            }
          CB[s].col(k) = c.matrix();
        }
      }
      kernel.apply(rB, grB, dvr, dV);
      for (int s = 0; s < 2; s++) {
        const Array dv = w * dvr[s].col(0).array();
        Mat t = (phi.array().colwise() * (0.5 * dv)).matrix();
        Mat Left = (X0[s].array().colwise() * (0.5 * dv)).matrix();
        if constexpr (deriv == 1)
          for (int i = 0; i < 3; i++) {
            const Array ci = w * dV[s][i].col(0).array();
            t.array() += dphi[i]->array().colwise() * ci;
            Left.array() += Xg0[s][i].array().colwise() * ci;
          }
        Left.noalias() -= CB[s] * CoB[s];
        Mat Ct = X0[s].transpose() * t;
        Ct.noalias() += Left.transpose() * phi;
        Ct.noalias() -= (YB * CoB[s]).transpose() * wA[s];
        const Mat block_o = X0[s].transpose() * CB[s] +
                          (wA[s] * sb.C[s]).transpose() * YB;
        for (Eigen::Index k = 0; k < nb; k++)
          Ct.col(fns[k]) -= block_o.col(k);
        Ssum[s] += Ct;
        scatter(3 * latoms[l] + x, s, Ct);
      }
    }
    for (int s = 0; s < 2; s++)
      scatter(3 * parent + x, s, -Ssum[s]);
  }
}

// Skeleton XC Hessian of one chunk (see xc_hessian_block), unrestricted.
template <int deriv>
void xc_hessian_block_unrestricted(
    const GTOValues &gto, const Mat &Co, Eigen::Index na,
    const Functionals &funcs, Eigen::Ref<const Mat3N> pts, const Vec &w_in,
    const LocalBasis &lb, int natom, double density_threshold,
    const occ::numint::BeckeWeightDerivative &dwdr, Mat &H, double *edens) {
  const Eigen::Index n = pts.cols();
  const int n3 = 3 * natom;
  const std::vector<int> &lat = lb.fn_atom, &latoms = lb.atoms;
  const auto &atom_fns = lb.atom_fns;
  const int nal = static_cast<int>(latoms.size());
  const int n3l = 3 * nal;
  const Eigen::Index nl = gto.phi.cols();
  const int parent = latoms[0];
  for (Eigen::Index g = 0; g < n; g++)
    edens[g] = 0.0;
  Vec exc;
  const SpinBlock<deriv> sb(gto, Co, na, funcs, w_in, density_threshold, &exc);
  if (sb.negligible)
    return;
  const detail::SpinKernel<deriv> kernel(sb.f, sb.rho);
  const Array &W = sb.w;
  for (Eigen::Index g = 0; g < n; g++)
    edens[g] = (W(g) == 0.0) ? 0.0 : exc(g) * sb.total(g);

  const Mat *d1[3] = {&gto.phi_x, &gto.phi_y, &gto.phi_z};
  auto by_atom = [&](const Mat &Mfn, Mat &out, int a, double scale) {
    for (Eigen::Index mu = 0; mu < nl; mu++)
      out.col(3 * lat[mu] + a) += scale * Mfn.col(mu);
  };

  // First derivatives of the spin densities (basis only, point fixed), local
  // columns 3 l + a, and the kernel applied to them.
  Array vr[2], V[2][3];
  Mat R[2], G[2][3], dvr[2], dV[2][3];
  for (int s = 0; s < 2; s++) {
    vr[s] = kernel.vr(s);
    R[s] = Mat::Zero(n, n3l);
    for (int i = 0; i < 3; i++)
      G[s][i] = Mat::Zero(n, n3l);
    for (int a = 0; a < 3; a++) {
      by_atom(d1[a]->cwiseProduct(sb.X[s]), R[s], a, -2.0);
      if constexpr (deriv == 1)
        for (int i = 0; i < 3; i++)
          by_atom(gto.second(i, a).cwiseProduct(sb.X[s]) +
                      d1[a]->cwiseProduct(sb.Xd[s][i]),
                  G[s][i], a, -2.0);
    }
    if constexpr (deriv == 1)
      for (int i = 0; i < 3; i++)
        V[s][i] = kernel.V(s, i);
  }
  kernel.apply(R, G, dvr, dV);

  Mat Hb = Mat::Zero(n3l, n3l);
  Mat Fb = Mat::Zero(n, n3l); // first derivative of the energy density
  for (int s = 0; s < 2; s++) {
    Hb.noalias() += R[s].transpose() * (dvr[s].array().colwise() * W).matrix();
    Fb += (R[s].array().colwise() * vr[s]).matrix();
    if constexpr (deriv == 1)
      for (int i = 0; i < 3; i++) {
        Hb.noalias() +=
            G[s][i].transpose() * (dV[s][i].array().colwise() * W).matrix();
        Fb += (G[s][i].array().colwise() * V[s][i]).matrix();
      }

    // Off-diagonal function pairs through P^s = C^s C^s^T.
    const Mat &Cs = sb.C[s];
    const Eigen::Index o = Cs.cols();
    auto project = [&](const Mat &Mfn) {
      Mat out(n * o, nal);
      for (int l = 0; l < nal; l++) {
        Mat acc = Mat::Zero(n, o);
        for (int mu : atom_fns[l])
          acc.noalias() += Mfn.col(mu) * Cs.row(mu);
        out.col(l) = Eigen::Map<const Vec>(acc.data(), n * o);
      }
      return out;
    };
    Mat alpha[3], delta[3], beta[3];
    for (int a = 0; a < 3; a++) {
      Mat Aa = (d1[a]->array().colwise() * (W * vr[s])).matrix();
      if constexpr (deriv == 1) {
        Mat Bb = Mat::Zero(n, nl);
        for (int i = 0; i < 3; i++) {
          const Array c = W * V[s][i];
          Aa.array() += gto.second(i, a).array().colwise() * c;
          Bb.array() += gto.second(i, a).array().colwise() * c;
        }
        beta[a] = project(Bb);
      }
      alpha[a] = project(Aa);
      delta[a] = project(*d1[a]);
    }
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        Mat blocks = 2.0 * alpha[a].transpose() * delta[b];
        if constexpr (deriv == 1)
          blocks.noalias() += 2.0 * delta[a].transpose() * beta[b];
        for (int D = 0; D < nal; D++)
          for (int E = 0; E < nal; E++)
            Hb(3 * D + a, 3 * E + b) += blocks(D, E);
      }
    // Both derivatives on the same function.
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        Mat c = gto.second(a, b).cwiseProduct(sb.X[s]).array().colwise() * (W * vr[s]);
        if constexpr (deriv == 1)
          for (int i = 0; i < 3; i++)
            c.array() += (gto.third(i, a, b).cwiseProduct(sb.X[s]) +
                          gto.second(a, b).cwiseProduct(sb.Xd[s][i]))
                             .array()
                             .colwise() *
                         (W * V[s][i]);
        const Vec colsum = c.colwise().sum().transpose();
        for (Eigen::Index mu = 0; mu < nl; mu++)
          Hb(3 * lat[mu] + a, 3 * lat[mu] + b) += 2.0 * colsum(mu);
      }
  }

  // Local to global coordinates with the points moving with the parent, and
  // the first-order weight cross terms.
  Mat T = Mat::Zero(n3, n3l);
  for (int l = 0; l < nal; l++)
    for (int x = 0; x < 3; x++) {
      T(3 * latoms[l] + x, 3 * l + x) += 1.0;
      T(3 * parent + x, 3 * l + x) -= 1.0;
    }
  H += T * Hb * T.transpose();
  const Mat Ft = Fb * T.transpose();
  const Mat cross = dwdr.batch(pts, W.matrix(), parent).transpose() * Ft;
  H += cross + cross.transpose();
}

} // namespace

occ::numint::BeckeWeightDerivative DFT::becke_weight_derivative() const {
  return occ::numint::BeckeWeightDerivative(
      atom_positions(), m_grid.atomic_radii(),
      m_grid.settings().treutler_alrichs_adjustment);
}

Mat3N DFT::atom_positions() const {
  const auto &atoms = m_hf.atoms();
  Mat3N positions(3, atoms.size());
  for (size_t a = 0; a < atoms.size(); a++)
    positions.col(a) = atoms[a].position();
  return positions;
}

std::vector<Mat> DFT::compute_xc_derivative_fock(const MolecularOrbitals &mo) const {
  if (mo.kind != SpinorbitalKind::Restricted)
    throw std::runtime_error("XC derivative Fock: restricted only");
  if (needs_laplacian())
    throw std::runtime_error("XC derivative Fock: Laplacian-dependent "
                             "meta-GGAs are not supported");
  const int deriv = density_derivative();
  occ::timing::start(occ::timing::category::dft_xc);
  const auto &basis = m_hf.aobasis();
  const int natom = static_cast<int>(m_hf.atoms().size());
  const size_t nbf = basis.nbf();
  const Mat P = 2.0 * mo.D;
  const auto &funcs = m_method.functionals;

  std::vector<std::vector<int>> atom_bfs(natom);
  const auto &bf_atom = basis.bf_to_atom();
  for (size_t mu = 0; mu < nbf; mu++)
    atom_bfs[bf_atom[mu]].push_back(static_cast<int>(mu));

  const auto dwdr = becke_weight_derivative();
  const auto &grid = m_grid.get_molecular_grid_points();
  const auto chunks = atom_chunks(grid);
  occ::parallel::thread_local_storage<std::vector<Mat>> tls([natom, nbf]() {
    return std::vector<Mat>(3 * natom, Mat::Zero(nbf, nbf));
  });
  occ::parallel::parallel_for(size_t(0), chunks.size(), [&](size_t c) {
    const auto &ch = chunks[c];
    const auto pts = grid.points().middleCols(ch.start, ch.count);
    const Vec w = grid.weights().segment(ch.start, ch.count);
    if (deriv == 0)
      xc_derivative_fock_block<0>(basis, P, funcs, pts, w, ch.atom,
                                  m_density_threshold, dwdr, atom_bfs,
                                  tls.local());
    else if (deriv == 1)
      xc_derivative_fock_block<1>(basis, P, funcs, pts, w, ch.atom,
                                  m_density_threshold, dwdr, atom_bfs,
                                  tls.local());
    else
      xc_derivative_fock_block<2>(basis, P, funcs, pts, w, ch.atom,
                                  m_density_threshold, dwdr, atom_bfs,
                                  tls.local());
  });
  std::vector<Mat> result(3 * natom, Mat::Zero(nbf, nbf));
  for (const auto &t : tls)
    for (int k = 0; k < 3 * natom; k++)
      result[k] += t[k];
  occ::timing::stop(occ::timing::category::dft_xc);
  return result;
}

std::vector<Mat>
DFT::compute_xc_derivative_fock_occ(const MolecularOrbitals &mo) const {
  const bool unrestricted = mo.kind == SpinorbitalKind::Unrestricted;
  if (needs_laplacian())
    throw std::runtime_error("XC derivative Fock: Laplacian-dependent "
                             "meta-GGAs are not supported");
  const int deriv = density_derivative();
  if (unrestricted && deriv > 1)
    throw std::runtime_error(
        "XC derivative Fock: unrestricted meta-GGAs are not supported");
  occ::timing::start(occ::timing::category::dft_xc);
  const auto &basis = m_hf.aobasis();
  const int natom = static_cast<int>(m_hf.atoms().size());
  const size_t nbf = basis.nbf();
  const Mat P = unrestricted ? Mat() : Mat(2.0 * mo.D);
  // every spin channel's occupied columns (the CPHF layout)
  const Mat Co = qm::cphf::occupied_orbitals(mo);
  const Eigen::Index o = Co.cols();
  const auto &funcs =
      unrestricted ? m_method.functionals_polarized : m_method.functionals;
  const auto dwdr = becke_weight_derivative();
  const Vec extents =
      m_xc_screening_threshold > 0.0
          ? occ::gto::evaluate_decay_cutoff(basis, m_xc_screening_threshold)
          : Vec();

  const auto &grid = m_grid.get_molecular_grid_points();
  const auto chunks = atom_chunks(grid);
  occ::parallel::thread_local_storage<std::vector<Mat>> tls([natom, nbf, o]() {
    return std::vector<Mat>(3 * natom, Mat::Zero(o, nbf));
  });
  occ::parallel::parallel_for(size_t(0), chunks.size(), [&](size_t c) {
    const auto &ch = chunks[c];
    const auto pts = grid.points().middleCols(ch.start, ch.count);
    const Vec w = grid.weights().segment(ch.start, ch.count);
    const LocalBasis lb(basis, extents, pts, ch.atom, natom);
    if (lb.shells.empty())
      return;
    const Mat Pl = unrestricted ? Mat() : lb.block_of(P);
    const Mat Col = lb.rows_of(Co);
    const auto gto = occ::gto::evaluate_basis_subset(
        basis, pts, std::min(deriv, 1) + 1, lb.shells);
    const double thr = m_density_threshold, fthr = m_xc_screening_threshold;
    auto &out = tls.local();
    if (unrestricted) {
      if (deriv == 0)
        xc_derivative_fock_occ_block_unrestricted<0>(
            gto, Col, mo.n_alpha, funcs, pts, w, lb, natom, thr, dwdr, fthr,
            out);
      else
        xc_derivative_fock_occ_block_unrestricted<1>(
            gto, Col, mo.n_alpha, funcs, pts, w, lb, natom, thr, dwdr, fthr,
            out);
    } else if (deriv == 0)
      xc_derivative_fock_occ_block<0>(gto, Pl, Col, funcs, pts, w, lb, natom,
                                      thr, dwdr, fthr, out);
    else if (deriv == 1)
      xc_derivative_fock_occ_block<1>(gto, Pl, Col, funcs, pts, w, lb, natom,
                                      thr, dwdr, fthr, out);
    else
      xc_derivative_fock_occ_block<2>(gto, Pl, Col, funcs, pts, w, lb, natom,
                                      thr, dwdr, fthr, out);
  });
  std::vector<Mat> result(3 * natom, Mat::Zero(nbf, o));
  for (const auto &t : tls)
    for (int k = 0; k < 3 * natom; k++)
      result[k] += t[k].transpose();
  occ::timing::stop(occ::timing::category::dft_xc);
  return result;
}

Mat DFT::compute_xc_hessian(const MolecularOrbitals &mo) const {
  const bool unrestricted = mo.kind == SpinorbitalKind::Unrestricted;
  if (needs_laplacian())
    throw std::runtime_error("XC Hessian: Laplacian-dependent meta-GGAs are "
                             "not supported");
  const int deriv = density_derivative();
  if (unrestricted && deriv > 1)
    throw std::runtime_error(
        "XC Hessian: unrestricted meta-GGAs are not supported");
  occ::timing::start(occ::timing::category::dft_xc);
  const auto &basis = m_hf.aobasis();
  const int natom = static_cast<int>(m_hf.atoms().size());
  const int n3 = 3 * natom;
  const Mat P = unrestricted ? Mat() : Mat(2.0 * mo.D);
  // A factor with P = 2 Co Co^T: the occupied orbitals, or for fractional
  // occupations the density's own eigenvectors. Unrestricted: the spin
  // channels' occupied orbitals side by side.
  Mat Co = qm::cphf::occupied_orbitals(mo);
  if (!unrestricted && (2.0 * Co * Co.transpose() - P).cwiseAbs().maxCoeff() >
                           1e-10 * std::max(1.0, P.cwiseAbs().maxCoeff())) {
    Eigen::SelfAdjointEigenSolver<Mat> es(mo.D);
    std::vector<Eigen::Index> keep;
    for (Eigen::Index i = 0; i < es.eigenvalues().size(); i++)
      if (es.eigenvalues()(i) > 1e-14)
        keep.push_back(i);
    Co.resize(mo.D.rows(), keep.size());
    for (size_t k = 0; k < keep.size(); k++)
      Co.col(k) = es.eigenvectors().col(keep[k]) *
                  std::sqrt(es.eigenvalues()(keep[k]));
  }
  const auto &funcs =
      unrestricted ? m_method.functionals_polarized : m_method.functionals;
  const auto dwdr = becke_weight_derivative();
  // Shells whose functions are negligible over a chunk drop out of it (as in
  // the SCF's XC build); threshold 0 keeps every shell.
  const Vec extents =
      m_xc_screening_threshold > 0.0
          ? occ::gto::evaluate_decay_cutoff(basis, m_xc_screening_threshold)
          : Vec();

  const auto &grid = m_grid.get_molecular_grid_points();
  const auto chunks = atom_chunks(grid);
  Vec edens = Vec::Zero(grid.num_points());
  occ::parallel::thread_local_storage<Mat> tls(
      [n3]() { return Mat(Mat::Zero(n3, n3)); });
  occ::parallel::parallel_for(size_t(0), chunks.size(), [&](size_t c) {
    const auto &ch = chunks[c];
    const auto pts = grid.points().middleCols(ch.start, ch.count);
    const Vec w = grid.weights().segment(ch.start, ch.count);
    const LocalBasis lb(basis, extents, pts, ch.atom, natom);
    if (lb.shells.empty())
      return;
    const Mat Pl = unrestricted ? Mat() : lb.block_of(P);
    const Mat Col = lb.rows_of(Co);
    const auto gto = occ::gto::evaluate_basis_subset(
        basis, pts, std::min(deriv, 1) + 2, lb.shells);
    const double thr = m_density_threshold;
    double *e = edens.data() + ch.start;
    if (unrestricted) {
      if (deriv == 0)
        xc_hessian_block_unrestricted<0>(gto, Col, mo.n_alpha, funcs, pts, w,
                                         lb, natom, thr, dwdr, tls.local(), e);
      else
        xc_hessian_block_unrestricted<1>(gto, Col, mo.n_alpha, funcs, pts, w,
                                         lb, natom, thr, dwdr, tls.local(), e);
    } else if (deriv == 0)
      xc_hessian_block<0>(gto, Pl, Col, funcs, pts, w, lb, natom, thr, dwdr,
                          tls.local(), e);
    else if (deriv == 1)
      xc_hessian_block<1>(gto, Pl, Col, funcs, pts, w, lb, natom, thr, dwdr,
                          tls.local(), e);
    else
      xc_hessian_block<2>(gto, Pl, Col, funcs, pts, w, lb, natom, thr, dwdr,
                          tls.local(), e);
  });
  Mat H = Mat::Zero(n3, n3);
  for (const auto &t : tls)
    H += t;
  // Second-order weight term.
  H += occ::numint::becke_weight_hessian(
      grid.points(), grid.weights(), grid.atom_blocks(), edens,
      atom_positions(), m_grid.atomic_radii(),
      m_grid.settings().treutler_alrichs_adjustment);
  occ::timing::stop(occ::timing::category::dft_xc);
  return 0.5 * (H + H.transpose());
}

Mat DFT::two_electron_hessian(const MolecularOrbitals &mo,
                              const Mat &Schwarz) const {
  return m_hf.compute_jk_hessian(mo, 1.0, exchange_operators(), Schwarz) +
         compute_xc_hessian(mo);
}

std::vector<Mat>
DFT::two_electron_derivative_fock(const MolecularOrbitals &mo,
                                  const Mat &Schwarz) const {
  auto F = m_hf.compute_jk_derivative_fock(mo, 1.0, exchange_operators(),
                                           Schwarz);
  const auto V = compute_xc_derivative_fock(mo);
  for (size_t k = 0; k < F.size(); k++)
    F[k] += V[k];
  return F;
}

std::vector<Mat>
DFT::two_electron_derivative_fock_occ(const MolecularOrbitals &mo,
                                      const Mat &Schwarz) const {
  auto F = m_hf.compute_jk_derivative_fock_occ(mo, 1.0, exchange_operators(),
                                               Schwarz);
  const auto V = compute_xc_derivative_fock_occ(mo);
  for (size_t k = 0; k < F.size(); k++)
    F[k] += V[k];
  return F;
}

} // namespace occ::dft
