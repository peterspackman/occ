// Second derivatives of the COSX (seminumerical) exchange energy.
//
// With the per-spin density D = C_o C_o^T, the projector Q = S_num^{-1} S and
// K'[Z] = sum_g w_g phi_g (A(g) Z phi_g)^T, the SCF's exchange matrix is
// K = sym(K'[D Q^T]) and the energy (see seminumerical_exchange_grad.cpp) is
//
//   E_x = -e[D Q^T],   e[Z] = sum_g w_g (D phi_g) . A(g) (Z phi_g) = tr(Z K'[D]).
//
// e is linear in Q, e = <Q, K'[D] D>, so with N = S_num^{-1} K'[D] D = N_o C_o^T
// and dQ = S_num^{-1} (dS - dS_num Q),
//
//   e^xy = e^xy|_Q + <Q^x C_o, K'^y[D] C_o> + <Q^y C_o, K'^x[D] C_o>
//          + <S^xy, N> - <S_num^xy, N Q^T>
//          - <Q^x C_o, S_num^y N_o> - <Q^y C_o, S_num^x N_o>.
//
// e^xy|_Q (with the S_num^xy term as the extra integrand phi^T B phi,
// B = -sym(N Q^T)) and the Fock derivatives come from one sweep over the
// grid. Per batch of points with one parent atom, points held fixed: the
// basis functions on each atom (in phi, u = D phi and v = D Q^T phi), and
// the ESP integrals through their centres (first derivatives per point,
// second derivatives contracted with W = w (u v^T + v u^T)). The points then
// move with their parent (translational invariance), and the partition
// weights enter through their first derivatives against the integrand's
// per-point first derivatives and their second derivatives against the
// integrand itself. Every vector A(g) meets is in the span of C_o (u = C_o
// psi, the derivative vectors likewise), so A(g) is applied to the columns of
// C_o only and the vectors follow point by point.
//
// The SCF's exchange derivative at fixed density is
// sym(K'^x[D Q^T] + K'[D Q^x^T]). The COSX Fock is not exactly
// -1/2 dE_x/dD = 1/2 [K + sym(Q^T K'[D])], and the gradient's density
// response sees the derivative of the latter, `energy_fock`.

#include "detail/cosx_contract.h"
#include "detail/cosx_grid.h"
#include <Eigen/Cholesky>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/gto/gto.h>
#include <occ/numint/grid_utils.h>
#include <occ/qm/seminumerical_exchange.h>

namespace occ::qm::cosx {

namespace {

constexpr int sym3[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};

// d/dX = d/dX|points fixed - delta(X, parent) sum_Y d/dY|points fixed, per
// direction, on the rows or columns indexed 3 X + x.
void move_points_rows(Mat &M, int parent) {
  const Eigen::Index natom = M.rows() / 3;
  for (int x = 0; x < 3; x++) {
    Eigen::RowVectorXd s = Eigen::RowVectorXd::Zero(M.cols());
    for (Eigen::Index A = 0; A < natom; A++)
      s += M.row(3 * A + x);
    M.row(3 * parent + x) -= s;
  }
}

void move_points_cols(Mat &M, int parent) {
  const Eigen::Index natom = M.cols() / 3;
  for (int x = 0; x < 3; x++) {
    Vec s = Vec::Zero(M.rows());
    for (Eigen::Index A = 0; A < natom; A++)
      s += M.col(3 * A + x);
    M.col(3 * parent + x) -= s;
  }
}

Mat rows_of(const Mat &M, const std::vector<int> &idx) {
  Mat out(idx.size(), M.cols());
  for (size_t k = 0; k < idx.size(); k++)
    out.row(k) = M.row(idx[k]);
  return out;
}

} // namespace

std::vector<Mat>
SemiNumericalExchange::numerical_overlap_derivatives(const Mat &Y) const {
  const int nbf = static_cast<int>(m_basis.nbf());
  const int natom = static_cast<int>(m_atoms.size());
  const int n3 = 3 * natom;
  const Eigen::Index nc = Y.cols();
  std::vector<std::vector<int>> atom_bfs(natom);
  for (int mu = 0; mu < nbf; mu++)
    atom_bfs[m_basis.bf_to_atom()[mu]].push_back(mu);
  const occ::numint::BeckeWeightDerivative dwdr(
      detail::atom_positions(m_atoms), m_grid.atomic_radii(),
      m_grid.settings().treutler_alrichs_adjustment);
  const auto batches = detail::atom_batches(m_atom_grids, 128);

  occ::parallel::thread_local_storage<std::vector<Mat>> tls(
      [&]() { return std::vector<Mat>(n3, Mat::Zero(nbf, nc)); });
  occ::parallel::parallel_for(size_t(0), batches.size(), [&](size_t bi) {
    const auto &bt = batches[bi];
    auto &F = tls.local();
    const int P = bt.atom;
    const auto &ag = m_atom_grids[P];
    const auto pts = ag.points.middleCols(bt.start, bt.count);
    const Vec w = ag.weights.segment(bt.start, bt.count);
    const Eigen::Index npt = bt.count;
    occ::gto::GTOValues ao;
    occ::gto::evaluate_basis(m_basis, pts, ao, 1);
    const Mat *dphi[3] = {&ao.phi_x, &ao.phi_y, &ao.phi_z};
    const Mat phiY = ao.phi * Y;
    const Mat wphiY = phiY.array().colwise() * w.array();
    // S_num = sum_g w phi phi^T: basis functions, points fixed ...
    Mat sum[3];
    for (int x = 0; x < 3; x++)
      sum[x] = Mat::Zero(nbf, nc);
    for (int A = 0; A < natom; A++) {
      const auto &fns = atom_bfs[A];
      const Mat YA = rows_of(Y, fns);
      for (int a = 0; a < 3; a++) {
        Mat Pk(npt, fns.size());
        for (size_t j = 0; j < fns.size(); j++)
          Pk.col(j) = -dphi[a]->col(fns[j]);
        Mat f = ao.phi.transpose() *
                ((Pk * YA).array().colwise() * w.array()).matrix();
        const Mat r = Pk.transpose() * wphiY;
        for (size_t j = 0; j < fns.size(); j++)
          f.row(fns[j]) += r.row(j);
        F[3 * A + a] += f;
        sum[a] += f;
      }
    }
    // ... points moving with the parent, and the partition weights
    for (int x = 0; x < 3; x++)
      F[3 * P + x] -= sum[x];
    const Mat dW = dwdr.batch(pts, w, P);
    Mat R(npt, n3 * nc);
    for (int X = 0; X < n3; X++)
      R.middleCols(X * nc, nc) = phiY.array().colwise() * dW.col(X).array();
    const Mat O = ao.phi.transpose() * R;
    for (int X = 0; X < n3; X++)
      F[X] += O.middleCols(X * nc, nc);
  });
  std::vector<Mat> result(n3, Mat::Zero(nbf, nc));
  for (const auto &F : tls)
    for (int X = 0; X < n3; X++)
      result[X] += F[X];
  return result;
}

SemiNumericalExchange::SecondDerivatives
SemiNumericalExchange::compute_K_second_derivatives(
    const qm::MolecularOrbitals &mo, double omega, double threshold) const {
  if (mo.kind != SpinorbitalKind::Restricted)
    throw std::runtime_error("COSX exchange Hessian: unrestricted orbitals are not supported");
  if (!m_use_esp)
    throw std::runtime_error("COSX exchange Hessian requires ESP mode");
  occ::timing::start(occ::timing::category::cosx);
  prepare_screening();

  const auto &shells = m_basis.shells();
  const auto &first_bf = m_basis.first_bf();
  const auto &shell_atom = m_basis.shell_to_atom();
  const int nbf = static_cast<int>(m_basis.nbf());
  const int natom = static_cast<int>(m_atoms.size());
  const int n3 = 3 * natom;
  const bool spherical = m_basis.is_pure();
  std::vector<std::vector<int>> atom_bfs(natom);
  for (int mu = 0; mu < nbf; mu++)
    atom_bfs[m_basis.bf_to_atom()[mu]].push_back(mu);

  const Mat &D = mo.D;
  const Mat Co = mo.C.leftCols(mo.n_alpha);
  if ((Co * Co.transpose() - D).cwiseAbs().maxCoeff() >
      1e-10 * std::max(1.0, D.cwiseAbs().maxCoeff()))
    throw std::runtime_error(
        "COSX exchange Hessian: needs D = C_o C_o^T (integer occupations)");
  const int no = static_cast<int>(Co.cols());
  const Mat &Q = m_overlap_projector;
  const Mat QD = Q * D, DQt = QD.transpose(), QCo = Q * Co;
  const auto Sn_ldlt = m_numerical_overlap.ldlt();

  // K'[D], and the overlap-fitting quantities N_o and B
  const Mat Kd = exchange_sweep({D}, omega)[0];
  const Mat KdCo = Kd * Co;
  const Mat No = Sn_ldlt.solve(KdCo);
  const Mat NQt = No * QCo.transpose();
  const Mat B = -0.5 * (NQt + NQt.transpose());

  // Q^X C_o = S_num^{-1} (S^X C_o - S_num^X Q C_o), with the analytic overlap
  // derivative dS_mn/dR_B = -[(grad m|n) + (grad n|m)], m, n on B; and
  // S^X N_o, S_num^X N_o for the projector's second-order terms.
  Mat Ysn(nbf, 2 * no);
  Ysn << QCo, No;
  const auto SnY = numerical_overlap_derivatives(Ysn);
  std::vector<Mat> QxCo(n3), SNo(n3);
  {
    const MatTriple Sg = m_engine.one_electron_operator_grad(
        qm::IntegralEngine::Op::overlap);
    Mat CN(nbf, 2 * no);
    CN << Co, No;
    for (int x = 0; x < 3; x++) {
      const Mat SgCN = Sg[x] * CN;
      for (int A = 0; A < natom; A++) {
        const auto &fns = atom_bfs[A];
        const int X = 3 * A + x;
        Mat SX = -(rows_of(Sg[x], fns).transpose() * rows_of(CN, fns));
        for (size_t j = 0; j < fns.size(); j++)
          SX.row(fns[j]) -= SgCN.row(fns[j]);
        QxCo[X] = Sn_ldlt.solve(SX.leftCols(no) - SnY[X].leftCols(no));
        SNo[X] = SX.rightCols(no);
      }
    }
  }
  Mat QxCoAll(nbf, static_cast<Eigen::Index>(no) * n3);
  for (int X = 0; X < n3; X++)
    QxCoAll.middleCols(X * no, no) = QxCo[X];
  // bound on |D Q^X^T phi| per function, for the pair screen
  Vec zcol = Vec::Zero(no);
  for (int X = 0; X < n3; X++)
    zcol = zcol.cwiseMax(QxCo[X].cwiseAbs().colwise().sum().transpose());

  const size_t npairs = m_significant_pairs.size();
  occ::ints::ESPEvaluator<double> hesp(occ::ints::boys().table());
  hesp.resize_hessian_pairs(npairs);
  occ::parallel::parallel_for(size_t(0), npairs, [&](size_t e) {
    const auto [pp, qq] = m_significant_pairs[e];
    const auto &sp = shells[pp], &sq = shells[qq];
    hesp.set_shell_pair_hessian(
        e, sp.l, sq.l, sp.num_primitives(), sq.num_primitives(),
        sp.exponents.data(), sq.exponents.data(),
        sp.contraction_coefficients.col(0).data(),
        sq.contraction_coefficients.col(0).data(), sp.origin.data(),
        sq.origin.data(), spherical);
  });

  // Batches of one atom's points, as large as fit ~64 MB of per-point
  // vectors (A C_o, and five of nbf per coordinate) per thread.
  const double per_point = (no + 15.0 * natom + 2) * nbf * sizeof(double);
  Eigen::Index BLOCK = 128;
  while (BLOCK > 32 && per_point * BLOCK > double(1 << 26))
    BLOCK /= 2;
  std::vector<std::pair<size_t, size_t>> atom_blocks;
  const auto batches = detail::atom_batches(m_atom_grids, BLOCK, &atom_blocks);
  const auto &grid = grid_batches();
  Vec edens = Vec::Zero(grid.weights.size());

  const Mat3N positions = detail::atom_positions(m_atoms);
  const bool ta = m_grid.settings().treutler_alrichs_adjustment;
  const occ::numint::BeckeWeightDerivative dwdr(positions,
                                                m_grid.atomic_radii(), ta);

  // Per coordinate X, nbf x 3 no: [dK/dX C_o | K'^X[D] C_o |
  // K'^X[D]^T Q C_o], the last two at fixed Q.
  constexpr int PAD = occ::ints::ESPEvaluator<double>::soa_padding;
  const int nf = 3 * no;
  struct TL {
    Mat H;
    std::vector<Mat> F;
    std::vector<double> x, y, z, Vh, W, dsum;
  };
  occ::parallel::thread_local_storage<TL> tls([&]() {
    return TL{Mat::Zero(n3, n3), std::vector<Mat>(n3, Mat::Zero(nbf, nf))};
  });

  occ::parallel::parallel_for(size_t(0), batches.size(), [&](size_t bi) {
    const auto &bt = batches[bi];
    auto &t = tls.local();
    const int P = bt.atom;
    const auto &ag = m_atom_grids[P];
    const auto pts = ag.points.middleCols(bt.start, bt.count);
    const int npt = static_cast<int>(bt.count);
    const int npad = (npt + PAD - 1) / PAD * PAD;
    detail::soa_coordinates(pts, npad, t.x, t.y, t.z);
    Vec w = Vec::Zero(npad);
    w.head(npt) = ag.weights.segment(bt.start, npt);
    auto wscale = [&](const auto &M) {
      return Mat(M.array().colwise() * w.array());
    };

    // Basis values padded with zero rows, so everything built from them
    // vanishes on the padding points.
    occ::gto::GTOValues ao;
    occ::gto::evaluate_basis(m_basis, pts, ao, 2);
    auto pad = [&](const Mat &m) {
      Mat out = Mat::Zero(npad, nbf);
      out.topRows(npt) = m;
      return out;
    };
    const Mat phi = pad(ao.phi);
    const Mat dphi[3] = {pad(ao.phi_x), pad(ao.phi_y), pad(ao.phi_z)};

    // u = C_o psi, v = C_o chi; the derivative vectors u^k, v^k for
    // k = 3 i + a (atom bl[i], direction a) likewise with psi^k, chi^k from
    // -d_a phi on the atom's functions; and D Q^X^T phi = C_o xi^X.
    const Mat U = phi * D, V = phi * QD;
    const Mat psi = phi * Co, chi = phi * QCo, xi = phi * QxCoAll;
    std::vector<int> bl;
    Vec dmax = Vec::Zero(nbf); // max |d phi| over the batch
    for (int A = 0; A < natom; A++) {
      double m = 0.0;
      for (int mu : atom_bfs[A]) {
        for (int a = 0; a < 3; a++)
          dmax(mu) = std::max(dmax(mu), dphi[a].col(mu).cwiseAbs().maxCoeff());
        m = std::max(m, dmax(mu));
      }
      if (m > 1e-12)
        bl.push_back(A);
    }
    const int nb = 3 * static_cast<int>(bl.size());
    const Eigen::Index blk = static_cast<Eigen::Index>(npad) * nbf;
    const Eigen::Index oblk = static_cast<Eigen::Index>(npad) * no;
    auto block = [&](Mat &M, int k) {
      return Eigen::Map<Mat>(M.col(k).data(), npad, M.rows() / npad);
    };
    auto cblock = [&](const Mat &M, int k) {
      return Eigen::Map<const Mat>(M.col(k).data(), npad, M.rows() / npad);
    };
    Mat psik(oblk, nb), chik(oblk, nb);
    std::vector<Mat> Pa(nb); // -d_a phi on the atom's functions
    // bound on |u^k|, |v^k| per function, for the pair screen
    Vec kmax = Vec::Zero(nbf);
    for (size_t i = 0; i < bl.size(); i++) {
      const auto &fns = atom_bfs[bl[i]];
      const Mat Coa = rows_of(Co, fns), QCoa = rows_of(QCo, fns);
      Vec m(fns.size());
      for (size_t j = 0; j < fns.size(); j++)
        m(j) = dmax(fns[j]);
      kmax = kmax.cwiseMax(
          (rows_of(D, fns).cwiseAbs().transpose() * m)
              .cwiseMax(rows_of(QD, fns).cwiseAbs().transpose() * m));
      for (int a = 0; a < 3; a++) {
        const int k = 3 * i + a;
        Pa[k].resize(npad, fns.size());
        for (size_t j = 0; j < fns.size(); j++)
          Pa[k].col(j) = -dphi[a].col(fns[j]);
        block(psik, k) = Pa[k] * Coa;
        block(chik, k) = Pa[k] * QCoa;
      }
    }

    // Shell pairs, screened on their ESP bound against the largest vector
    // they meet; the atoms whose integrals they differentiate get slots.
    const Vec zmax =
        Co.cwiseAbs() * (zcol * phi.cwiseAbs().maxCoeff());
    const Vec fmax = U.cwiseAbs()
                         .colwise()
                         .maxCoeff()
                         .transpose()
                         .cwiseMax(V.cwiseAbs().colwise().maxCoeff().transpose())
                         .cwiseMax(kmax)
                         .cwiseMax(zmax);
    Vec smax(shells.size());
    for (size_t s = 0; s < shells.size(); s++)
      smax(s) = fmax.segment(first_bf[s], shells[s].size()).maxCoeff();
    const GridBatchInfo sphere = compute_batch_info(pts);
    std::vector<size_t> live;
    std::vector<int> slot(natom, -1), touched;
    for (size_t e = 0; e < npairs; e++) {
      const auto [p, q] = m_significant_pairs[e];
      const double bound = detail::pair_charge_bound(
          m_pair_charge(e), m_pair_centers.col(e), m_pair_far.col(e).data(),
          sphere);
      if (bound * smax(p) * smax(q) < threshold)
        continue;
      live.push_back(e);
      for (int A : {shell_atom[p], shell_atom[q]})
        if (slot[A] < 0) {
          slot[A] = static_cast<int>(touched.size());
          touched.push_back(A);
        }
    }
    const int nt = 3 * static_cast<int>(touched.size());

    // A C_o (one npad x nbf block per column), and (dA/dX) applied to u and
    // v (YU, YV, column 3 slot + d; contract_pair subtracts, so these are
    // negated until the end of the pair loop).
    Mat GC = Mat::Zero(blk, no);
    Mat YU = Mat::Zero(blk, nt), YV = Mat::Zero(blk, nt);
    Mat Hf = Mat::Zero(n3, n3);
    using occ::qm::cosx::detail::contract_pair;
    using occ::qm::cosx::detail::contract_pair_coefficients;
    for (size_t e : live) {
      const auto [ps, qs] = m_significant_pairs[e];
      const int p = static_cast<int>(ps), q = static_cast<int>(qs);
      const bool two = p != q;
      const int np = shells[p].size(), nq = shells[q].size();
      const int bp = first_bf[p], bq = first_bf[q], nab = np * nq;
      const int C = shell_atom[p], Dt = shell_atom[q];
      t.W.resize(static_cast<size_t>(nab) * npad);
      for (int a = 0; a < np; a++)
        for (int b = 0; b < nq; b++) {
          double *Wab = t.W.data() + static_cast<size_t>(a * nq + b) * npad;
          const double *ua = U.col(bp + a).data(), *vb = V.col(bq + b).data();
          const double *va = V.col(bp + a).data(), *ub = U.col(bq + b).data();
          for (int g = 0; g < npad; g++)
            Wab[g] = w(g) * (ua[g] * vb[g] + (two ? va[g] * ub[g] : 0.0));
        }
      t.Vh.resize(static_cast<size_t>(7 * nab) * npad);
      double s21[21] = {};
      hesp.evaluate_soa_hessian(e, t.x.data(), t.y.data(), t.z.data(), npad,
                                t.Vh.data(), npad, t.W.data(), npad, s21,
                                omega);
      const double *val = t.Vh.data();
      contract_pair_coefficients(Co.data(), nbf, no, GC.data(), blk, npad,
                                 val, bp, np, bq, nq, two);
      auto apply = [&](const double *F, double *G, const double *Vb,
                       bool both) {
        contract_pair(F, G, npad, Vb, bp, np, bq, nq, both);
      };
      const size_t sz = static_cast<size_t>(nab) * npad;
      for (int d = 0; d < 3; d++) {
        const double *dA = val + (1 + d) * sz, *dB = val + (4 + d) * sz;
        const int sc = 3 * slot[C] + d, sd = 3 * slot[Dt] + d;
        if (two) {
          apply(U.data(), YU.col(sc).data(), dA, true);
          apply(V.data(), YV.col(sc).data(), dA, true);
          apply(U.data(), YU.col(sd).data(), dB, true);
          apply(V.data(), YV.col(sd).data(), dB, true);
        } else {
          t.dsum.resize(sz);
          for (size_t i = 0; i < sz; i++)
            t.dsum[i] = dA[i] + dB[i];
          apply(U.data(), YU.col(sc).data(), t.dsum.data(), false);
          apply(V.data(), YV.col(sc).data(), t.dsum.data(), false);
        }
      }
      for (int d = 0; d < 3; d++)
        for (int f = 0; f < 3; f++) {
          const double aa = s21[sym3[d][f]], ab = s21[6 + 3 * d + f],
                       ba = s21[6 + 3 * f + d], bb = s21[15 + sym3[d][f]];
          if (two) {
            Hf(3 * C + d, 3 * C + f) += aa;
            Hf(3 * C + d, 3 * Dt + f) += ab;
            Hf(3 * Dt + f, 3 * C + d) += ab;
            Hf(3 * Dt + d, 3 * Dt + f) += bb;
          } else {
            Hf(3 * C + d, 3 * C + f) += aa + ab + ba + bb;
          }
        }
    }
    YU = -YU;
    YV = -YV;

    // A u = (A C_o) psi point by point, and so on for v, u^k, v^k and
    // D Q^X^T phi: one product (nbf x no) (no x nk) per point, and C_o^T
    // times it alongside. Columns: u, v, u^k (nb), v^k (nb), xi^X (n3), as
    // npad x nbf (npad x no) blocks.
    const int nk = 2 + 2 * nb + n3;
    Mat Gall(blk, nk), GallCo(oblk, nk);
    {
      // Eight points at a time, so the results go out to the point-fastest
      // layout a cache line at a time.
      constexpr int G8 = 8;
      Mat Cg(nbf, no), Pg(no, nk), Mg(no, no);
      Mat Og(static_cast<Eigen::Index>(nbf) * nk, G8),
          OCg(static_cast<Eigen::Index>(no) * nk, G8);
      for (int g0 = 0; g0 < npad; g0 += G8) {
        for (int i = 0; i < G8; i++) {
          const int g = g0 + i;
          for (int j = 0; j < no; j++)
            for (int mu = 0; mu < nbf; mu++)
              Cg(mu, j) = GC(g + static_cast<Eigen::Index>(npad) * mu, j);
          Pg.col(0) = psi.row(g).transpose();
          Pg.col(1) = chi.row(g).transpose();
          for (int k = 0; k < nb; k++) {
            Pg.col(2 + k) = cblock(psik, k).row(g).transpose();
            Pg.col(2 + nb + k) = cblock(chik, k).row(g).transpose();
          }
          Pg.rightCols(n3) = Eigen::Map<const Mat, 0, Eigen::InnerStride<>>(
              xi.data() + g, no, n3, Eigen::InnerStride<>(npad));
          Eigen::Map<Mat>(Og.col(i).data(), nbf, nk).noalias() = Cg * Pg;
          Mg.noalias() = Co.transpose() * Cg;
          Eigen::Map<Mat>(OCg.col(i).data(), no, nk).noalias() = Mg * Pg;
        }
        for (Eigen::Index r = 0; r < Og.rows(); r++)
          for (int i = 0; i < G8; i++)
            Gall.data()[r * npad + g0 + i] = Og(r, i);
        for (Eigen::Index r = 0; r < OCg.rows(); r++)
          for (int i = 0; i < G8; i++)
            GallCo.data()[r * npad + g0 + i] = OCg(r, i);
      }
    }
    const auto GU = cblock(Gall, 0), GV = cblock(Gall, 1);
    auto flat = [&](const Mat &M, int c0, int n, int width) {
      return Eigen::Map<const Mat>(M.col(c0).data(), npad,
                                   static_cast<Eigen::Index>(width) * n);
    };
    const auto GUk = flat(Gall, 2, nb, nbf), GVk = flat(Gall, 2 + nb, nb, nbf);
    const auto GUkCo = flat(GallCo, 2, nb, no),
               GVkCo = flat(GallCo, 2 + nb, nb, no);
    Mat YUCo(npad, static_cast<Eigen::Index>(no) * nt),
        YVCo(npad, static_cast<Eigen::Index>(no) * nt);
    for (int l = 0; l < nt; l++) {
      YUCo.middleCols(l * no, no).noalias() = cblock(YU, l) * Co;
      YVCo.middleCols(l * no, no).noalias() = cblock(YV, l) * Co;
    }

    // The integrand e = u.A v + phi^T B phi per point, and its first
    // derivatives with points fixed: the basis part through the rows of
    // H1 = GV D + GU D Q^T + 2 phi B, the integral part u.(dA/dX) v.
    const Mat PB = phi * B;
    const Mat H1 = GV * D + GU * DQt + 2.0 * PB;
    const Vec e = (U.cwiseProduct(GV) + phi.cwiseProduct(PB)).rowwise().sum();
    Mat Ef = Mat::Zero(npad, n3);
    for (size_t i = 0; i < bl.size(); i++)
      for (int a = 0; a < 3; a++)
        for (int mu : atom_bfs[bl[i]])
          Ef.col(3 * bl[i] + a) -= dphi[a].col(mu).cwiseProduct(H1.col(mu));
    for (size_t j = 0; j < touched.size(); j++)
      for (int d = 0; d < 3; d++)
        Ef.col(3 * touched[j] + d) +=
            U.cwiseProduct(cblock(YV, 3 * j + d)).rowwise().sum();

    // Second derivatives, points fixed: basis x basis on different functions
    // (u^k . A v^l, both orders, = psi^k . C_o^T A v^l), basis x integral
    // (u^k . dA v + v^k . dA u), the same function twice (H1 against second
    // derivatives of phi) and B's cross term 2 sum_g w d phi^T B d phi.
    Mat wpsik = psik, wchik = chik;
    for (int k = 0; k < nb; k++) {
      block(wpsik, k).array().colwise() *= w.array();
      block(wchik, k).array().colwise() *= w.array();
    }
    auto gidx = [&](int k) { return 3 * bl[k / 3] + k % 3; };
    auto tidx = [&](int k) { return 3 * touched[k / 3] + k % 3; };
    {
      const auto flat_o = [&](const auto &M, int n) {
        return Eigen::Map<const Mat>(M.data(), oblk, n);
      };
      const Mat M1 = wpsik.transpose() * flat_o(GVkCo, nb);
      const Mat M2 = wpsik.transpose() * flat_o(YVCo, nt) +
                     wchik.transpose() * flat_o(YUCo, nt);
      for (int k = 0; k < nb; k++) {
        for (int l = 0; l < nb; l++)
          Hf(gidx(k), gidx(l)) += M1(k, l) + M1(l, k);
        for (int l = 0; l < nt; l++) {
          Hf(gidx(k), tidx(l)) += M2(k, l);
          Hf(tidx(l), gidx(k)) += M2(k, l);
        }
      }
    }
    for (int a = 0; a < 3; a++)
      for (int b = a; b < 3; b++) {
        const Vec c =
            (ao.second(a, b).cwiseProduct(H1.topRows(npt)).array().colwise() *
             w.head(npt).array())
                .colwise()
                .sum()
                .transpose();
        for (int A : bl) {
          double s = 0.0;
          for (int mu : atom_bfs[A])
            s += c(mu);
          Hf(3 * A + a, 3 * A + b) += s;
          if (b != a)
            Hf(3 * A + b, 3 * A + a) += s;
        }
      }
    for (int k = 0; k < nb; k++) {
      const Mat wPB = wscale(Pa[k] * rows_of(B, atom_bfs[bl[k / 3]]));
      for (int l = 0; l < nb; l++) {
        const auto &fns = atom_bfs[bl[l / 3]];
        double s = 0.0;
        for (size_t j = 0; j < fns.size(); j++)
          s += wPB.col(fns[j]).dot(Pa[l].col(j));
        Hf(gidx(k), gidx(l)) += 2.0 * s;
      }
    }
    move_points_rows(Hf, P);
    move_points_cols(Hf, P);
    t.H += Hf;
    move_points_cols(Ef, P);

    // Partition weights: first derivatives against the integrand's.
    Mat dW = Mat::Zero(npad, n3);
    dW.topRows(npt) = dwdr.batch(pts, w.head(npt), P);
    const Mat cross = dW.transpose() * Ef;
    t.H += cross + cross.transpose();
    edens.segment(bt.global, npt) = e.head(npt);

    // Fock derivatives, points fixed, then with the points moving; each kind
    // of term for every coordinate in one product.
    const Mat phiCo = phi * Co, phiQCo = phi * QCo;
    const Mat wGV = wscale(GV), wGU = wscale(GU);
    const Mat wphiCo = wscale(phiCo), wphiQCo = wscale(phiQCo);
    const Mat wGVCo = wGV * Co, wGUCo = wGU * Co;
    std::vector<int> bslot(natom, -1);
    for (size_t i = 0; i < bl.size(); i++)
      bslot[bl[i]] = static_cast<int>(i);
    // Per coordinate (atoms with functions or integrals here), A v^k + dA v
    // and A u^k + dA u, and the same times C_o.
    std::vector<int> coords;
    for (int A = 0; A < natom; A++)
      if (bslot[A] >= 0 || slot[A] >= 0)
        for (int a = 0; a < 3; a++)
          coords.push_back(3 * A + a);
    const int nx = static_cast<int>(coords.size());
    Mat TV = Mat::Zero(npad, static_cast<Eigen::Index>(nbf) * nx),
        TU = Mat::Zero(npad, static_cast<Eigen::Index>(nbf) * nx),
        TVCo = Mat::Zero(npad, static_cast<Eigen::Index>(no) * nx),
        TUCo = Mat::Zero(npad, static_cast<Eigen::Index>(no) * nx);
    for (int c = 0; c < nx; c++) {
      const int A = coords[c] / 3, a = coords[c] % 3;
      auto tv = TV.middleCols(c * nbf, nbf), tu = TU.middleCols(c * nbf, nbf);
      auto tvo = TVCo.middleCols(c * no, no), tuo = TUCo.middleCols(c * no, no);
      if (bslot[A] >= 0) {
        const int k = 3 * bslot[A] + a;
        tv += GVk.middleCols(k * nbf, nbf);
        tu += GUk.middleCols(k * nbf, nbf);
        tvo += GVkCo.middleCols(k * no, no);
        tuo += GUkCo.middleCols(k * no, no);
      }
      if (slot[A] >= 0) {
        const int l = 3 * slot[A] + a;
        tv += cblock(YV, l);
        tu += cblock(YU, l);
        tvo += YVCo.middleCols(l * no, no);
        tuo += YUCo.middleCols(l * no, no);
      }
    }
    // (A v^X)^T w phi C_o in rows c nbf.., phi^T w (A v^X C_o) in columns
    // c no.., and so on for u (with Q C_o)
    const Mat VT = TV.transpose() * wphiCo, UT = TU.transpose() * wphiQCo;
    const Mat VP = phi.transpose() * wscale(TVCo),
              UP = phi.transpose() * wscale(TUCo);
    Mat sum[3];
    for (int x = 0; x < 3; x++)
      sum[x] = Mat::Zero(nbf, nf);
    Mat f(nbf, nf);
    for (int c = 0; c < nx; c++) {
      const int A = coords[c] / 3, a = coords[c] % 3;
      f.leftCols(no) =
          0.5 * (VP.middleCols(c * no, no) + VT.middleRows(c * nbf, nbf));
      f.middleCols(no, no) = UP.middleCols(c * no, no);
      f.rightCols(no) = UT.middleRows(c * nbf, nbf);
      if (bslot[A] >= 0) {
        // the basis functions differentiated themselves
        const int k = 3 * bslot[A] + a;
        const auto &fns = atom_bfs[A];
        const Mat r0 = 0.5 * Pa[k].transpose() * wGVCo;
        const Mat r1 = Pa[k].transpose() * wGUCo;
        for (size_t j = 0; j < fns.size(); j++) {
          f.row(fns[j]).head(no) += r0.row(j);
          f.row(fns[j]).segment(no, no) += r1.row(j);
        }
        f.leftCols(no) += 0.5 * wGV.transpose() * cblock(psik, k);
        f.rightCols(no) += wGU.transpose() * cblock(chik, k);
      }
      t.F[coords[c]] += f;
      sum[a] += f;
    }
    for (int x = 0; x < 3; x++)
      t.F[3 * P + x] -= sum[x];

    // Partition weights, for every coordinate at once: with the rank-two
    // per-point terms, sum_g dW_gX (phi_g a_g^T + b_g c_g^T) = phi^T (dW_X a)
    // + b^T (dW_X c).
    const Mat GVCo = GV * Co, GUCo = GU * Co;
    Mat R1(npad, n3 * 2 * no), R2(npad, n3 * no), R3(npad, n3 * no);
    for (int X = 0; X < n3; X++) {
      const auto s = dW.col(X).array();
      R1.middleCols(2 * no * X, no) = (0.5 * GVCo).array().colwise() * s;
      R1.middleCols(2 * no * X + no, no) = GUCo.array().colwise() * s;
      R2.middleCols(no * X, no) = (0.5 * phiCo).array().colwise() * s;
      R3.middleCols(no * X, no) = phiQCo.array().colwise() * s;
    }
    const Mat O1 = phi.transpose() * R1, O2 = GV.transpose() * R2,
              O3 = GU.transpose() * R3;
    // The projector's derivative in the SCF exchange, sym(K'[D Q^X^T]) C_o:
    // with A D Q^X^T phi = (A C_o) xi^X, 1/2 sum_g w [phi (C_o^T A C_o xi^X)^T
    // + (A C_o xi^X) psi^T].
    const Mat XP = phi.transpose() *
                   wscale(flat(GallCo, 2 + 2 * nb, n3, no));
    const Mat XT = flat(Gall, 2 + 2 * nb, n3, nbf).transpose() * wscale(psi);
    for (int X = 0; X < n3; X++) {
      auto &F = t.F[X];
      F.leftCols(no) += O1.middleCols(2 * no * X, no) +
                        O2.middleCols(no * X, no) +
                        0.5 * (XP.middleCols(X * no, no) +
                               XT.middleRows(X * nbf, nbf));
      F.middleCols(no, no) += O1.middleCols(2 * no * X + no, no);
      F.rightCols(no) += O3.middleCols(no * X, no);
    }
  });

  Mat H = Mat::Zero(n3, n3);
  std::vector<Mat> F(n3, Mat::Zero(nbf, nf));
  for (const auto &t : tls) {
    H += t.H;
    for (int X = 0; X < n3; X++)
      F[X] += t.F[X];
  }
  H += occ::numint::becke_weight_hessian(grid.points, grid.weights,
                                         atom_blocks, edens, positions,
                                         m_grid.atomic_radii(), ta);

  SecondDerivatives result;
  result.fock.resize(n3);
  result.energy_fock.resize(n3);
  for (int X = 0; X < n3; X++) {
    // Q^X^T K'[D] C_o = S^X N_o - Q^T S_num^X N_o
    const Mat QxtKd = SNo[X] - Q.transpose() * SnY[X].rightCols(no);
    result.fock[X] = F[X].leftCols(no);
    result.energy_fock[X] =
        0.5 * result.fock[X] +
        0.25 * (QxtKd + Kd.transpose() * QxCo[X] +
                Q.transpose() * F[X].middleCols(no, no) + F[X].rightCols(no));
  }
  for (int X = 0; X < n3; X++)
    for (int Y = 0; Y < n3; Y++)
      H(X, Y) += QxCo[Y].cwiseProduct(F[X].middleCols(no, no)).sum() +
                 QxCo[X].cwiseProduct(F[Y].middleCols(no, no)).sum() -
                 QxCo[X].cwiseProduct(SnY[Y].rightCols(no)).sum() -
                 QxCo[Y].cwiseProduct(SnY[X].rightCols(no)).sum();
  const Mat N = No * Co.transpose();
  H += m_engine.one_electron_hessian(Mat::Zero(nbf, nbf),
                                     -0.5 * (N + N.transpose()));
  result.hessian = -0.5 * (H + H.transpose());
  occ::timing::stop(occ::timing::category::cosx);
  return result;
}

} // namespace occ::qm::cosx
