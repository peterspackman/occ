// Nuclear gradient of the COSX (seminumerical) exchange energy.
//
// With X_g = sqrt(w_g) phi(r_g), the projector Q = S_num^{-1} S and total
// density P, the restricted build gives K = (K' + K'^T)/4 with
// K' = sum_g X_g^T (A(g) X_g Q P), so the exchange energy -tr(D K) is
//
//   E_x = -1/4 sum_g w_g  v_g . A(g) u_g,   u = P phi,  v = (Q P)^T phi,
//
// A(g)_{pq} = (pq | 1/|r - g|) the shell-pair ESP at the grid point. Every
// dependence on the nuclei is differentiated:
//
//  1. basis functions in u and v                 (grad phi, like XC)
//  2. A(g) through its basis centres              (l +/- 1 shifted shells)
//  3. the grid point moving with its parent atom  (translational invariance:
//     minus the sum of 1 and 2 over that point's group, to the parent)
//  4. the partition weights                       (Becke weight derivative)
//  5. the projector Q = S_num^{-1} S              (dS analytic; dS_num is a
//     grid quantity with its own basis, point and weight terms)
//
// The centre derivatives of A(g) come from one Hermite expansion one order
// higher per shell pair (ESPEvaluator derivative pairs), all six (both
// centres, three directions) from a single evaluation, in the pair's own
// (spherical or Cartesian) functions.
//
// Unrestricted, E_x = -1/2 sum_s tr(P^s K[P^s]) is the same expression per
// spin density P^s with twice the prefactor; the integrals A(g) and their
// centre derivatives are shared by both spins.
//
// Note the COSX Fock (K symmetrised, Q on one side) is not exactly dE_x/dD,
// so the SCF is not fully variational in COSX and this fixed-density gradient
// differs from finite differences of the COSX energy by that residual.

#include <Eigen/Cholesky>
#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/gto/gto.h>
#include <occ/numint/grid_utils.h>
#include <occ/qm/seminumerical_exchange.h>
#include "detail/cosx_grid.h"

namespace occ::qm::cosx {

Mat3N SemiNumericalExchange::compute_K_gradient(const qm::MolecularOrbitals &mo,
                                                double omega,
                                                double threshold) const {
  if (mo.kind == SpinorbitalKind::General)
    throw std::runtime_error(
        "COSX exchange gradient: general spinorbitals not supported");
  if (!m_use_esp)
    throw std::runtime_error("COSX exchange gradient requires ESP mode");
  occ::timing::start(occ::timing::category::cosx);
  prepare_screening();

  const auto &shells = m_basis.shells();
  const auto &first_bf = m_basis.first_bf();
  const auto &shell_atom = m_basis.shell_to_atom();
  const auto &bf_atom = m_basis.bf_to_atom();
  const int nbf = static_cast<int>(m_basis.nbf());
  const int natom = static_cast<int>(m_atoms.size());
  const bool spherical = m_basis.is_pure();

  // Total density (restricted) or spin densities (unrestricted), and the
  // prefactor of each one's -1/4 sum_g w v . A u.
  std::vector<Mat> P;
  if (mo.kind == SpinorbitalKind::Restricted)
    P = {2.0 * mo.D};
  else
    P = {2.0 * block::a(mo.D), 2.0 * block::b(mo.D)};
  const int nspin = static_cast<int>(P.size());
  const double ef = -0.25 * (nspin == 1 ? 1.0 : 2.0);
  std::vector<Mat> QP(nspin), QPt(nspin);
  for (int sp = 0; sp < nspin; sp++) {
    QP[sp] = m_overlap_projector * P[sp];
    QPt[sp] = QP[sp].transpose();
  }

  // Derivative ESP pairs, one per significant pair (index e), the same shells and
  // coefficients as the evaluator's own.
  const size_t npairs = m_significant_pairs.size();
  occ::ints::ESPEvaluator<double> desp(occ::ints::boys().table());
  desp.resize_derivative_pairs(npairs);
  occ::parallel::parallel_for(size_t(0), npairs, [&](size_t e) {
    const auto [pp, qq] = m_significant_pairs[e];
    const auto &sp = shells[pp], &sq = shells[qq];
    desp.set_shell_pair_derivative(
        e, sp.l, sq.l, sp.num_primitives(), sq.num_primitives(),
        sp.exponents.data(), sq.exponents.data(),
        sp.contraction_coefficients.col(0).data(),
        sq.contraction_coefficients.col(0).data(), sp.origin.data(),
        sq.origin.data(), spherical);
  });

  // Batches of one atom's points, so every batch has one parent.
  std::vector<std::pair<size_t, size_t>> atom_blocks;
  const auto batches = detail::atom_batches(m_atom_grids, 128, &atom_blocks);
  const auto &grid = grid_batches();
  Vec energy_density = Vec::Zero(grid.weights.size());

  // Integrals come from the point-fastest kernel the Fock build uses,
  // X[ab * npad + g], so every contraction below is a loop over points.
  constexpr int PAD = occ::ints::ESPEvaluator<double>::soa_padding;
  struct TL {
    Mat3N grad;
    Mat M;
    std::vector<double> x, y, z, A, W;
  };
  occ::parallel::thread_local_storage<TL> tls([&]() {
    return TL{Mat3N::Zero(3, natom), Mat::Zero(nbf, nbf)};
  });

  occ::parallel::parallel_for(size_t(0), batches.size(), [&](size_t bi) {
    const auto &bt = batches[bi];
    auto &t = tls.local();
    const auto &ag = m_atom_grids[bt.atom];
    const auto pts = ag.points.middleCols(bt.start, bt.count);
    const Vec w = ag.weights.segment(bt.start, bt.count);
    const Eigen::Index npt = bt.count;
    const int npad = static_cast<int>((npt + PAD - 1) / PAD * PAD);
    detail::soa_coordinates(pts, npad, t.x, t.y, t.z);

    occ::gto::GTOValues ao;
    occ::gto::evaluate_basis(m_basis, pts, ao, 1);
    std::vector<Mat> U(nspin), V(nspin), Uw(nspin), Vw(nspin), Ga(nspin),
        Gb(nspin);
    Vec max_uv = Vec::Zero(shells.size());
    for (int sp = 0; sp < nspin; sp++) {
      U[sp] = ao.phi * P[sp];  // rows u_g
      V[sp] = ao.phi * QP[sp]; // rows v_g
      Ga[sp] = Mat::Zero(npt, nbf);
      Gb[sp] = Mat::Zero(npt, nbf);
      // u and v weighted by the quadrature weight, for the centre-derivative
      // term sum_g w_g sum_ab W_ab(g) dA_ab, W = v_a u_b + u_a v_b.
      Uw[sp] = U[sp].array().colwise() * w.array();
      Vw[sp] = V[sp].array().colwise() * w.array();
      for (size_t s = 0; s < shells.size(); s++) {
        const Eigen::Index b = first_bf[s], n = shells[s].size();
        max_uv(s) = std::max({max_uv(s),
                              U[sp].middleCols(b, n).cwiseAbs().maxCoeff(),
                              V[sp].middleCols(b, n).cwiseAbs().maxCoeff()});
      }
    }
    Mat3N g_batch = Mat3N::Zero(3, natom);
    const GridBatchInfo sphere = compute_batch_info(pts);

    for (size_t e = 0; e < npairs; e++) {
      const auto [ps, qs] = m_significant_pairs[e];
      const int p = static_cast<int>(ps), q = static_cast<int>(qs);
      const double bound = detail::pair_charge_bound(
          m_pair_charge(e), m_pair_centers.col(e), m_pair_far.col(e).data(),
          sphere);
      if (bound * max_uv(p) * max_uv(q) < threshold)
        continue;
      const int np = shells[p].size(), nq = shells[q].size();
      const int bp = first_bf[p], bq = first_bf[q];
      const int nab = np * nq;

      // Weights of the centre-derivative term, W_ab = w (v_a u_b + u_a v_b)
      // summed over spins, zero on the padding points.
      const int ldw = npad;
      t.W.assign(static_cast<size_t>(nab) * ldw, 0.0);
      for (int sp = 0; sp < nspin; sp++)
        for (int a = 0; a < np; a++) {
          const double *va = Vw[sp].col(bp + a).data(),
                       *ua = Uw[sp].col(bp + a).data();
          for (int b = 0; b < nq; b++) {
            const double *ub = U[sp].col(bq + b).data(),
                         *vb = V[sp].col(bq + b).data();
            double *Wab = t.W.data() + static_cast<size_t>(a * nq + b) * ldw;
            for (Eigen::Index g = 0; g < npt; g++)
              Wab[g] += va[g] * ub[g] + ua[g] * vb[g];
          }
        }

      // One Hermite pass gives A(g) for the pair and the contraction of its
      // centre derivatives with W; the derivative integrals stay in registers.
      t.A.resize(static_cast<size_t>(nab) * npad);
      double s6[6] = {0, 0, 0, 0, 0, 0};
      desp.evaluate_soa_value_and_derivative_contraction(
          e, t.x.data(), t.y.data(), t.z.data(), npad, t.A.data(), npad,
          t.W.data(), ldw, (p != q) ? 2 : 1, s6, omega);
      for (int x = 0; x < 3; x++) {
        g_batch(x, shell_atom[p]) += ef * s6[x];
        if (p != q)
          g_batch(x, shell_atom[q]) += ef * s6[3 + x];
      }

      // a = A u, b = A v (A symmetric over the full index range).
      for (int sp = 0; sp < nspin; sp++)
        for (int a = 0; a < np; a++) {
          const double *up = U[sp].col(bp + a).data(),
                       *vp = V[sp].col(bp + a).data();
          double *gap = Ga[sp].col(bp + a).data(),
                 *gbp = Gb[sp].col(bp + a).data();
          for (int b = 0; b < nq; b++) {
            const double *Ab = t.A.data() + (a * nq + b) * npad;
            double *gaq = Ga[sp].col(bq + b).data(),
                   *gbq = Gb[sp].col(bq + b).data();
            for (Eigen::Index g = 0; g < npt; g++) {
              gaq[g] += up[g] * Ab[g];
              gbq[g] += vp[g] * Ab[g];
            }
            if (p != q) {
              const double *uq = U[sp].col(bq + b).data(),
                           *vq = V[sp].col(bq + b).data();
              for (Eigen::Index g = 0; g < npt; g++) {
                gap[g] += uq[g] * Ab[g];
                gbp[g] += vq[g] * Ab[g];
              }
            }
          }
        }
    }

    // Basis functions in u and v: -ef sum_g w grad phi_mu (P b + QP a)_mu.
    Mat H = Mat::Zero(npt, nbf);
    for (int sp = 0; sp < nspin; sp++)
      H.noalias() -= ef * (Gb[sp] * P[sp] + Ga[sp] * QPt[sp]);
    H = H.array().colwise() * w.array();
    for (int mu = 0; mu < nbf; mu++) {
      const int A = bf_atom[mu];
      g_batch(0, A) += H.col(mu).dot(ao.phi_x.col(mu));
      g_batch(1, A) += H.col(mu).dot(ao.phi_y.col(mu));
      g_batch(2, A) += H.col(mu).dot(ao.phi_z.col(mu));
    }
    // Points move with their parent atom.
    g_batch.col(bt.atom) -= g_batch.rowwise().sum();
    t.grad += g_batch;

    // Energy density per unit weight, and dE/dQ.
    auto edens = energy_density.segment(bt.global, npt);
    edens.setZero();
    for (int sp = 0; sp < nspin; sp++) {
      edens += ef * (V[sp].array() * Ga[sp].array()).rowwise().sum().matrix();
      t.M.noalias() +=
          ef * ao.phi.transpose() *
          ((Ga[sp] * P[sp]).array().colwise() * w.array()).matrix();
    }
  });

  Mat3N grad = Mat3N::Zero(3, natom);
  Mat M = Mat::Zero(nbf, nbf);
  for (const auto &t : tls) {
    grad += t.grad;
    M += t.M;
  }

  // Projector: dE = tr(M^T dQ), dQ = S_num^{-1} (dS - dS_num Q).
  const auto Sn_ldlt = m_numerical_overlap.ldlt();
  const Mat N1 = Sn_ldlt.solve(M);
  const Mat N2 = Sn_ldlt.solve(M * m_overlap_projector.transpose());
  const Mat N1s = N1 + N1.transpose(), N2s = N2 + N2.transpose();

  // Analytic overlap: dS_mn/dR_B = -[(grad m|n) + (grad n|m)] for m, n on B.
  {
    const MatTriple Sg = m_engine.one_electron_operator_grad(
        qm::IntegralEngine::Op::overlap);
    for (int mu = 0; mu < nbf; mu++)
      for (int x = 0; x < 3; x++)
        grad(x, bf_atom[mu]) -= Sg[x].row(mu).dot(N1s.row(mu));
  }

  // Numerical overlap: -tr(N2 dS_num), S_num = sum_g w phi phi^T.
  occ::parallel::thread_local_storage<Mat3N> tls2(
      [natom]() { return Mat3N(Mat3N::Zero(3, natom)); });
  occ::parallel::parallel_for(size_t(0), batches.size(), [&](size_t bi) {
    const auto &bt = batches[bi];
    const auto &ag = m_atom_grids[bt.atom];
    const auto pts = ag.points.middleCols(bt.start, bt.count);
    const Vec w = ag.weights.segment(bt.start, bt.count);
    occ::gto::GTOValues ao;
    occ::gto::evaluate_basis(m_basis, pts, ao, 1);
    // rows w_g (N2 + N2^T) phi_g
    const Mat Y = (ao.phi * N2s).array().colwise() * w.array();
    Mat3N g_batch = Mat3N::Zero(3, natom);
    for (int mu = 0; mu < nbf; mu++) {
      const int A = bf_atom[mu];
      g_batch(0, A) += Y.col(mu).dot(ao.phi_x.col(mu));
      g_batch(1, A) += Y.col(mu).dot(ao.phi_y.col(mu));
      g_batch(2, A) += Y.col(mu).dot(ao.phi_z.col(mu));
    }
    g_batch.col(bt.atom) -= g_batch.rowwise().sum();
    tls2.local() += g_batch;
    const Mat Z = ao.phi * N2;
    energy_density.segment(bt.global, bt.count) -=
        (ao.phi.array() * Z.array()).rowwise().sum().matrix();
  });
  for (const auto &t : tls2)
    grad += t;

  // Partition weights.
  grad += occ::numint::becke_weight_gradient(
      grid.points, grid.weights, atom_blocks, energy_density,
      detail::atom_positions(m_atoms),
      m_grid.atomic_radii(), m_grid.settings().treutler_alrichs_adjustment);

  occ::timing::stop(occ::timing::category::cosx);
  return grad;
}

} // namespace occ::qm::cosx
