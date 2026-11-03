// Linear response of the Kohn-Sham Fock matrix, for CPKS.
//
// The XC part is the exact derivative of occ's own V_xc expression, so the
// response is consistent with the SCF Fock matrix by construction. For the
// restricted GGA potential
//
//   V_xc = K + K^T,  K = phi^T (1/2 v_rho phi + 2 v_sigma grad rho . grad phi)
//
// the response to a first-order (total) density rho1 is
//
//   V1 = K1 + K1^T,
//   K1 = phi^T (1/2 dv_rho phi + 2 dv_sigma grad rho . grad phi
//               + 2 v_sigma grad rho1 . grad phi),
//   dv_rho   = f_rho,rho rho1 + f_rho,sigma sigma1,
//   dv_sigma = f_rho,sigma rho1 + f_sigma,sigma sigma1,
//   sigma1   = 2 grad rho . grad rho1,
//
// and for LDA V1 = phi^T (f_rho,rho rho1) phi. A meta-GGA adds
//
//   K += 1/4 v_tau grad phi^T grad phi,  tau = 1/2 sum_i |grad psi_i|^2,
//
// so f_rho,tau tau1 and f_sigma,tau tau1 join dv_rho and dv_sigma, and
//
//   K1 += 1/4 grad phi^T (dv_tau grad phi),
//   dv_tau = f_rho,tau rho1 + f_sigma,tau sigma1 + f_tau,tau tau1.
//
// All quantities are for the unpolarized functional at the total density.

#include <occ/core/log.h>
#include <occ/core/parallel.h>
#include <occ/core/timings.h>
#include <occ/dft/dft.h>
#include <occ/dft/dft_kernels.h>
#include <occ/gto/density.h>
#include <occ/qm/cphf.h>
#include "detail/xc_kernel.h"

namespace occ::dft {

namespace {

template <int deriv>
void xc_response_block(const occ::gto::AOBasis &basis, const Mat &D0,
                       const std::vector<Mat> &D1,
                       const std::vector<DensityFunctional> &funcs,
                       Eigen::Ref<const Mat3N> pts, const Vec &w,
                       double density_threshold, std::vector<Mat> &out) {
  const Eigen::Index npt = pts.cols();
  auto gto = occ::gto::evaluate_basis(basis, pts, deriv);
  const int ncomp = occ::density::num_components(deriv);
  Mat rho(npt, ncomp);
  occ::density::evaluate_density<deriv, SpinorbitalKind::Restricted>(D0, gto,
                                                                     rho);
  if (rho.col(0).maxCoeff() < density_threshold || funcs.empty())
    return;
  const auto f = detail::second_derivatives<SpinorbitalKind::Restricted, deriv>(
      funcs, rho);
  const Vec wt =
      detail::screened_weights(w, rho.col(0).array(), density_threshold)
          .matrix();

  const auto &phi = gto.phi;
  Mat rho1(npt, ncomp);
  for (size_t k = 0; k < D1.size(); k++) {
    occ::density::evaluate_density<deriv, SpinorbitalKind::Restricted>(
        2.0 * D1[k], gto, rho1);
    if constexpr (deriv == 0) {
      const Array c = wt.array() * f.v2rho2.col(0).array() * rho1.col(0).array();
      out[k].noalias() += phi.transpose() * (phi.array().colwise() * c).matrix();
    } else {
      const Array dx = rho.col(1).array(), dy = rho.col(2).array(),
                  dz = rho.col(3).array();
      const Array r1 = rho1.col(0).array(), d1x = rho1.col(1).array(),
                  d1y = rho1.col(2).array(), d1z = rho1.col(3).array();
      const Array sigma1 = 2.0 * (dx * d1x + dy * d1y + dz * d1z);
      const Array frr = f.v2rho2.col(0).array(), frs = f.v2rhosigma.col(0).array(),
                  fss = f.v2sigma2.col(0).array(), vs = f.vsigma.col(0).array();
      const Array W = wt.array();
      Array dvr = W * (frr * r1 + frs * sigma1);
      Array dvs = W * (frs * r1 + fss * sigma1);
      Array dvt;
      if constexpr (deriv == 2) {
        const Array t1 = rho1.col(5).array();
        const Array frt = f.v2rhotau.col(0).array(),
                    fst = f.v2sigmatau.col(0).array(),
                    ftt = f.v2tau2.col(0).array();
        dvr += W * frt * t1;
        dvs += W * fst * t1;
        dvt = W * (frt * r1 + fst * sigma1 + ftt * t1);
      }
      const Array vsw = W * vs;
      // coefficient of grad phi, per direction
      const Array cx = 2.0 * (dvs * dx + vsw * d1x);
      const Array cy = 2.0 * (dvs * dy + vsw * d1y);
      const Array cz = 2.0 * (dvs * dz + vsw * d1z);
      Mat t = (0.5 * (phi.array().colwise() * dvr) +
               gto.phi_x.array().colwise() * cx +
               gto.phi_y.array().colwise() * cy +
               gto.phi_z.array().colwise() * cz)
                  .matrix();
      Mat K1 = phi.transpose() * t;
      if constexpr (deriv == 2) {
        for (const Mat *d : {&gto.phi_x, &gto.phi_y, &gto.phi_z})
          K1.noalias() +=
              d->transpose() * (d->array().colwise() * (0.25 * dvt)).matrix();
      }
      out[k] += K1 + K1.transpose();
    }
  }
}

} // namespace

std::vector<Mat> DFT::compute_xc_response(const MolecularOrbitals &mo,
                                          const std::vector<Mat> &D1) const {
  if (mo.kind != SpinorbitalKind::Restricted)
    throw std::runtime_error("XC response: restricted only");
  if (needs_laplacian())
    throw std::runtime_error(
        "XC response: Laplacian-dependent meta-GGAs are not supported");
  const int deriv = density_derivative();
  occ::timing::start(occ::timing::category::dft_xc);
  const auto &basis = m_hf.aobasis();
  const size_t nbf = basis.nbf();
  const size_t nd = D1.size();
  const Mat D0 = 2.0 * mo.D;
  const auto &funcs = m_method.functionals;

  const auto &grid = m_grid.get_molecular_grid_points();
  const auto &pts = grid.points();
  const auto &wts = grid.weights();
  const size_t npt = pts.cols();
  constexpr size_t block = 256;
  const size_t nblocks = (npt + block - 1) / block;

  occ::parallel::thread_local_storage<std::vector<Mat>> tls(
      [nd, nbf]() { return std::vector<Mat>(nd, Mat::Zero(nbf, nbf)); });
  occ::parallel::parallel_for(size_t(0), nblocks, [&](size_t b) {
    const size_t start = b * block;
    const size_t n = std::min(block, npt - start);
    const Vec w = wts.segment(start, n);
    const auto p = pts.middleCols(start, n);
    if (deriv == 0)
      xc_response_block<0>(basis, D0, D1, funcs, p, w, m_density_threshold,
                           tls.local());
    else if (deriv == 1)
      xc_response_block<1>(basis, D0, D1, funcs, p, w, m_density_threshold,
                           tls.local());
    else
      xc_response_block<2>(basis, D0, D1, funcs, p, w, m_density_threshold,
                           tls.local());
  });
  std::vector<Mat> result(nd, Mat::Zero(nbf, nbf));
  for (const auto &t : tls)
    for (size_t k = 0; k < nd; k++)
      result[k] += t[k];
  occ::timing::stop(occ::timing::category::dft_xc);
  return result;
}

std::vector<Mat> DFT::compute_response_fock(const MolecularOrbitals &mo,
                                            const std::vector<Mat> &D1,
                                            const Mat &Schwarz) const {
  if (mo.kind != SpinorbitalKind::Restricted)
    throw std::runtime_error("DFT response Fock: restricted only");
  // Coulomb with the full-range exchange in one pass, then any attenuated
  // exchange, then the XC kernel.
  const auto ops = exchange_operators();
  double c0 = 0.0;
  for (const auto &[c, omega] : ops)
    if (omega == 0.0)
      c0 += c;
  std::vector<Mat> G =
      m_hf.compute_response_jk(D1, 1.0, c0, 0.0, Schwarz, &mo);
  for (const auto &[c, omega] : ops) {
    if (c == 0.0 || omega == 0.0)
      continue;
    const auto K = m_hf.compute_response_jk(D1, 0.0, c, omega, Schwarz, &mo);
    for (size_t k = 0; k < G.size(); k++)
      G[k] += K[k];
  }
  auto Vxc1 = compute_xc_response(mo, D1);
  for (size_t k = 0; k < G.size(); k++)
    G[k] += Vxc1[k];
  return G;
}

// ---------------------------------------------------------------------------
// Occupied-projected response: V1 C_o directly.
//
// With D1 = Cx C_o^T + C_o Cx^T (any density without a virtual-virtual block)
// and, per point, X = phi C_o, Y = phi Cx (and their gradients),
//
//   rho1 = 4 sum_i Y_i X_i,  grad rho1 = 4 sum_i (grad Y_i X_i + Y_i grad X_i),
//
// and for V1 = K1 + K1^T, K1 = phi^T t, t = 1/2 dv_rho phi + c . grad phi,
//
//   V1 C_o = phi^T [dv_rho X + c . grad X] + sum_x grad_x phi^T (c_x X),
//
// so every product is npt x nbf x nocc, batched over all densities.

namespace {

// One batch of points with its basis values (all functions, or a screened
// subset with D0, Co, Cx_all and out restricted to the same rows).
template <int deriv>
void xc_response_occ_block(const occ::gto::GTOValues &gto, const Mat &D0,
                           const Mat &Co, const Mat &Cx_all, Eigen::Index nd,
                           const std::vector<DensityFunctional> &funcs,
                           const Vec &w, double density_threshold, Mat &out) {
  const Eigen::Index npt = gto.phi.rows();
  const Eigen::Index o = Co.cols();
  const int ncomp = occ::density::num_components(deriv);
  Mat rho(npt, ncomp);
  occ::density::evaluate_density<deriv, SpinorbitalKind::Restricted>(D0, gto,
                                                                     rho);
  if (rho.col(0).maxCoeff() < density_threshold || funcs.empty())
    return;
  const auto f = detail::second_derivatives<SpinorbitalKind::Restricted, deriv>(
      funcs, rho);
  const Vec wt =
      detail::screened_weights(w, rho.col(0).array(), density_threshold)
          .matrix();
  const Array W = wt.array();

  // out is accumulated transposed, (o nd) x nl, so that scattering a
  // block's rows of functions into the global result is contiguous.
  // Densities go in groups of about 256 columns so that a group's
  // intermediates (Y, grad Y, M and the gradient coefficients) stay in cache;
  // all of them at once is tens of MB per batch and memory-bound.
  const auto &phi = gto.phi;
  const Mat X = phi * Co; // npt x o
  Mat Xg[3];
  if constexpr (deriv >= 1)
    for (int x = 0; x < 3; x++)
      Xg[x] = (x == 0 ? gto.phi_x : x == 1 ? gto.phi_y : gto.phi_z) * Co;
  const Eigen::Index kc = std::max<Eigen::Index>(1, 256 / std::max<Eigen::Index>(o, 1));
  for (Eigen::Index k0 = 0; k0 < nd; k0 += kc) {
    const Eigen::Index nk = std::min(kc, nd - k0);
    const auto Cx_c = Cx_all.middleCols(o * k0, o * nk);
    auto out_c = out.middleRows(o * k0, o * nk);
    const Mat Y = phi * Cx_c; // npt x (o nk)
    Mat M(npt, o * nk);
    if constexpr (deriv == 0) {
      const Array frr = f.v2rho2.col(0).array();
      for (Eigen::Index k = 0; k < nk; k++) {
        const Array r1 =
            4.0 * (Y.middleCols(o * k, o).array() * X.array()).rowwise().sum();
        const Array c = W * frr * r1;
        M.middleCols(o * k, o) = (X.array().colwise() * c).matrix();
      }
      out_c.noalias() += M.transpose() * phi;
    } else {
      const Mat Yg[3] = {gto.phi_x * Cx_c, gto.phi_y * Cx_c,
                         gto.phi_z * Cx_c};
      const double *dx = rho.col(1).data(), *dy = rho.col(2).data(),
                   *dz = rho.col(3).data();
      const double *frr = f.v2rho2.col(0).data(),
                   *frs = f.v2rhosigma.col(0).data(),
                   *fss = f.v2sigma2.col(0).data(), *vs = f.vsigma.col(0).data();
      const double *frt = nullptr, *fst = nullptr, *ftt = nullptr;
      if constexpr (deriv == 2) {
        frt = f.v2rhotau.col(0).data();
        fst = f.v2sigmatau.col(0).data();
        ftt = f.v2tau2.col(0).data();
      }
      const double *Wp = wt.data();
      Mat Cg[3] = {Mat(npt, o * nk), Mat(npt, o * nk), Mat(npt, o * nk)};
      // Fused per density: point sums over the occupied index (column-major,
      // so each orbital's column is contiguous), then the kernel, then the
      // coefficient columns.
      // tau1 = 2 sum_i grad Y_i . grad X_i, and dvt carries the 1/2 of the
      // tau term in V1 C_o = ... + sum_x grad_x phi^T (1/2 dv_tau grad_x X).
      std::vector<double> r1(npt), g1x(npt), g1y(npt), g1z(npt), t1, dvr(npt),
          cx(npt), cy(npt), cz(npt), dvt;
      if constexpr (deriv == 2) {
        t1.resize(npt);
        dvt.resize(npt);
      }
      for (Eigen::Index k = 0; k < nk; k++) {
        std::fill(r1.begin(), r1.end(), 0.0);
        std::fill(g1x.begin(), g1x.end(), 0.0);
        std::fill(g1y.begin(), g1y.end(), 0.0);
        std::fill(g1z.begin(), g1z.end(), 0.0);
        std::fill(t1.begin(), t1.end(), 0.0);
        for (Eigen::Index i = 0; i < o; i++) {
          const double *x0 = X.col(i).data(), *xx = Xg[0].col(i).data(),
                       *xy = Xg[1].col(i).data(), *xz = Xg[2].col(i).data();
          const Eigen::Index col = o * k + i;
          const double *y0 = Y.col(col).data(), *yx = Yg[0].col(col).data(),
                       *yy = Yg[1].col(col).data(), *yz = Yg[2].col(col).data();
          for (Eigen::Index g = 0; g < npt; g++) {
            r1[g] += y0[g] * x0[g];
            g1x[g] += yx[g] * x0[g] + y0[g] * xx[g];
            g1y[g] += yy[g] * x0[g] + y0[g] * xy[g];
            g1z[g] += yz[g] * x0[g] + y0[g] * xz[g];
          }
          if constexpr (deriv == 2)
            for (Eigen::Index g = 0; g < npt; g++)
              t1[g] += yx[g] * xx[g] + yy[g] * xy[g] + yz[g] * xz[g];
        }
        for (Eigen::Index g = 0; g < npt; g++) {
          const double r = 4.0 * r1[g], gx = 4.0 * g1x[g], gy = 4.0 * g1y[g],
                       gz = 4.0 * g1z[g];
          const double s1 = 2.0 * (dx[g] * gx + dy[g] * gy + dz[g] * gz);
          double dv_r = Wp[g] * (frr[g] * r + frs[g] * s1);
          double dv_s = Wp[g] * (frs[g] * r + fss[g] * s1);
          if constexpr (deriv == 2) {
            const double t = 2.0 * t1[g];
            dv_r += Wp[g] * frt[g] * t;
            dv_s += Wp[g] * fst[g] * t;
            dvt[g] = 0.5 * Wp[g] * (frt[g] * r + fst[g] * s1 + ftt[g] * t);
          }
          const double vsw = Wp[g] * vs[g];
          dvr[g] = dv_r;
          cx[g] = 2.0 * (dv_s * dx[g] + vsw * gx);
          cy[g] = 2.0 * (dv_s * dy[g] + vsw * gy);
          cz[g] = 2.0 * (dv_s * dz[g] + vsw * gz);
        }
        for (Eigen::Index i = 0; i < o; i++) {
          const double *x0 = X.col(i).data(), *xx = Xg[0].col(i).data(),
                       *xy = Xg[1].col(i).data(), *xz = Xg[2].col(i).data();
          const Eigen::Index col = o * k + i;
          double *m = M.col(col).data(), *c0 = Cg[0].col(col).data(),
                 *c1 = Cg[1].col(col).data(), *c2 = Cg[2].col(col).data();
          for (Eigen::Index g = 0; g < npt; g++) {
            m[g] = x0[g] * dvr[g] + xx[g] * cx[g] + xy[g] * cy[g] +
                   xz[g] * cz[g];
            c0[g] = x0[g] * cx[g];
            c1[g] = x0[g] * cy[g];
            c2[g] = x0[g] * cz[g];
          }
          if constexpr (deriv == 2)
            for (Eigen::Index g = 0; g < npt; g++) {
              c0[g] += dvt[g] * xx[g];
              c1[g] += dvt[g] * xy[g];
              c2[g] += dvt[g] * xz[g];
            }
        }
      }
      out_c.noalias() += M.transpose() * phi;
      out_c.noalias() += Cg[0].transpose() * gto.phi_x;
      out_c.noalias() += Cg[1].transpose() * gto.phi_y;
      out_c.noalias() += Cg[2].transpose() * gto.phi_z;
    }
  }
}

// Unrestricted: per spin, X^s = phi C_o^s and Y^s = phi Cx^s give
// rho1^s = 2 sum_i Y^s_i X^s_i (D1^s is the spin density), the spin kernel
// couples the two, and V1^s C_o^s follows as above with dv_rho^s and dV^s.
// Co and Cx_all are in the CPHF layout: n_alpha alpha columns, then beta.
template <int deriv>
void xc_response_occ_block_unrestricted(
    const occ::gto::GTOValues &gto, const Mat &Co, const Mat &Cx_all,
    Eigen::Index na, Eigen::Index nd,
    const std::vector<DensityFunctional> &funcs, const Vec &w,
    double density_threshold, Mat &out) {
  const Eigen::Index npt = gto.phi.rows();
  const Eigen::Index o = Co.cols();
  const Eigen::Index no[2] = {na, o - na}, c0[2] = {0, na};
  const Mat *dphi[3] = {&gto.phi_x, &gto.phi_y, &gto.phi_z};
  const int ncomp = occ::density::num_components(deriv);
  Mat X[2], Xg[2][3];
  Mat rho(2 * npt, ncomp);
  for (int s = 0; s < 2; s++) {
    X[s] = gto.phi * Co.middleCols(c0[s], no[s]);
    rho.col(0).segment(s * npt, npt) = X[s].array().square().rowwise().sum();
    if constexpr (deriv == 1)
      for (int i = 0; i < 3; i++) {
        Xg[s][i] = (*dphi[i]) * Co.middleCols(c0[s], no[s]);
        rho.col(1 + i).segment(s * npt, npt) =
            2.0 * (X[s].array() * Xg[s][i].array()).rowwise().sum();
      }
  }
  const Array rho_total =
      rho.col(0).head(npt).array() + rho.col(0).tail(npt).array();
  if (rho_total.maxCoeff() < density_threshold || funcs.empty())
    return;
  const auto f =
      detail::second_derivatives<SpinorbitalKind::Unrestricted, deriv>(funcs,
                                                                       rho);
  const Array W = detail::screened_weights(w, rho_total, density_threshold);
  const detail::SpinKernel<deriv> kernel(f, rho);

  // All densities at once, out transposed ((o nd) x nl) as above.
  const Mat Y = gto.phi * Cx_all;
  Mat Yg[3];
  if constexpr (deriv == 1)
    for (int i = 0; i < 3; i++)
      Yg[i] = (*dphi[i]) * Cx_all;
  Mat r1[2], g1[2][3], dvr[2], dV[2][3];
  for (int s = 0; s < 2; s++) {
    r1[s].resize(npt, nd);
    for (int i = 0; i < 3 && deriv == 1; i++)
      g1[s][i].resize(npt, nd);
  }
  for (Eigen::Index k = 0; k < nd; k++)
    for (int s = 0; s < 2; s++) {
      const auto Yk = Y.middleCols(o * k + c0[s], no[s]);
      r1[s].col(k) = 2.0 * (Yk.array() * X[s].array()).rowwise().sum();
      if constexpr (deriv == 1)
        for (int i = 0; i < 3; i++)
          g1[s][i].col(k) =
              2.0 * (Yg[i].middleCols(o * k + c0[s], no[s]).array() *
                         X[s].array() +
                     Yk.array() * Xg[s][i].array())
                        .rowwise()
                        .sum();
    }
  kernel.apply(r1, g1, dvr, dV);
  Mat M(npt, o * nd), Cg[3];
  if constexpr (deriv == 1)
    for (int i = 0; i < 3; i++)
      Cg[i].resize(npt, o * nd);
  for (Eigen::Index k = 0; k < nd; k++)
    for (int s = 0; s < 2; s++) {
      const Eigen::Index c = o * k + c0[s];
      const Array dv = W * dvr[s].col(k).array();
      M.middleCols(c, no[s]) = (X[s].array().colwise() * dv).matrix();
      if constexpr (deriv == 1)
        for (int i = 0; i < 3; i++) {
          const Array d = W * dV[s][i].col(k).array();
          M.middleCols(c, no[s]).array() += Xg[s][i].array().colwise() * d;
          Cg[i].middleCols(c, no[s]) = (X[s].array().colwise() * d).matrix();
        }
    }
  out.noalias() += M.transpose() * gto.phi;
  if constexpr (deriv == 1)
    for (int i = 0; i < 3; i++)
      out.noalias() += Cg[i].transpose() * (*dphi[i]);
}

} // namespace

std::vector<Mat>
DFT::compute_xc_response_occ(const MolecularOrbitals &mo,
                             const std::vector<Mat> &Cx,
                             const occ::numint::MolecularGrid *grid_in) const {
  const occ::numint::MolecularGrid &mgrid = grid_in ? *grid_in : m_grid;
  if (needs_laplacian())
    throw std::runtime_error(
        "XC response: Laplacian-dependent meta-GGAs are not supported");
  const int deriv = density_derivative();
  const bool unrestricted = mo.kind == SpinorbitalKind::Unrestricted;
  if (unrestricted && deriv > 1)
    throw std::runtime_error(
        "XC response: unrestricted meta-GGAs are not supported");
  // every spin channel's occupied columns (the CPHF layout)
  const Mat Co = qm::cphf::occupied_orbitals(mo);
  const Eigen::Index nocc = Co.cols();
  const size_t nd = Cx.size();
  const auto &basis = m_hf.aobasis();
  const Eigen::Index nbf = basis.nbf();
  Mat Cx_all(nbf, nocc * nd);
  for (size_t k = 0; k < nd; k++)
    Cx_all.middleCols(nocc * k, nocc) = Cx[k];

  occ::timing::start(occ::timing::category::dft_xc);
  const Mat D0 = unrestricted ? Mat() : Mat(2.0 * mo.D);
  const auto &funcs =
      unrestricted ? m_method.functionals_polarized : m_method.functionals;
  const auto &grid = mgrid.get_molecular_grid_points();
  const auto &pts = grid.points();
  const auto &wts = grid.weights();
  const size_t npt = pts.cols();
  constexpr size_t block = 256;
  const size_t nblocks = (npt + block - 1) / block;
  occ::parallel::thread_local_storage<Mat> tls(
      [nbf, nocc, nd]() { return Mat(Mat::Zero(nocc * nd, nbf)); });
  auto run_block = [&](const occ::gto::GTOValues &gto, const Mat &D0b,
                       const Mat &Cob, const Mat &Cxb, const Vec &w,
                       Mat &out) {
    if (unrestricted) {
      if (deriv == 0)
        xc_response_occ_block_unrestricted<0>(gto, Cob, Cxb, mo.n_alpha, nd,
                                              funcs, w, m_density_threshold,
                                              out);
      else
        xc_response_occ_block_unrestricted<1>(gto, Cob, Cxb, mo.n_alpha, nd,
                                              funcs, w, m_density_threshold,
                                              out);
    } else if (deriv == 0)
      xc_response_occ_block<0>(gto, D0b, Cob, Cxb, nd, funcs, w,
                               m_density_threshold, out);
    else if (deriv == 1)
      xc_response_occ_block<1>(gto, D0b, Cob, Cxb, nd, funcs, w,
                               m_density_threshold, out);
    else
      xc_response_occ_block<2>(gto, D0b, Cob, Cxb, nd, funcs, w,
                               m_density_threshold, out);
  };
  if (m_xc_screening_threshold > 0.0) {
    // Screened as the SCF's XC build is: each spatial leaf of points only
    // sees the shells that reach it, so this is the derivative of the
    // potential actually used, and the products shrink to those functions.
    const auto &shells = basis.shells();
    const auto &first_bf = basis.first_bf();
    const Vec extents =
        occ::gto::evaluate_decay_cutoff(basis, m_xc_screening_threshold);
    const auto &hier = mgrid.get_molecular_grid_points().get_hierarchy();
    const auto &spts = hier.sorted_points();
    const auto &swts = hier.sorted_weights();
    occ::parallel::parallel_for(size_t(0), hier.num_leaves(), [&](size_t li) {
      const auto &leaf = hier.leaves()[li];
      if (leaf.count == 0)
        return;
      std::vector<int> sig, rows;
      for (size_t sh = 0; sh < shells.size(); ++sh) {
        const double dist = (shells[sh].origin - leaf.bounds.center).norm();
        if (dist - extents(sh) < leaf.bounds.radius) {
          sig.push_back(static_cast<int>(sh));
          for (size_t k = 0; k < shells[sh].size(); k++)
            rows.push_back(first_bf[sh] + k);
        }
      }
      if (sig.empty())
        return;
      const Eigen::Index nl = rows.size();
      Mat D0b(nl, nl), Cob(nl, nocc), Cxb(nl, nocc * nd);
      for (Eigen::Index i = 0; i < nl; i++) {
        for (Eigen::Index j = 0; j < nl && !unrestricted; j++)
          D0b(i, j) = D0(rows[i], rows[j]);
        Cob.row(i) = Co.row(rows[i]);
        Cxb.row(i) = Cx_all.row(rows[i]);
      }
      const auto pts_l = spts.middleCols(leaf.offset, leaf.count);
      const Vec w = swts.segment(leaf.offset, leaf.count);
      auto gto = occ::gto::evaluate_basis_subset(basis, pts_l, deriv, sig);
      Mat outb = Mat::Zero(nocc * nd, nl);
      run_block(gto, D0b, Cob, Cxb, w, outb);
      auto &out = tls.local();
      for (Eigen::Index i = 0; i < nl; i++)
        out.col(rows[i]) += outb.col(i);
    });
  } else {
    occ::parallel::parallel_for(size_t(0), nblocks, [&](size_t b) {
      const size_t start = b * block;
      const size_t n = std::min(block, npt - start);
      const Vec w = wts.segment(start, n);
      auto gto = occ::gto::evaluate_basis(basis, pts.middleCols(start, n),
                                          deriv);
      run_block(gto, D0, Co, Cx_all, w, tls.local());
    });
  }
  Mat total = Mat::Zero(nocc * nd, nbf);
  for (const auto &t : tls)
    total += t;
  occ::timing::stop(occ::timing::category::dft_xc);
  std::vector<Mat> result(nd);
  for (size_t k = 0; k < nd; k++)
    result[k] = total.middleRows(nocc * k, nocc).transpose();
  return result;
}

std::vector<Mat> DFT::response_occ(const MolecularOrbitals &mo,
                                   const std::vector<Mat> &Cx,
                                   const Mat &Schwarz,
                                   qm::HartreeFock::ResponseMode mode) const {
  const bool approx = mode == qm::HartreeFock::ResponseMode::Approximate;
  const occ::numint::MolecularGrid *grid = nullptr;
  if (approx) {
    if (!m_coarse_grid) {
      // Coarse in both directions (the default LMG radial grid is set by
      // its precision, not a point count). Only the starting point of the
      // exact CPHF iterations depends on it.
      auto settings = m_grid.settings();
      settings.radial_precision = 1e-5;
      settings.max_angular_points = 110;
      settings.min_angular_points = 50;
      m_coarse_grid = std::make_unique<occ::numint::MolecularGrid>(
          m_hf.aobasis(), settings);
    }
    grid = m_coarse_grid.get();
  }
  auto G = m_hf.response_occ(mo, Cx, 1.0, exchange_operators(), Schwarz, mode);
  const auto V = compute_xc_response_occ(mo, Cx, grid);
  for (size_t k = 0; k < G.size(); k++)
    G[k] += V[k];
  return G;
}

std::vector<Mat> DFT::compute_response_fock_occ(const MolecularOrbitals &mo,
                                                const std::vector<Mat> &Cx,
                                                const Mat &Schwarz) const {
  return response_occ(mo, Cx, Schwarz, qm::HartreeFock::ResponseMode::Exact);
}

std::vector<Mat>
DFT::compute_response_fock_occ_approx(const MolecularOrbitals &mo,
                                      const std::vector<Mat> &Cx,
                                      const Mat &Schwarz) const {
  return response_occ(mo, Cx, Schwarz,
                      qm::HartreeFock::ResponseMode::Approximate);
}

} // namespace occ::dft
