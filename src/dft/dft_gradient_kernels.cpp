#include <occ/dft/dft_gradient_kernels.h>
#include <occ/gto/density.h>

using occ::gto::GTOValues;

namespace occ::dft::kernels {

namespace {

// The orbital part of the XC gradient,
//   dE/dR_A = -2 sum_s sum_{mu in A} sum_nu V^s_X(mu, nu) D^s(mu, nu),
// with V^s_X the derivative potential matrix built from the bra derivatives.
// Every term of V_X has the form sum_g a(g, mu) b(g, nu) for some pair of
// (weighted) basis-function columns, so with D symmetric its contraction
// with D is sum_g a(g, mu) (b D)(g, mu): a per-point, per-function quantity
// c(g, mu). Summing c over mu in A gives the orbital gradient, summing over
// mu for one point gives what moving that point with its parent atom costs.
// The only products left are (b D) for b = phi (already needed for the
// density), the GGA combination aow, and for meta-GGAs the gradient
// columns: one GEMM for a GGA in place of six nbf x nbf ones.
struct SpinTerms {
  Mat Dphi;     // phi D
  Mat Dgrad[3]; // phi_x D, phi_y D, phi_z D (meta-GGA only)
};

// Adds sum_g c(g, mu) to fsum(:, mu) and sum_mu c(g, mu) to psum(:, g) for
// one spin. vr = vrho (LDA) or vrho / 2 (GGA, meta-GGA); f = 2 vsigma_ss
// grad rho_s + vsigma_ab grad rho_s' (GGA); vt = vtau / 2 (meta-GGA).
template <int derivative_order>
void accumulate_spin(const BatchDensity &dens, int s, const GTOValues &gv,
                     const SpinTerms &t, const double *vr, const double *fx,
                     const double *fy, const double *fz, const double *vt,
                     Mat3N &fsum, Mat3N &psum, Mat &scratch) {
  const Eigen::Index npt = gv.phi.rows();
  const Eigen::Index nbf = gv.phi.cols();
  double *px_sum = psum.data();
  if constexpr (derivative_order == 0) {
    for (Eigen::Index mu = 0; mu < nbf; mu++) {
      const double *dp = t.Dphi.col(mu).data();
      const double *gx = gv.phi_x.col(mu).data();
      const double *gy = gv.phi_y.col(mu).data();
      const double *gz = gv.phi_z.col(mu).data();
      double sx = 0, sy = 0, sz = 0;
      for (Eigen::Index g = 0; g < npt; g++) {
        const double w = vr[g] * dp[g];
        const double cx = gx[g] * w, cy = gy[g] * w, cz = gz[g] * w;
        sx += cx;
        sy += cy;
        sz += cz;
        px_sum[3 * g] += cx;
        px_sum[3 * g + 1] += cy;
        px_sum[3 * g + 2] += cz;
      }
      fsum(0, mu) += sx;
      fsum(1, mu) += sy;
      fsum(2, mu) += sz;
    }
  } else {
    // aow = phi vr + sum_j phi_j f_j, B = aow D
    Mat &B = scratch;
    {
      Mat aow(npt, nbf);
      for (Eigen::Index mu = 0; mu < nbf; mu++) {
        const double *p = gv.phi.col(mu).data();
        const double *gx = gv.phi_x.col(mu).data();
        const double *gy = gv.phi_y.col(mu).data();
        const double *gz = gv.phi_z.col(mu).data();
        double *a = aow.col(mu).data();
        for (Eigen::Index g = 0; g < npt; g++)
          a[g] = p[g] * vr[g] + gx[g] * fx[g] + gy[g] * fy[g] + gz[g] * fz[g];
      }
      dens.times(aow, s, B);
    }
    for (Eigen::Index mu = 0; mu < nbf; mu++) {
      const double *b = B.col(mu).data();
      const double *dp = t.Dphi.col(mu).data();
      const double *gx = gv.phi_x.col(mu).data();
      const double *gy = gv.phi_y.col(mu).data();
      const double *gz = gv.phi_z.col(mu).data();
      const double *hxx = gv.phi_xx.col(mu).data();
      const double *hxy = gv.phi_xy.col(mu).data();
      const double *hxz = gv.phi_xz.col(mu).data();
      const double *hyy = gv.phi_yy.col(mu).data();
      const double *hyz = gv.phi_yz.col(mu).data();
      const double *hzz = gv.phi_zz.col(mu).data();
      const double *dxp = nullptr, *dyp = nullptr, *dzp = nullptr;
      if constexpr (derivative_order > 1) {
        dxp = t.Dgrad[0].col(mu).data();
        dyp = t.Dgrad[1].col(mu).data();
        dzp = t.Dgrad[2].col(mu).data();
      }
      double sx = 0, sy = 0, sz = 0;
      for (Eigen::Index g = 0; g < npt; g++) {
        double cx = gx[g] * b[g] + (gx[g] * vr[g] + hxx[g] * fx[g] +
                                    hxy[g] * fy[g] + hxz[g] * fz[g]) *
                                       dp[g];
        double cy = gy[g] * b[g] + (gy[g] * vr[g] + hxy[g] * fx[g] +
                                    hyy[g] * fy[g] + hyz[g] * fz[g]) *
                                       dp[g];
        double cz = gz[g] * b[g] + (gz[g] * vr[g] + hxz[g] * fx[g] +
                                    hyz[g] * fy[g] + hzz[g] * fz[g]) *
                                       dp[g];
        if constexpr (derivative_order > 1) {
          cx += vt[g] * (hxx[g] * dxp[g] + hxy[g] * dyp[g] + hxz[g] * dzp[g]);
          cy += vt[g] * (hxy[g] * dxp[g] + hyy[g] * dyp[g] + hyz[g] * dzp[g]);
          cz += vt[g] * (hxz[g] * dxp[g] + hyz[g] * dyp[g] + hzz[g] * dzp[g]);
        }
        sx += cx;
        sy += cy;
        sz += cz;
        px_sum[3 * g] += cx;
        px_sum[3 * g + 1] += cy;
        px_sum[3 * g + 2] += cz;
      }
      fsum(0, mu) += sx;
      fsum(1, mu) += sy;
      fsum(2, mu) += sz;
    }
  }
}

} // namespace

template <int derivative_order, SpinorbitalKind spinorbital_kind>
void process_grid_block_gradient(const BatchDensity &D,
                                 const GTOValues &gto_vals,
                                 Eigen::Ref<const Mat3N> points_block,
                                 const Vec &weights_block,
                                 const std::vector<DensityFunctional> &funcs,
                                 Mat3N &gradient, double density_threshold,
                                 const std::vector<int> &bf_to_atom,
                                 double *energy_density,
                                 const int *point_parent) {
  constexpr bool unrestricted =
      spinorbital_kind == SpinorbitalKind::Unrestricted;
  constexpr int nspin = unrestricted ? 2 : 1;
  const Eigen::Index npt = points_block.cols();
  const Eigen::Index nbf = gto_vals.phi.cols();

  Mat rho(nspin * npt, occ::density::num_components(derivative_order));
  SpinTerms terms[nspin];
  for (int s = 0; s < nspin; s++)
    spin_density<derivative_order>(D, s, gto_vals, terms[s].Dphi,
                                   terms[s].Dgrad,
                                   rho.middleRows(s * npt, npt));

  if (rho.col(0).maxCoeff() < density_threshold)
    return;

  const auto family = impl::family_for(derivative_order);
  DensityFunctional::Params params(npt, family, spinorbital_kind);
  impl::set_params<spinorbital_kind, derivative_order>(params, rho);
  DensityFunctional::Result res(npt, family, spinorbital_kind);
  for (const auto &func : funcs)
    res += func.evaluate(params);

  // Unweighted XC energy density exc * rho_total per point, for the grid
  // weight response (E_xc = sum_g w_g e_g).
  if (energy_density) {
    for (Eigen::Index i = 0; i < npt; i++)
      energy_density[i] =
          unrestricted ? res.exc(i) * (rho(i, 0) + rho(npt + i, 0))
                       : res.exc(i) * rho(i, 0);
  }

  res.weight_by(weights_block);

  Mat3N fsum = Mat3N::Zero(3, nbf);
  Mat3N psum = Mat3N::Zero(3, npt);
  Mat scratch;
  Vec vr(npt), f[3], vt;
  for (int s = 0; s < nspin; s++) {
    const int o = 1 - s; // the other spin
    if constexpr (derivative_order == 0) {
      vr = res.vrho.col(s);
    } else {
      vr = 0.5 * res.vrho.col(s).array();
      for (int j = 0; j < 3; j++) {
        if constexpr (unrestricted) {
          // vsigma columns: aa, ab, bb
          const auto vss = res.vsigma.col(s == 0 ? 0 : 2).array();
          const auto vab = res.vsigma.col(1).array();
          f[j] = 2.0 * vss * rho.col(1 + j).segment(s * npt, npt).array() +
                 vab * rho.col(1 + j).segment(o * npt, npt).array();
        } else {
          f[j] = 2.0 * res.vsigma.col(0).array() * rho.col(1 + j).array();
        }
      }
      if constexpr (derivative_order > 1)
        vt = 0.5 * res.vtau.col(s).array();
    }
    accumulate_spin<derivative_order>(
        D, s, gto_vals, terms[s], vr.data(),
        derivative_order > 0 ? f[0].data() : nullptr,
        derivative_order > 0 ? f[1].data() : nullptr,
        derivative_order > 0 ? f[2].data() : nullptr,
        derivative_order > 1 ? vt.data() : nullptr, fsum, psum, scratch);
  }

  for (Eigen::Index mu = 0; mu < nbf; mu++)
    gradient.col(bf_to_atom[mu]) -= 2.0 * fsum.col(mu);
  // Grid response: the points move with their parent atom, which by
  // translational invariance is minus the orbital term of those points.
  if (point_parent)
    for (Eigen::Index g = 0; g < npt; g++)
      gradient.col(point_parent[g]) += 2.0 * psum.col(g);
}

#define OCC_INSTANTIATE(order, kind)                                           \
  template void process_grid_block_gradient<order, kind>(                      \
      const BatchDensity &, const GTOValues &, Eigen::Ref<const Mat3N>,        \
      const Vec &, const std::vector<DensityFunctional> &, Mat3N &, double,    \
      const std::vector<int> &, double *, const int *);

OCC_INSTANTIATE(0, SpinorbitalKind::Restricted)
OCC_INSTANTIATE(1, SpinorbitalKind::Restricted)
OCC_INSTANTIATE(2, SpinorbitalKind::Restricted)
OCC_INSTANTIATE(0, SpinorbitalKind::Unrestricted)
OCC_INSTANTIATE(1, SpinorbitalKind::Unrestricted)
OCC_INSTANTIATE(2, SpinorbitalKind::Unrestricted)
#undef OCC_INSTANTIATE

} // namespace occ::dft::kernels
