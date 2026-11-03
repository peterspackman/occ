#pragma once
// Shared pieces of the XC second-derivative code (CPKS response, derivative
// Fock matrices, Hessian): functional derivatives on a batch of points, and
// the spin-polarized kernel.
#include <occ/core/linear_algebra.h>
#include <occ/dft/dft_kernels.h>
#include <occ/dft/functional.h>

namespace occ::dft::detail {

/// First and second derivatives of the sum of `funcs` at the densities rho
/// (columns as occ::density::evaluate_density; unrestricted, alpha rows over
/// beta rows). With `exc`, also the summed energy per particle. funcs must
/// not be empty.
template <SpinorbitalKind kind, int deriv>
DensityFunctional::SecondDerivatives
second_derivatives(const std::vector<DensityFunctional> &funcs, const Mat &rho,
                   Vec *exc = nullptr) {
  const Eigen::Index n =
      kind == SpinorbitalKind::Unrestricted ? rho.rows() / 2 : rho.rows();
  DensityFunctional::Params params(n, impl::family_for(deriv), kind);
  impl::set_params<kind, deriv>(params, rho);
  auto f = funcs[0].evaluate_second_derivatives(params);
  for (size_t i = 1; i < funcs.size(); i++)
    f += funcs[i].evaluate_second_derivatives(params);
  if (exc) {
    *exc = Vec::Zero(n);
    for (const auto &func : funcs)
      *exc += func.evaluate(params).exc;
  }
  return f;
}

/// Quadrature weights with the points whose (total) density is below the
/// threshold zeroed: those points contribute nothing, as in the V_xc build.
inline Array screened_weights(const Vec &w, const Array &density,
                              double threshold) {
  return (density < threshold).select(0.0, w.array());
}

// The spin-polarized XC kernel on a batch of points, for unrestricted
// response and second derivatives (LDA and GGA).
//
// With spin densities rho^s and sigma = (aa, ab, bb), occ's unrestricted
// potential is V_xc^s = H + H^T, H = phi^T (v_rho^s / 2 phi + V^s . grad phi),
//
//   V^a = 2 v_sigma^aa grad rho^a + v_sigma^ab grad rho^b   (b likewise),
//
// and a change (rho1^s, grad rho1^s) of both spin densities changes the
// coefficients (v_rho^s, V^s) by
//
//   dv_rho^s   = sum_t f_{rho^s rho^t} rho1^t
//                + sum_c f_{rho^s sigma^c} sigma1^c
//   dv_sigma^c = sum_t f_{rho^t sigma^c} rho1^t
//                + sum_d f_{sigma^c sigma^d} sigma1^d
//   dV^a       = 2 dv_sigma^aa grad rho^a + 2 v_sigma^aa grad rho1^a
//                + dv_sigma^ab grad rho^b + v_sigma^ab grad rho1^b
//   sigma1     = (2 grad rho^a . grad rho1^a,
//                 grad rho^a . grad rho1^b + grad rho^b . grad rho1^a,
//                 2 grad rho^b . grad rho1^b).
//
// The two channels couple only here; everything else in an unrestricted
// second-derivative term is the restricted expression per spin.
template <int deriv> struct SpinKernel {
  // libxc's polarized layout: v2rho2 (aa, ab, bb); vsigma (aa, ab, bb);
  // v2rhosigma (a.aa, a.ab, a.bb, b.aa, b.ab, b.bb); v2sigma2 the upper
  // triangle of sigma x sigma, row by row.
  const DensityFunctional::SecondDerivatives &f;
  Array grad[2][3]; ///< grad rho^s

  /// rho: alpha rows over beta rows, columns (rho, grad rho) as
  /// occ::density::evaluate_density.
  SpinKernel(const DensityFunctional::SecondDerivatives &f, const Mat &rho)
      : f(f) {
    if constexpr (deriv == 1) {
      const Eigen::Index n = rho.rows() / 2;
      for (int s = 0; s < 2; s++)
        for (int i = 0; i < 3; i++)
          grad[s][i] = rho.col(1 + i).segment(s * n, n).array();
    }
  }

  inline auto frr(int s, int t) const { return f.v2rho2.col(s + t).array(); }
  inline auto vs(int c) const { return f.vsigma.col(c).array(); }
  inline auto frs(int s, int c) const {
    return f.v2rhosigma.col(3 * s + c).array();
  }
  inline auto fss(int c, int d) const {
    static const int idx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}};
    return f.v2sigma2.col(idx[c][d]).array();
  }
  inline auto vr(int s) const { return f.vrho.col(s).array(); }

  /// V^s along direction i.
  Array V(int s, int i) const {
    return 2.0 * vs(2 * s) * grad[s][i] + vs(1) * grad[1 - s][i];
  }

  /// The change of (v_rho^s, V^s) for each column of (r1^s, g1^s): every
  /// argument is n x m, one perturbation per column.
  void apply(const Mat r1[2], const Mat g1[2][3], Mat dvr[2],
             Mat dV[2][3]) const {
    auto times = [](const Mat &m, const Array &a) -> Mat {
      return (m.array().colwise() * a).matrix();
    };
    for (int s = 0; s < 2; s++)
      dvr[s] = times(r1[0], frr(s, 0)) + times(r1[1], frr(s, 1));
    if constexpr (deriv == 1) {
      Mat s1[3];
      for (int c = 0; c < 3; c++)
        s1[c] = Mat::Zero(r1[0].rows(), r1[0].cols());
      for (int i = 0; i < 3; i++) {
        s1[0] += 2.0 * times(g1[0][i], grad[0][i]);
        s1[1] += times(g1[1][i], grad[0][i]) + times(g1[0][i], grad[1][i]);
        s1[2] += 2.0 * times(g1[1][i], grad[1][i]);
      }
      Mat dvs[3];
      for (int c = 0; c < 3; c++) {
        dvs[c] = times(r1[0], frs(0, c)) + times(r1[1], frs(1, c));
        for (int d = 0; d < 3; d++)
          dvs[c] += times(s1[d], fss(c, d));
        for (int s = 0; s < 2; s++)
          dvr[s] += times(s1[c], frs(s, c));
      }
      for (int s = 0; s < 2; s++) {
        const int ss = 2 * s, o = 1 - s;
        for (int i = 0; i < 3; i++)
          dV[s][i] = 2.0 * (times(dvs[ss], grad[s][i]) +
                            times(g1[s][i], vs(ss))) +
                     times(dvs[1], grad[o][i]) + times(g1[o][i], vs(1));
      }
    }
  }
};

} // namespace occ::dft::detail
