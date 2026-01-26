#include <occ/dft/dft_gradient_kernels.h>
#include <occ/gto/density.h>
#include <occ/qm/opmatrix.h>

constexpr auto U = occ::qm::SpinorbitalKind::Unrestricted;
constexpr auto R = occ::qm::SpinorbitalKind::Restricted;
using occ::gto::GTOValues;
using occ::gto::AOBasis;
namespace block = occ::qm::block;

namespace occ::dft::kernels {

template <int derivative_order>
void accumulate_gradient_contribution_R(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const GTOValues &gto_vals, MatRef Vx,
                                        MatRef Vy, MatRef Vz);

template <int derivative_order>
void accumulate_gradient_contribution_U(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const GTOValues &gto_vals, MatRef Vxa,
                                        MatRef Vya, MatRef Vza, MatRef Vxb,
                                        MatRef Vyb, MatRef Vzb);

template <>
void accumulate_gradient_contribution_R<0>(
    const DensityFunctional::Result &res, MatConstRef rho,
    const GTOValues &gto_vals, MatRef Vx, MatRef Vy, MatRef Vz) {

  const auto &phi = gto_vals.phi;
  const auto &dx = gto_vals.phi_x;
  const auto &dy = gto_vals.phi_y;
  const auto &dz = gto_vals.phi_z;

  const Vec vrho = res.vrho.col(0);

  Vx.noalias() +=
      dx.transpose() * (phi.array().colwise() * vrho.array()).matrix();
  Vy.noalias() +=
      dy.transpose() * (phi.array().colwise() * vrho.array()).matrix();
  Vz.noalias() +=
      dz.transpose() * (phi.array().colwise() * vrho.array()).matrix();
}

template <>
void accumulate_gradient_contribution_R<1>(
    const DensityFunctional::Result &res, MatConstRef rho,
    const GTOValues &gto_vals, MatRef Vx, MatRef Vy, MatRef Vz) {
  const auto &p = gto_vals.phi;
  const auto &px = gto_vals.phi_x;
  const auto &py = gto_vals.phi_y;
  const auto &pz = gto_vals.phi_z;
  const auto &pxx = gto_vals.phi_xx;
  const auto &pxy = gto_vals.phi_xy;
  const auto &pxz = gto_vals.phi_xz;
  const auto &pyy = gto_vals.phi_yy;
  const auto &pyz = gto_vals.phi_yz;
  const auto &pzz = gto_vals.phi_zz;
  const Array dx = rho.col(1).array();
  const Array dy = rho.col(2).array();
  const Array dz = rho.col(3).array();

  const Array vrho = res.vrho.col(0).array() * 0.5;
  const Array vsigma = res.vsigma.col(0).array();

  Mat aow = (p.array().colwise() * vrho);
  aow.array() += px.array().colwise() * (2.0 * vsigma * dx);
  aow.array() += py.array().colwise() * (2.0 * vsigma * dy);
  aow.array() += pz.array().colwise() * (2.0 * vsigma * dz);

  Vx.noalias() += px.transpose() * aow;
  Vy.noalias() += py.transpose() * aow;
  Vz.noalias() += pz.transpose() * aow;

  aow = px.array().colwise() * vrho;
  aow.array() += pxx.array().colwise() * (2.0 * vsigma * dx);
  aow.array() += pxy.array().colwise() * (2.0 * vsigma * dy);
  aow.array() += pxz.array().colwise() * (2.0 * vsigma * dz);
  Vx.noalias() += (p.transpose() * aow).transpose();

  aow = py.array().colwise() * vrho;
  aow.array() += pxy.array().colwise() * (2.0 * vsigma * dx);
  aow.array() += pyy.array().colwise() * (2.0 * vsigma * dy);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma * dz);
  Vy.noalias() += (p.transpose() * aow).transpose();

  aow = pz.array().colwise() * vrho;
  aow.array() += pxz.array().colwise() * (2.0 * vsigma * dx);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma * dy);
  aow.array() += pzz.array().colwise() * (2.0 * vsigma * dz);
  Vz.noalias() += (p.transpose() * aow).transpose();
}

template <>
void accumulate_gradient_contribution_R<2>(
    const DensityFunctional::Result &res, MatConstRef rho,
    const GTOValues &gto_vals, MatRef Vx, MatRef Vy, MatRef Vz) {

  // MGGA case - derivative_order = 2
  const auto &p = gto_vals.phi;
  const auto &px = gto_vals.phi_x;
  const auto &py = gto_vals.phi_y;
  const auto &pz = gto_vals.phi_z;
  const auto &pxx = gto_vals.phi_xx;
  const auto &pxy = gto_vals.phi_xy;
  const auto &pxz = gto_vals.phi_xz;
  const auto &pyy = gto_vals.phi_yy;
  const auto &pyz = gto_vals.phi_yz;
  const auto &pzz = gto_vals.phi_zz;

  const Array dx = rho.col(1).array();
  const Array dy = rho.col(2).array();
  const Array dz = rho.col(3).array();

  const Array vrho = res.vrho.col(0).array() * 0.5;
  const Array vsigma = res.vsigma.col(0).array();
  const Array vtau = res.vtau.col(0).array() * 0.5;  // PySCF applies 0.5 to vtau

  // ========================================
  // GGA terms (same as derivative_order=1)
  // ========================================
  Mat aow = (p.array().colwise() * vrho);
  aow.array() += px.array().colwise() * (2.0 * vsigma * dx);
  aow.array() += py.array().colwise() * (2.0 * vsigma * dy);
  aow.array() += pz.array().colwise() * (2.0 * vsigma * dz);

  Vx.noalias() += px.transpose() * aow;
  Vy.noalias() += py.transpose() * aow;
  Vz.noalias() += pz.transpose() * aow;

  aow = px.array().colwise() * vrho;
  aow.array() += pxx.array().colwise() * (2.0 * vsigma * dx);
  aow.array() += pxy.array().colwise() * (2.0 * vsigma * dy);
  aow.array() += pxz.array().colwise() * (2.0 * vsigma * dz);
  Vx.noalias() += (p.transpose() * aow).transpose();

  aow = py.array().colwise() * vrho;
  aow.array() += pxy.array().colwise() * (2.0 * vsigma * dx);
  aow.array() += pyy.array().colwise() * (2.0 * vsigma * dy);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma * dz);
  Vy.noalias() += (p.transpose() * aow).transpose();

  aow = pz.array().colwise() * vrho;
  aow.array() += pxz.array().colwise() * (2.0 * vsigma * dx);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma * dy);
  aow.array() += pzz.array().colwise() * (2.0 * vsigma * dz);
  Vz.noalias() += (p.transpose() * aow).transpose();

  // ========================================
  // MGGA tau terms
  // ========================================
  // For tau gradient: ∫ vtau · (∂²φ_μ/∂X∂x · ∂φ_ν/∂x + ∂φ_μ/∂x · ∂²φ_ν/∂X∂x)
  // Like PySCF, compute only bra term - factor of 2 in distribution handles ket term

  // x-derivative contribution
  aow = px.array().colwise() * vtau;
  Vx.noalias() += pxx.transpose() * aow;
  Vy.noalias() += pxy.transpose() * aow;
  Vz.noalias() += pxz.transpose() * aow;

  // y-derivative contribution
  aow = py.array().colwise() * vtau;
  Vx.noalias() += pxy.transpose() * aow;
  Vy.noalias() += pyy.transpose() * aow;
  Vz.noalias() += pyz.transpose() * aow;

  // z-derivative contribution
  aow = pz.array().colwise() * vtau;
  Vx.noalias() += pxz.transpose() * aow;
  Vy.noalias() += pyz.transpose() * aow;
  Vz.noalias() += pzz.transpose() * aow;
}

template <>
void accumulate_gradient_contribution_U<0>(
    const DensityFunctional::Result &res, MatConstRef rho,
    const GTOValues &gto_vals, MatRef Vx_a, MatRef Vy_a, MatRef Vz_a,
    MatRef Vx_b, MatRef Vy_b, MatRef Vz_b) {

  // LDA case - derivative_order = 0
  const auto &phi = gto_vals.phi;
  const auto &dx = gto_vals.phi_x;
  const auto &dy = gto_vals.phi_y;
  const auto &dz = gto_vals.phi_z;

  // For unrestricted, res.vrho has 2 columns: vrho_alpha (col 0) and vrho_beta (col 1)
  const Vec vrho_a = res.vrho.col(0);
  const Vec vrho_b = res.vrho.col(1);

  // Alpha contributions
  Vx_a.noalias() +=
      dx.transpose() * (phi.array().colwise() * vrho_a.array()).matrix();
  Vy_a.noalias() +=
      dy.transpose() * (phi.array().colwise() * vrho_a.array()).matrix();
  Vz_a.noalias() +=
      dz.transpose() * (phi.array().colwise() * vrho_a.array()).matrix();

  // Beta contributions
  Vx_b.noalias() +=
      dx.transpose() * (phi.array().colwise() * vrho_b.array()).matrix();
  Vy_b.noalias() +=
      dy.transpose() * (phi.array().colwise() * vrho_b.array()).matrix();
  Vz_b.noalias() +=
      dz.transpose() * (phi.array().colwise() * vrho_b.array()).matrix();
}

template <>
void accumulate_gradient_contribution_U<1>(
    const DensityFunctional::Result &res, MatConstRef rho,
    const GTOValues &gto_vals, MatRef Vx_a, MatRef Vy_a, MatRef Vz_a,
    MatRef Vx_b, MatRef Vy_b, MatRef Vz_b) {

  // GGA case - derivative_order = 1
  const auto &p = gto_vals.phi;
  const auto &px = gto_vals.phi_x;
  const auto &py = gto_vals.phi_y;
  const auto &pz = gto_vals.phi_z;
  const auto &pxx = gto_vals.phi_xx;
  const auto &pxy = gto_vals.phi_xy;
  const auto &pxz = gto_vals.phi_xz;
  const auto &pyy = gto_vals.phi_yy;
  const auto &pyz = gto_vals.phi_yz;
  const auto &pzz = gto_vals.phi_zz;

  // For unrestricted GGA, rho has dimensions (2*npt, 4)
  // First npt rows are alpha: rho_a, grad_rho_a_x, grad_rho_a_y, grad_rho_a_z
  // Next npt rows are beta: rho_b, grad_rho_b_x, grad_rho_b_y, grad_rho_b_z
  const size_t npt = rho.rows() / 2;

  const Array dx_a = rho.block(0, 1, npt, 1).array();
  const Array dy_a = rho.block(0, 2, npt, 1).array();
  const Array dz_a = rho.block(0, 3, npt, 1).array();

  const Array dx_b = rho.block(npt, 1, npt, 1).array();
  const Array dy_b = rho.block(npt, 2, npt, 1).array();
  const Array dz_b = rho.block(npt, 3, npt, 1).array();

  // For unrestricted GGA, res.vrho has 2 columns: vrho_alpha, vrho_beta
  // res.vsigma has 3 columns: vsigma_aa, vsigma_ab, vsigma_bb
  // Apply 0.5 factor to match PySCF convention and compensate for matrix symmetrization
  const Array vrho_a = res.vrho.col(0).array() * 0.5;
  const Array vrho_b = res.vrho.col(1).array() * 0.5;

  const Array vsigma_aa = res.vsigma.col(0).array();
  const Array vsigma_ab = res.vsigma.col(1).array();
  const Array vsigma_bb = res.vsigma.col(2).array();

  // Alpha contributions
  Mat aow = (p.array().colwise() * vrho_a);
  aow.array() += px.array().colwise() * (2.0 * vsigma_aa * dx_a + vsigma_ab * dx_b);
  aow.array() += py.array().colwise() * (2.0 * vsigma_aa * dy_a + vsigma_ab * dy_b);
  aow.array() += pz.array().colwise() * (2.0 * vsigma_aa * dz_a + vsigma_ab * dz_b);

  Vx_a.noalias() += px.transpose() * aow;
  Vy_a.noalias() += py.transpose() * aow;
  Vz_a.noalias() += pz.transpose() * aow;

  aow = px.array().colwise() * vrho_a;
  aow.array() += pxx.array().colwise() * (2.0 * vsigma_aa * dx_a + vsigma_ab * dx_b);
  aow.array() += pxy.array().colwise() * (2.0 * vsigma_aa * dy_a + vsigma_ab * dy_b);
  aow.array() += pxz.array().colwise() * (2.0 * vsigma_aa * dz_a + vsigma_ab * dz_b);
  Vx_a.noalias() += (p.transpose() * aow).transpose();

  aow = py.array().colwise() * vrho_a;
  aow.array() += pxy.array().colwise() * (2.0 * vsigma_aa * dx_a + vsigma_ab * dx_b);
  aow.array() += pyy.array().colwise() * (2.0 * vsigma_aa * dy_a + vsigma_ab * dy_b);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma_aa * dz_a + vsigma_ab * dz_b);
  Vy_a.noalias() += (p.transpose() * aow).transpose();

  aow = pz.array().colwise() * vrho_a;
  aow.array() += pxz.array().colwise() * (2.0 * vsigma_aa * dx_a + vsigma_ab * dx_b);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma_aa * dy_a + vsigma_ab * dy_b);
  aow.array() += pzz.array().colwise() * (2.0 * vsigma_aa * dz_a + vsigma_ab * dz_b);
  Vz_a.noalias() += (p.transpose() * aow).transpose();

  // Beta contributions
  aow = (p.array().colwise() * vrho_b);
  aow.array() += px.array().colwise() * (2.0 * vsigma_bb * dx_b + vsigma_ab * dx_a);
  aow.array() += py.array().colwise() * (2.0 * vsigma_bb * dy_b + vsigma_ab * dy_a);
  aow.array() += pz.array().colwise() * (2.0 * vsigma_bb * dz_b + vsigma_ab * dz_a);

  Vx_b.noalias() += px.transpose() * aow;
  Vy_b.noalias() += py.transpose() * aow;
  Vz_b.noalias() += pz.transpose() * aow;

  aow = px.array().colwise() * vrho_b;
  aow.array() += pxx.array().colwise() * (2.0 * vsigma_bb * dx_b + vsigma_ab * dx_a);
  aow.array() += pxy.array().colwise() * (2.0 * vsigma_bb * dy_b + vsigma_ab * dy_a);
  aow.array() += pxz.array().colwise() * (2.0 * vsigma_bb * dz_b + vsigma_ab * dz_a);
  Vx_b.noalias() += (p.transpose() * aow).transpose();

  aow = py.array().colwise() * vrho_b;
  aow.array() += pxy.array().colwise() * (2.0 * vsigma_bb * dx_b + vsigma_ab * dx_a);
  aow.array() += pyy.array().colwise() * (2.0 * vsigma_bb * dy_b + vsigma_ab * dy_a);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma_bb * dz_b + vsigma_ab * dz_a);
  Vy_b.noalias() += (p.transpose() * aow).transpose();

  aow = pz.array().colwise() * vrho_b;
  aow.array() += pxz.array().colwise() * (2.0 * vsigma_bb * dx_b + vsigma_ab * dx_a);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma_bb * dy_b + vsigma_ab * dy_a);
  aow.array() += pzz.array().colwise() * (2.0 * vsigma_bb * dz_b + vsigma_ab * dz_a);
  Vz_b.noalias() += (p.transpose() * aow).transpose();
}

template <>
void accumulate_gradient_contribution_U<2>(
    const DensityFunctional::Result &res, MatConstRef rho,
    const GTOValues &gto_vals, MatRef Vx_a, MatRef Vy_a, MatRef Vz_a,
    MatRef Vx_b, MatRef Vy_b, MatRef Vz_b) {

  // MGGA case - derivative_order = 2
  const auto &p = gto_vals.phi;
  const auto &px = gto_vals.phi_x;
  const auto &py = gto_vals.phi_y;
  const auto &pz = gto_vals.phi_z;
  const auto &pxx = gto_vals.phi_xx;
  const auto &pxy = gto_vals.phi_xy;
  const auto &pxz = gto_vals.phi_xz;
  const auto &pyy = gto_vals.phi_yy;
  const auto &pyz = gto_vals.phi_yz;
  const auto &pzz = gto_vals.phi_zz;

  // For unrestricted MGGA, rho has dimensions (2*npt, 6)
  // First npt rows: alpha (ρ_α, ∇ρ_α_x, ∇ρ_α_y, ∇ρ_α_z, ∇²ρ_α, τ_α)
  // Next npt rows: beta (ρ_β, ∇ρ_β_x, ∇ρ_β_y, ∇ρ_β_z, ∇²ρ_β, τ_β)
  const size_t npt = rho.rows() / 2;

  const Array dx_a = rho.block(0, 1, npt, 1).array();
  const Array dy_a = rho.block(0, 2, npt, 1).array();
  const Array dz_a = rho.block(0, 3, npt, 1).array();

  const Array dx_b = rho.block(npt, 1, npt, 1).array();
  const Array dy_b = rho.block(npt, 2, npt, 1).array();
  const Array dz_b = rho.block(npt, 3, npt, 1).array();

  // XC potential derivatives
  const Array vrho_a = res.vrho.col(0).array() * 0.5;
  const Array vrho_b = res.vrho.col(1).array() * 0.5;

  const Array vsigma_aa = res.vsigma.col(0).array();
  const Array vsigma_ab = res.vsigma.col(1).array();
  const Array vsigma_bb = res.vsigma.col(2).array();

  const Array vtau_a = res.vtau.col(0).array() * 0.5;  // PySCF applies 0.5 to vtau
  const Array vtau_b = res.vtau.col(1).array() * 0.5;  // PySCF applies 0.5 to vtau

  // ========================================
  // ALPHA SPIN: GGA terms
  // ========================================
  Mat aow = (p.array().colwise() * vrho_a);
  aow.array() += px.array().colwise() * (2.0 * vsigma_aa * dx_a + vsigma_ab * dx_b);
  aow.array() += py.array().colwise() * (2.0 * vsigma_aa * dy_a + vsigma_ab * dy_b);
  aow.array() += pz.array().colwise() * (2.0 * vsigma_aa * dz_a + vsigma_ab * dz_b);

  Vx_a.noalias() += px.transpose() * aow;
  Vy_a.noalias() += py.transpose() * aow;
  Vz_a.noalias() += pz.transpose() * aow;

  aow = px.array().colwise() * vrho_a;
  aow.array() += pxx.array().colwise() * (2.0 * vsigma_aa * dx_a + vsigma_ab * dx_b);
  aow.array() += pxy.array().colwise() * (2.0 * vsigma_aa * dy_a + vsigma_ab * dy_b);
  aow.array() += pxz.array().colwise() * (2.0 * vsigma_aa * dz_a + vsigma_ab * dz_b);
  Vx_a.noalias() += (p.transpose() * aow).transpose();

  aow = py.array().colwise() * vrho_a;
  aow.array() += pxy.array().colwise() * (2.0 * vsigma_aa * dx_a + vsigma_ab * dx_b);
  aow.array() += pyy.array().colwise() * (2.0 * vsigma_aa * dy_a + vsigma_ab * dy_b);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma_aa * dz_a + vsigma_ab * dz_b);
  Vy_a.noalias() += (p.transpose() * aow).transpose();

  aow = pz.array().colwise() * vrho_a;
  aow.array() += pxz.array().colwise() * (2.0 * vsigma_aa * dx_a + vsigma_ab * dx_b);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma_aa * dy_a + vsigma_ab * dy_b);
  aow.array() += pzz.array().colwise() * (2.0 * vsigma_aa * dz_a + vsigma_ab * dz_b);
  Vz_a.noalias() += (p.transpose() * aow).transpose();

  // ========================================
  // ALPHA SPIN: MGGA tau terms
  // ========================================
  // For tau gradient: ∫ vtau · (∂²φ_μ/∂X∂x · ∂φ_ν/∂x + ∂φ_μ/∂x · ∂²φ_ν/∂X∂x)
  // Need to symmetrize: compute bra term and add its transpose for ket term

  const size_t nbf = p.cols();
  Mat Vx_tau_a = Mat::Zero(nbf, nbf);
  Mat Vy_tau_a = Mat::Zero(nbf, nbf);
  Mat Vz_tau_a = Mat::Zero(nbf, nbf);

  aow = px.array().colwise() * vtau_a;
  Vx_tau_a.noalias() += pxx.transpose() * aow;
  Vy_tau_a.noalias() += pxy.transpose() * aow;
  Vz_tau_a.noalias() += pxz.transpose() * aow;

  aow = py.array().colwise() * vtau_a;
  Vx_tau_a.noalias() += pxy.transpose() * aow;
  Vy_tau_a.noalias() += pyy.transpose() * aow;
  Vz_tau_a.noalias() += pyz.transpose() * aow;

  aow = pz.array().colwise() * vtau_a;
  Vx_tau_a.noalias() += pxz.transpose() * aow;
  Vy_tau_a.noalias() += pyz.transpose() * aow;
  Vz_tau_a.noalias() += pzz.transpose() * aow;

  // PySCF does NOT symmetrize tau contributions in the gradient
  // The symmetrization happens implicitly during the distribution loop
  Vx_a.noalias() += Vx_tau_a;
  Vy_a.noalias() += Vy_tau_a;
  Vz_a.noalias() += Vz_tau_a;

  // ========================================
  // BETA SPIN: GGA terms
  // ========================================
  aow = (p.array().colwise() * vrho_b);
  aow.array() += px.array().colwise() * (2.0 * vsigma_bb * dx_b + vsigma_ab * dx_a);
  aow.array() += py.array().colwise() * (2.0 * vsigma_bb * dy_b + vsigma_ab * dy_a);
  aow.array() += pz.array().colwise() * (2.0 * vsigma_bb * dz_b + vsigma_ab * dz_a);

  Vx_b.noalias() += px.transpose() * aow;
  Vy_b.noalias() += py.transpose() * aow;
  Vz_b.noalias() += pz.transpose() * aow;

  aow = px.array().colwise() * vrho_b;
  aow.array() += pxx.array().colwise() * (2.0 * vsigma_bb * dx_b + vsigma_ab * dx_a);
  aow.array() += pxy.array().colwise() * (2.0 * vsigma_bb * dy_b + vsigma_ab * dy_a);
  aow.array() += pxz.array().colwise() * (2.0 * vsigma_bb * dz_b + vsigma_ab * dz_a);
  Vx_b.noalias() += (p.transpose() * aow).transpose();

  aow = py.array().colwise() * vrho_b;
  aow.array() += pxy.array().colwise() * (2.0 * vsigma_bb * dx_b + vsigma_ab * dx_a);
  aow.array() += pyy.array().colwise() * (2.0 * vsigma_bb * dy_b + vsigma_ab * dy_a);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma_bb * dz_b + vsigma_ab * dz_a);
  Vy_b.noalias() += (p.transpose() * aow).transpose();

  aow = pz.array().colwise() * vrho_b;
  aow.array() += pxz.array().colwise() * (2.0 * vsigma_bb * dx_b + vsigma_ab * dx_a);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma_bb * dy_b + vsigma_ab * dy_a);
  aow.array() += pzz.array().colwise() * (2.0 * vsigma_bb * dz_b + vsigma_ab * dz_a);
  Vz_b.noalias() += (p.transpose() * aow).transpose();

  // ========================================
  // BETA SPIN: MGGA tau terms
  // ========================================
  // For tau gradient: ∫ vtau · (∂²φ_μ/∂X∂x · ∂φ_ν/∂x + ∂φ_μ/∂x · ∂²φ_ν/∂X∂x)
  // Need to symmetrize: compute bra term and add its transpose for ket term

  Mat Vx_tau_b = Mat::Zero(nbf, nbf);
  Mat Vy_tau_b = Mat::Zero(nbf, nbf);
  Mat Vz_tau_b = Mat::Zero(nbf, nbf);

  aow = px.array().colwise() * vtau_b;
  Vx_tau_b.noalias() += pxx.transpose() * aow;
  Vy_tau_b.noalias() += pxy.transpose() * aow;
  Vz_tau_b.noalias() += pxz.transpose() * aow;

  aow = py.array().colwise() * vtau_b;
  Vx_tau_b.noalias() += pxy.transpose() * aow;
  Vy_tau_b.noalias() += pyy.transpose() * aow;
  Vz_tau_b.noalias() += pyz.transpose() * aow;

  aow = pz.array().colwise() * vtau_b;
  Vx_tau_b.noalias() += pxz.transpose() * aow;
  Vy_tau_b.noalias() += pyz.transpose() * aow;
  Vz_tau_b.noalias() += pzz.transpose() * aow;

  // PySCF does NOT symmetrize tau contributions in the gradient
  // The symmetrization happens implicitly during the distribution loop
  Vx_b.noalias() += Vx_tau_b;
  Vy_b.noalias() += Vy_tau_b;
  Vz_b.noalias() += Vz_tau_b;
}

template <int derivative_order, SpinorbitalKind spinorbital_kind>
void process_grid_block_gradient(const Mat &D, const GTOValues &gto_vals,
                                 Eigen::Ref<const Mat3N> points_block,
                                 const Vec &weights_block,
                                 const std::vector<DensityFunctional> &funcs,
                                 Mat3N &gradient, double density_threshold,
                                 const AOBasis &basis) {
  const size_t npt = points_block.cols();
  const size_t nbf = gto_vals.phi.cols();
  const size_t natoms = gradient.cols();
  size_t num_rows_factor =
      (spinorbital_kind == SpinorbitalKind::Unrestricted) ? 2 : 1;

  // Evaluate density at grid points
  Mat rho(num_rows_factor * npt,
          occ::density::num_components(derivative_order));
  occ::density::evaluate_density<derivative_order, spinorbital_kind>(
      D, gto_vals, rho);

  // Skip computation if density is too small
  double max_density_block = rho.col(0).maxCoeff();
  if (max_density_block < density_threshold)
    return;

  // Set up functional parameters and evaluate
  DensityFunctional::Family family = DensityFunctional::Family::LDA;
  if constexpr (derivative_order == 1) {
    family = DensityFunctional::Family::GGA;
  }
  if constexpr (derivative_order == 2) {
    family = DensityFunctional::Family::MGGA;
  }

  DensityFunctional::Params params(npt, family, spinorbital_kind);
  impl::set_params<spinorbital_kind, derivative_order>(params, rho);

  // Evaluate functional
  DensityFunctional::Result res(npt, family, spinorbital_kind);
  for (const auto &func : funcs) {
    res += func.evaluate(params);
  }

  // Apply weights to XC potential
  res.weight_by(weights_block);

  // Map basis functions to atoms
  const auto &bf_to_atom = basis.bf_to_atom();

  // For restricted case
  if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
    // Initialize gradient matrices
    Mat Vx = Mat::Zero(nbf, nbf);
    Mat Vy = Mat::Zero(nbf, nbf);
    Mat Vz = Mat::Zero(nbf, nbf);

    // Accumulate all gradient contributions based on derivative order
    if constexpr (derivative_order == 0) {
      accumulate_gradient_contribution_R<0>(res, rho, gto_vals, Vx, Vy, Vz);
    } else if constexpr (derivative_order == 1) {
      accumulate_gradient_contribution_R<1>(res, rho, gto_vals, Vx, Vy, Vz);
    } else if constexpr (derivative_order == 2) {
      accumulate_gradient_contribution_R<2>(res, rho, gto_vals, Vx, Vy, Vz);
    }

    // Distribute to atoms
    for (size_t mu = 0; mu < nbf; mu++) {
      int atom_mu = bf_to_atom[mu];
      for (size_t nu = 0; nu < nbf; nu++) {
        double Dval = 2.0 * D(mu, nu); // Factor of 2 for restricted case
        if (std::abs(Dval) < 1e-12)
          continue;

        // Add contributions to gradients
        gradient(0, atom_mu) -= Vx(mu, nu) * Dval;
        gradient(1, atom_mu) -= Vy(mu, nu) * Dval;
        gradient(2, atom_mu) -= Vz(mu, nu) * Dval;
      }
    }
  }
  // For unrestricted case
  else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
    // Get alpha and beta blocks of the density matrix
    const auto Da = block::a(D);
    const auto Db = block::b(D);

    // Initialize gradient matrices for alpha and beta
    Mat Vx_a = Mat::Zero(nbf, nbf);
    Mat Vy_a = Mat::Zero(nbf, nbf);
    Mat Vz_a = Mat::Zero(nbf, nbf);

    Mat Vx_b = Mat::Zero(nbf, nbf);
    Mat Vy_b = Mat::Zero(nbf, nbf);
    Mat Vz_b = Mat::Zero(nbf, nbf);

    // Accumulate all gradient contributions based on derivative order
    if constexpr (derivative_order == 0) {
      accumulate_gradient_contribution_U<0>(res, rho, gto_vals, Vx_a, Vy_a, Vz_a, Vx_b, Vy_b, Vz_b);
    } else if constexpr (derivative_order == 1) {
      accumulate_gradient_contribution_U<1>(res, rho, gto_vals, Vx_a, Vy_a, Vz_a, Vx_b, Vy_b, Vz_b);
    } else if constexpr (derivative_order == 2) {
      accumulate_gradient_contribution_U<2>(res, rho, gto_vals, Vx_a, Vy_a, Vz_a, Vx_b, Vy_b, Vz_b);
    }

    // Distribute to atoms - PySCF style to avoid overcounting
    // Loop over atoms first, then only basis functions on that atom
    for (size_t atom = 0; atom < natoms; atom++) {
      for (size_t mu = 0; mu < nbf; mu++) {
        if (bf_to_atom[mu] != static_cast<int>(atom)) continue;

        for (size_t nu = 0; nu < nbf; nu++) {
          const double Dvala = Da(mu, nu);
          const double Dvalb = Db(mu, nu);
          if (std::abs(Dvala + Dvalb) < 1e-12)
            continue;

          // Factor of 2 accounts for both ∂/∂X_A ⟨μ|V|ν⟩ and ∂/∂X_A ⟨ν|V|μ⟩
          gradient(0, atom) -= 2.0 * (Vx_a(mu, nu) * Dvala + Vx_b(mu, nu) * Dvalb);
          gradient(1, atom) -= 2.0 * (Vy_a(mu, nu) * Dvala + Vy_b(mu, nu) * Dvalb);
          gradient(2, atom) -= 2.0 * (Vz_a(mu, nu) * Dvala + Vz_b(mu, nu) * Dvalb);
        }
      }
    }
  }
}

template void process_grid_block_gradient<0, R>(
    const Mat &D, const GTOValues &gto_vals,
    Eigen::Ref<const Mat3N> points_block, const Vec &weights_block,
    const std::vector<DensityFunctional> &funcs, Mat3N &gradient,
    double density_threshold, const AOBasis &basis);

template void process_grid_block_gradient<1, R>(
    const Mat &D, const GTOValues &gto_vals,
    Eigen::Ref<const Mat3N> points_block, const Vec &weights_block,
    const std::vector<DensityFunctional> &funcs, Mat3N &gradient,
    double density_threshold, const AOBasis &basis);

template void process_grid_block_gradient<2, R>(
    const Mat &D, const GTOValues &gto_vals,
    Eigen::Ref<const Mat3N> points_block, const Vec &weights_block,
    const std::vector<DensityFunctional> &funcs, Mat3N &gradient,
    double density_threshold, const AOBasis &basis);
template void process_grid_block_gradient<0, U>(
    const Mat &D, const GTOValues &gto_vals,
    Eigen::Ref<const Mat3N> points_block, const Vec &weights_block,
    const std::vector<DensityFunctional> &funcs, Mat3N &gradient,
    double density_threshold, const AOBasis &basis);
template void process_grid_block_gradient<1, U>(
    const Mat &D, const GTOValues &gto_vals,
    Eigen::Ref<const Mat3N> points_block, const Vec &weights_block,
    const std::vector<DensityFunctional> &funcs, Mat3N &gradient,
    double density_threshold, const AOBasis &basis);
template void process_grid_block_gradient<2, U>(
    const Mat &D, const GTOValues &gto_vals,
    Eigen::Ref<const Mat3N> points_block, const Vec &weights_block,
    const std::vector<DensityFunctional> &funcs, Mat3N &gradient,
    double density_threshold, const AOBasis &basis);

} // namespace occ::dft::kernels
