#include <occ/dft/xc_potential_matrix.h>

namespace occ::dft {
using occ::qm::SpinorbitalKind::Restricted;
using occ::qm::SpinorbitalKind::Unrestricted;
namespace block = occ::qm::block;

// Each of these writes (assigns) H with V_xc = H + H^T for the batch, so that
// the caller can accumulate H over batches and symmetrize once at the end.

namespace {

// H = phi^T X for the GGA combination
//   X = phi vrho / 2 + phi_x g_x + phi_y g_y + phi_z g_z
// (g = 2 vsigma grad rho for a closed shell), built column by column.
void gga_half(const occ::gto::GTOValues &gv, const double *vr,
              const double *gx, const double *gy, const double *gz, MatRef H,
              Mat &X) {
  const Eigen::Index npt = gv.phi.rows(), nbf = gv.phi.cols();
  X.resize(npt, nbf);
  for (Eigen::Index mu = 0; mu < nbf; mu++) {
    const double *p = gv.phi.col(mu).data();
    const double *px = gv.phi_x.col(mu).data();
    const double *py = gv.phi_y.col(mu).data();
    const double *pz = gv.phi_z.col(mu).data();
    double *x = X.col(mu).data();
    for (Eigen::Index g = 0; g < npt; g++)
      x[g] = p[g] * vr[g] + px[g] * gx[g] + py[g] * gy[g] + pz[g] * gz[g];
  }
  H.noalias() = gv.phi.transpose() * X;
}

// H += (phi_x^T (phi_x t) + phi_y^T (phi_y t) + phi_z^T (phi_z t)) for the
// meta-GGA tau term (t = vtau / 4: half of the symmetric vtau / 2 term).
void tau_half(const occ::gto::GTOValues &gv, const Array &t, MatRef H, Mat &X) {
  X = gv.phi_x.array().colwise() * t;
  H.noalias() += gv.phi_x.transpose() * X;
  X = gv.phi_y.array().colwise() * t;
  H.noalias() += gv.phi_y.transpose() * X;
  X = gv.phi_z.array().colwise() * t;
  H.noalias() += gv.phi_z.transpose() * X;
}

// E_xc of the batch for an unrestricted density (exc per unit total density).
double unrestricted_energy(const DensityFunctional::Result &res,
                           MatConstRef rho) {
  return res.exc.dot(block::a(rho).col(0)) + res.exc.dot(block::b(rho).col(0));
}

} // namespace

template <>
void xc_potential_matrix<Restricted, 0>(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const occ::gto::GTOValues &gto_vals,
                                        MatRef Vxc, double &energy) {
  energy += rho.col(0).dot(res.exc);
  const Array vr = 0.5 * res.vrho.col(0).array();
  Mat X = gto_vals.phi.array().colwise() * vr;
  Vxc.noalias() = gto_vals.phi.transpose() * X;
}

template <>
void xc_potential_matrix<Restricted, 1>(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const occ::gto::GTOValues &gto_vals,
                                        MatRef Vxc, double &energy) {
  energy += rho.col(0).dot(res.exc);
  const Array vr = 0.5 * res.vrho.col(0).array();
  const auto vsigma = res.vsigma.col(0).array();
  const Array gx = 2 * (rho.col(1).array() * vsigma);
  const Array gy = 2 * (rho.col(2).array() * vsigma);
  const Array gz = 2 * (rho.col(3).array() * vsigma);
  Mat X;
  gga_half(gto_vals, vr.data(), gx.data(), gy.data(), gz.data(), Vxc, X);
}

template <>
void xc_potential_matrix<Restricted, 2>(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const occ::gto::GTOValues &gto_vals,
                                        MatRef Vxc, double &energy) {
  xc_potential_matrix<Restricted, 1>(res, rho, gto_vals, Vxc, energy);
  Mat X;
  tau_half(gto_vals, 0.25 * res.vtau.col(0).array(), Vxc, X);
}

template <>
void xc_potential_matrix<Unrestricted, 0>(const DensityFunctional::Result &res,
                                          MatConstRef rho,
                                          const occ::gto::GTOValues &gto_vals,
                                          MatRef Vxc, double &energy) {
  energy += unrestricted_energy(res, rho);
  Mat X = gto_vals.phi.array().colwise() * (0.5 * res.vrho.col(0).array());
  block::a(Vxc).noalias() = gto_vals.phi.transpose() * X;
  X = gto_vals.phi.array().colwise() * (0.5 * res.vrho.col(1).array());
  block::b(Vxc).noalias() = gto_vals.phi.transpose() * X;
}

template <>
void xc_potential_matrix<Unrestricted, 1>(const DensityFunctional::Result &res,
                                          MatConstRef rho,
                                          const occ::gto::GTOValues &gto_vals,
                                          MatRef Vxc, double &energy) {
  const Eigen::Index npt = res.npts;
  energy += unrestricted_energy(res, rho);

  // factor of 2 for vsigma up up, 1 for up down
  const auto ra = block::a(rho).block(0, 1, npt, 3).array();
  const auto rb = block::b(rho).block(0, 1, npt, 3).array();
  const Eigen::ArrayXXd ga = ra.colwise() * (2 * res.vsigma.col(0).array()) +
                             rb.colwise() * res.vsigma.col(1).array();
  const Eigen::ArrayXXd gb = rb.colwise() * (2 * res.vsigma.col(2).array()) +
                             ra.colwise() * res.vsigma.col(1).array();
  const Array vra = 0.5 * res.vrho.col(0).array();
  const Array vrb = 0.5 * res.vrho.col(1).array();
  Mat X;
  gga_half(gto_vals, vra.data(), ga.col(0).data(), ga.col(1).data(),
           ga.col(2).data(), block::a(Vxc), X);
  gga_half(gto_vals, vrb.data(), gb.col(0).data(), gb.col(1).data(),
           gb.col(2).data(), block::b(Vxc), X);
}

template <>
void xc_potential_matrix<Unrestricted, 2>(const DensityFunctional::Result &res,
                                          MatConstRef rho,
                                          const occ::gto::GTOValues &gto_vals,
                                          MatRef Vxc, double &energy) {
  xc_potential_matrix<Unrestricted, 1>(res, rho, gto_vals, Vxc, energy);
  Mat X;
  tau_half(gto_vals, 0.25 * res.vtau.col(0).array(), block::a(Vxc), X);
  tau_half(gto_vals, 0.25 * res.vtau.col(1).array(), block::b(Vxc), X);
}

} // namespace occ::dft
