#include <occ/dft/dft_kernels.h>
#include <occ/dft/xc_potential_matrix.h>
#include <occ/gto/density.h>
#include <occ/qm/mo.h>
#include <occ/qm/opmatrix.h>

namespace occ::dft::kernels {

using occ::gto::GTOValues;

void BatchDensity::times(const Mat &X, int s, Mat &out) const {
  if (F) {
    const auto Fs = F->middleRows(s * nbf, nbf);
    const Mat XF = X * Fs;
    out.noalias() = XF * Fs.transpose();
  } else {
    out.noalias() = X * D->middleRows(s * nbf, nbf);
  }
}

Mat density_factor(const occ::qm::MolecularOrbitals &mo, const Mat &D2,
                   int nspin) {
  const Eigen::Index nbf = D2.cols();
  if (mo.Cocc.rows() != nspin * nbf || mo.occupation.size() != nspin * nbf)
    return Mat();
  // occupied columns of each spin's block of Cocc (already scaled by the
  // square roots of the occupations)
  std::vector<std::vector<Eigen::Index>> cols(nspin);
  Eigen::Index k = 0;
  for (int s = 0; s < nspin; s++) {
    for (Eigen::Index j = 0; j < mo.Cocc.cols() && j < nbf; j++)
      if (mo.occupation(s * nbf + j) > 0.0)
        cols[s].push_back(j);
    k = std::max<Eigen::Index>(k, cols[s].size());
  }
  if (k == 0)
    return Mat();
  Mat F = Mat::Zero(nspin * nbf, k);
  const double scale = std::sqrt(2.0); // D2 = 2 D, D = Cocc Cocc^T
  for (int s = 0; s < nspin; s++)
    for (size_t c = 0; c < cols[s].size(); c++)
      F.col(c).segment(s * nbf, nbf) =
          scale * mo.Cocc.col(cols[s][c]).segment(s * nbf, nbf);
  for (int s = 0; s < nspin; s++) {
    const auto Fs = F.middleRows(s * nbf, nbf);
    const auto Ds = D2.middleRows(s * nbf, nbf);
    const double dmax = std::max(1.0, Ds.cwiseAbs().maxCoeff());
    if ((Fs * Fs.transpose() - Ds).cwiseAbs().maxCoeff() > 1e-12 * dmax)
      return Mat();
  }
  return F;
}

template <int derivative_order>
void spin_density(const BatchDensity &dens, int s, const GTOValues &gv,
                  Mat &Dphi, Mat *Dgrad, MatRef rho) {
  dens.times(gv.phi, s, Dphi);
  rho.col(0) = (gv.phi.array() * Dphi.array()).rowwise().sum();
  if constexpr (derivative_order > 0) {
    rho.col(1) = 2 * (gv.phi_x.array() * Dphi.array()).rowwise().sum();
    rho.col(2) = 2 * (gv.phi_y.array() * Dphi.array()).rowwise().sum();
    rho.col(3) = 2 * (gv.phi_z.array() * Dphi.array()).rowwise().sum();
  }
  if constexpr (derivative_order > 1) {
    rho.col(4) = 2 * ((gv.phi_xx.array() + gv.phi_yy.array() +
                       gv.phi_zz.array()) *
                      Dphi.array())
                         .rowwise()
                         .sum();
    Mat tmp[3];
    Mat *g = Dgrad ? Dgrad : tmp;
    dens.times(gv.phi_x, s, g[0]);
    rho.col(5) = (gv.phi_x.array() * g[0].array()).rowwise().sum();
    dens.times(gv.phi_y, s, g[1]);
    rho.col(5).array() += (gv.phi_y.array() * g[1].array()).rowwise().sum();
    dens.times(gv.phi_z, s, g[2]);
    rho.col(5).array() += (gv.phi_z.array() * g[2].array()).rowwise().sum();
    rho.col(4).array() += 2 * rho.col(5).array();
    rho.col(5).array() *= 0.5;
  }
}

template void spin_density<0>(const BatchDensity &, int, const GTOValues &,
                              Mat &, Mat *, MatRef);
template void spin_density<1>(const BatchDensity &, int, const GTOValues &,
                              Mat &, Mat *, MatRef);
template void spin_density<2>(const BatchDensity &, int, const GTOValues &,
                              Mat &, Mat *, MatRef);

template <int derivative_order, occ::qm::SpinorbitalKind spinorbital_kind>
bool process_grid_block(const BatchDensity &D,
                        const occ::gto::GTOValues &gto_vals,
                        Eigen::Ref<const Mat3N> points_block,
                        const Vec &weights_block,
                        const std::vector<DensityFunctional> &funcs, Mat &K,
                        double &energy, double &total_density_a,
                        double &total_density_b, double density_threshold) {

  const size_t npt = points_block.cols();
  size_t num_rows_factor =
      (spinorbital_kind == SpinorbitalKind::Unrestricted) ? 2 : 1;

  Mat rho(num_rows_factor * npt,
          occ::density::num_components(derivative_order));

  // Evaluate density at grid points
  {
    Mat Dphi;
    for (size_t sp = 0; sp < num_rows_factor; sp++)
      spin_density<derivative_order>(D, static_cast<int>(sp), gto_vals, Dphi,
                                     nullptr, rho.middleRows(sp * npt, npt));
  }

  // Calculate local total density for this block
  if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
    total_density_a += rho.col(0).dot(weights_block);
  } else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
    Vec rho_a_tmp = qm::block::a(rho.col(0));
    Vec rho_b_tmp = qm::block::b(rho.col(0));
    total_density_a += rho_a_tmp.dot(weights_block);
    total_density_b += rho_b_tmp.dot(weights_block);
  }

  // Skip computation if density is too small
  double max_density_block = rho.col(0).maxCoeff();
  if (max_density_block < density_threshold)
    return false;

  // Set up functional parameters
  const auto family = impl::family_for(derivative_order);
  DensityFunctional::Params params(npt, family, spinorbital_kind);
  impl::set_params<spinorbital_kind, derivative_order>(params, rho);

  // Evaluate functional
  DensityFunctional::Result res(npt, family, spinorbital_kind);
  for (const auto &func : funcs) {
    res += func.evaluate(params);
  }

  // Apply weights and compute potential matrix
  res.weight_by(weights_block);

  double block_energy = 0.0;
  xc_potential_matrix<spinorbital_kind, derivative_order>(res, rho, gto_vals, K,
                                                          block_energy);
  energy += block_energy;
  return true;
}

// Explicit template instantiations
template bool process_grid_block<0, occ::qm::SpinorbitalKind::Restricted>(
    const BatchDensity &, const occ::gto::GTOValues &, Eigen::Ref<const Mat3N>,
    const Vec &, const std::vector<DensityFunctional> &, Mat &, double &,
    double &, double &, double);
template bool process_grid_block<1, occ::qm::SpinorbitalKind::Restricted>(
    const BatchDensity &, const occ::gto::GTOValues &, Eigen::Ref<const Mat3N>,
    const Vec &, const std::vector<DensityFunctional> &, Mat &, double &,
    double &, double &, double);
template bool process_grid_block<2, occ::qm::SpinorbitalKind::Restricted>(
    const BatchDensity &, const occ::gto::GTOValues &, Eigen::Ref<const Mat3N>,
    const Vec &, const std::vector<DensityFunctional> &, Mat &, double &,
    double &, double &, double);
template bool process_grid_block<0, occ::qm::SpinorbitalKind::Unrestricted>(
    const BatchDensity &, const occ::gto::GTOValues &, Eigen::Ref<const Mat3N>,
    const Vec &, const std::vector<DensityFunctional> &, Mat &, double &,
    double &, double &, double);
template bool process_grid_block<1, occ::qm::SpinorbitalKind::Unrestricted>(
    const BatchDensity &, const occ::gto::GTOValues &, Eigen::Ref<const Mat3N>,
    const Vec &, const std::vector<DensityFunctional> &, Mat &, double &,
    double &, double &, double);
template bool process_grid_block<2, occ::qm::SpinorbitalKind::Unrestricted>(
    const BatchDensity &, const occ::gto::GTOValues &, Eigen::Ref<const Mat3N>,
    const Vec &, const std::vector<DensityFunctional> &, Mat &, double &,
    double &, double &, double);

} // namespace occ::dft::kernels
