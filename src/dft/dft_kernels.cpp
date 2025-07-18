#include <occ/dft/dft_kernels.h>
#include <occ/dft/xc_potential_matrix.h>
#include <occ/gto/density.h>
#include <occ/qm/opmatrix.h>

namespace occ::dft::kernels {

template <int derivative_order, occ::qm::SpinorbitalKind spinorbital_kind>
void process_grid_block(const Mat &D, const occ::gto::GTOValues &gto_vals,
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
  occ::density::evaluate_density<derivative_order, spinorbital_kind>(
      D, gto_vals, rho);

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
    return;

  // Set up functional parameters
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

  // Apply weights and compute potential matrix
  res.weight_by(weights_block);

  Mat KK = Mat::Zero(K.rows(), K.cols());
  double block_energy = 0.0;
  xc_potential_matrix<spinorbital_kind, derivative_order>(res, rho, gto_vals,
                                                          KK, block_energy);

  K.noalias() += KK;
  energy += block_energy;
}

// Explicit template instantiations
template void process_grid_block<0, occ::qm::SpinorbitalKind::Restricted>(
    const Mat &, const occ::gto::GTOValues &, Eigen::Ref<const Mat3N>,
    const Vec &, const std::vector<DensityFunctional> &, Mat &, double &,
    double &, double &, double);
template void process_grid_block<1, occ::qm::SpinorbitalKind::Restricted>(
    const Mat &, const occ::gto::GTOValues &, Eigen::Ref<const Mat3N>,
    const Vec &, const std::vector<DensityFunctional> &, Mat &, double &,
    double &, double &, double);
template void process_grid_block<2, occ::qm::SpinorbitalKind::Restricted>(
    const Mat &, const occ::gto::GTOValues &, Eigen::Ref<const Mat3N>,
    const Vec &, const std::vector<DensityFunctional> &, Mat &, double &,
    double &, double &, double);
template void process_grid_block<0, occ::qm::SpinorbitalKind::Unrestricted>(
    const Mat &, const occ::gto::GTOValues &, Eigen::Ref<const Mat3N>,
    const Vec &, const std::vector<DensityFunctional> &, Mat &, double &,
    double &, double &, double);
template void process_grid_block<1, occ::qm::SpinorbitalKind::Unrestricted>(
    const Mat &, const occ::gto::GTOValues &, Eigen::Ref<const Mat3N>,
    const Vec &, const std::vector<DensityFunctional> &, Mat &, double &,
    double &, double &, double);
template void process_grid_block<2, occ::qm::SpinorbitalKind::Unrestricted>(
    const Mat &, const occ::gto::GTOValues &, Eigen::Ref<const Mat3N>,
    const Vec &, const std::vector<DensityFunctional> &, Mat &, double &,
    double &, double &, double);

} // namespace occ::dft::kernels
