#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/dft/dft_method.h>
#include <occ/dft/functional.h>
#include <occ/gto/gto.h>
#include <occ/qm/spinorbital.h>

namespace occ::dft::impl {
using occ::Mat;
using occ::dft::DensityFunctional;
using occ::qm::SpinorbitalKind;

namespace block = occ::qm::block;

template <SpinorbitalKind spinorbital_kind, int derivative_order>
void set_params(DensityFunctional::Params &params, const Mat &rho) {
  if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
    params.rho.col(0) = rho.col(0);
  } else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
    // correct assignment
    params.rho.col(0) = block::a(rho.col(0));
    params.rho.col(1) = block::b(rho.col(0));
  }

  if constexpr (derivative_order > 0) {
    if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
      params.sigma.col(0) = (rho.block(0, 1, rho.rows(), 3).array() *
                             rho.block(0, 1, rho.rows(), 3).array())
                                .rowwise()
                                .sum();
    } else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
      const auto rho_a = block::a(rho.array());
      const auto rho_b = block::b(rho.array());
      const auto &dx_rho_a = rho_a.col(1);
      const auto &dy_rho_a = rho_a.col(2);
      const auto &dz_rho_a = rho_a.col(3);
      const auto &dx_rho_b = rho_b.col(1);
      const auto &dy_rho_b = rho_b.col(2);
      const auto &dz_rho_b = rho_b.col(3);
      params.sigma.col(0) =
          dx_rho_a * dx_rho_a + dy_rho_a * dy_rho_a + dz_rho_a * dz_rho_a;
      params.sigma.col(1) =
          dx_rho_a * dx_rho_b + dy_rho_a * dy_rho_b + dz_rho_a * dz_rho_b;
      params.sigma.col(2) =
          dx_rho_b * dx_rho_b + dy_rho_b * dy_rho_b + dz_rho_b * dz_rho_b;
    }
  }
  if constexpr (derivative_order > 1) {
    if constexpr (spinorbital_kind == SpinorbitalKind::Restricted) {
      params.laplacian.col(0) = rho.col(4);
      params.tau.col(0) = rho.col(5);
    } else if constexpr (spinorbital_kind == SpinorbitalKind::Unrestricted) {
      params.laplacian.col(0) = block::a(rho.col(4));
      params.laplacian.col(1) = block::b(rho.col(4));
      params.tau.col(0) = block::a(rho.col(5));
      params.tau.col(1) = block::b(rho.col(5));
    }
  }
}

} // namespace occ::dft::impl

namespace occ::dft::kernels {

template <int derivative_order, occ::qm::SpinorbitalKind spinorbital_kind>
void process_grid_block(const Mat &D, const occ::gto::GTOValues &gto_vals,
                        Eigen::Ref<const Mat3N> points_block,
                        const Vec &weights_block,
                        const std::vector<DensityFunctional> &funcs, Mat &K,
                        double &energy, double &total_density_a,
                        double &total_density_b, double density_threshold);

} // namespace occ::dft::kernels
