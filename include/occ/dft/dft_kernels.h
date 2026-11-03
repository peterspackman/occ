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

/// The functional family whose inputs a density derivative order provides:
/// 0 LDA, 1 GGA, 2 meta-GGA (tau).
inline DensityFunctional::Family family_for(int derivative_order) {
  using Family = DensityFunctional::Family;
  return derivative_order == 0   ? Family::LDA
         : derivative_order == 1 ? Family::GGA
                                 : Family::MGGA;
}

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

namespace occ::qm {
struct MolecularOrbitals;
}

namespace occ::dft::kernels {

/// A batch's density matrix: D_s for spin s is rows [s n, (s + 1) n) of D.
/// When F is set, D_s = F_s F_s^T as well (F_s the rows of F likewise, a
/// few columns: the occupied orbitals), and products X D_s are formed as
/// (X F_s) F_s^T, which costs 2 k / n of the direct product.
struct BatchDensity {
  const Mat *D{nullptr};
  const Mat *F{nullptr};
  Eigen::Index nbf{0};

  /// out = X D_s
  void times(const Mat &X, int s, Mat &out) const;
};

/// Factor F of the (doubled) density matrix D2 = 2 mo.D, D2_s = F_s F_s^T
/// for each spin block, from the occupied orbitals; empty unless mo.D
/// really is that product (to 1e-12 relative), e.g. for a guess density.
Mat density_factor(const occ::qm::MolecularOrbitals &mo, const Mat &D2,
                   int nspin);

/// rho (one spin's rows) from the batch's basis values, keeping the
/// products with D: Dphi = phi D_s, and for meta-GGAs Dgrad[i] = phi_i D_s.
/// The same arithmetic as occ::density::evaluate_density.
template <int derivative_order>
void spin_density(const BatchDensity &dens, int s,
                  const occ::gto::GTOValues &gv, Mat &Dphi, Mat *Dgrad,
                  MatRef rho);

// XC energy and potential of one batch of grid points. Assigns K = H, with
// the batch's V_xc = H + H^T (per spin block for unrestricted), and returns
// false, leaving K untouched, when the batch's density is negligible.
template <int derivative_order, occ::qm::SpinorbitalKind spinorbital_kind>
bool process_grid_block(const BatchDensity &D,
                        const occ::gto::GTOValues &gto_vals,
                        Eigen::Ref<const Mat3N> points_block,
                        const Vec &weights_block,
                        const std::vector<DensityFunctional> &funcs, Mat &K,
                        double &energy, double &total_density_a,
                        double &total_density_b, double density_threshold);

} // namespace occ::dft::kernels
