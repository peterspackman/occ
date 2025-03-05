#include <occ/dft/dft_gradient_kernels.h>
#include <occ/gto/density.h>
#include <occ/qm/opmatrix.h>

constexpr auto U = occ::qm::SpinorbitalKind::Unrestricted;
constexpr auto R = occ::qm::SpinorbitalKind::Restricted;

namespace occ::dft::kernels {

template <int derivative_order>
void accumulate_gradient_contribution_R(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const occ::gto::GTOValues &gto_vals,
                                        MatRef Vx, MatRef Vy, MatRef Vz,
                                        double fac = 1.0);
template <int derivative_order>
void accumulate_gradient_contribution_U(const DensityFunctional::Result &res,
                                        MatConstRef rho,
                                        const occ::gto::GTOValues &gto_vals,
                                        MatRef Vxa, MatRef Vya, MatRef Vza,
                                        MatRef Vxb, MatRef Vyb, MatRef Vzb);

template <>
inline void accumulate_gradient_contribution_R<0>(
    const DensityFunctional::Result &res, MatConstRef rho,
    const occ::gto::GTOValues &gto_vals, MatRef Vx, MatRef Vy, MatRef Vz,
    double fac) {

  const size_t npt = rho.rows();
  const auto &phi = gto_vals.phi;
  const auto &dx = gto_vals.phi_x;
  const auto &dy = gto_vals.phi_y;
  const auto &dz = gto_vals.phi_z;

  const Vec vrho = res.vrho.col(0) * fac;

  Vx.noalias() +=
      dx.transpose() * (phi.array().colwise() * vrho.array()).matrix();
  Vy.noalias() +=
      dy.transpose() * (phi.array().colwise() * vrho.array()).matrix();
  Vz.noalias() +=
      dz.transpose() * (phi.array().colwise() * vrho.array()).matrix();
}

template <>
inline void accumulate_gradient_contribution_R<1>(
    const DensityFunctional::Result &res, MatConstRef rho,
    const occ::gto::GTOValues &gto_vals, MatRef Vx, MatRef Vy, MatRef Vz,
    double fac) {
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

  const Array vrho = res.vrho.col(0).array() * (fac * 0.5); // wv[0]
  const Array vsigma = res.vsigma.col(0).array() * fac;

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
inline void accumulate_gradient_contribution_R<2>(
    const DensityFunctional::Result &res, MatConstRef rho,
    const occ::gto::GTOValues &gto_vals, MatRef Vx, MatRef Vy, MatRef Vz,
    double fac) {

  throw std::runtime_error("Not implemented: MGGA gradients");
}

template <>
inline void accumulate_gradient_contribution_U<0>(
    const DensityFunctional::Result &res, MatConstRef rho,
    const occ::gto::GTOValues &gto_vals, MatRef Vx_a, MatRef Vy_a, MatRef Vz_a,
    MatRef Vx_b, MatRef Vy_b, MatRef Vz_b) {

  throw std::runtime_error("Not implemented: unrestricted LDA gradients");
}

template <>
inline void accumulate_gradient_contribution_U<1>(
    const DensityFunctional::Result &res, MatConstRef rho,
    const occ::gto::GTOValues &gto_vals, MatRef Vx_a, MatRef Vy_a, MatRef Vz_a,
    MatRef Vx_b, MatRef Vy_b, MatRef Vz_b) {

  throw std::runtime_error("Not implemented: unrestricted GGA gradients");
}

template <>
inline void accumulate_gradient_contribution_U<2>(
    const DensityFunctional::Result &res, MatConstRef rho,
    const occ::gto::GTOValues &gto_vals, MatRef Vx_a, MatRef Vy_a, MatRef Vz_a,
    MatRef Vx_b, MatRef Vy_b, MatRef Vz_b) {

  throw std::runtime_error("Not implemented: MGGA gradients");
}

template <int derivative_order, SpinorbitalKind spinorbital_kind>
void process_grid_block_gradient(const Mat &D,
                                 const occ::gto::GTOValues &gto_vals,
                                 Eigen::Ref<const Mat3N> points_block,
                                 const Vec &weights_block,
                                 const std::vector<DensityFunctional> &funcs,
                                 Mat3N &gradient, double density_threshold,
                                 const qm::AOBasis &basis) {
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
    accumulate_gradient_contribution_R<derivative_order>(res, rho, gto_vals, Vx,
                                                         Vy, Vz);

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
    const auto Da = qm::block::a(D);
    const auto Db = qm::block::b(D);

    // Initialize gradient matrices for alpha and beta
    Mat Vx_a = Mat::Zero(nbf, nbf);
    Mat Vy_a = Mat::Zero(nbf, nbf);
    Mat Vz_a = Mat::Zero(nbf, nbf);

    Mat Vx_b = Mat::Zero(nbf, nbf);
    Mat Vy_b = Mat::Zero(nbf, nbf);
    Mat Vz_b = Mat::Zero(nbf, nbf);

    // Accumulate all gradient contributions based on derivative order
    accumulate_gradient_contribution_U<derivative_order>(
        res, rho, gto_vals, Vx_a, Vy_a, Vz_a, Vx_b, Vy_b, Vz_b);

    // Distribute to atoms
    for (size_t mu = 0; mu < nbf; mu++) {
      int atom_mu = bf_to_atom[mu];

      for (size_t nu = 0; nu < nbf; nu++) {
        const double Dvala = Da(mu, nu);
        const double Dvalb = Db(mu, nu);
        if (std::abs(Dvala + Dvalb) < 1e-12)
          continue;

        // Contribution to atom_mu
        gradient(0, atom_mu) -= Vx_a(mu, nu) * Dvala + Vx_b(mu, nu) * Dvalb;
        gradient(1, atom_mu) -= Vy_a(mu, nu) * Dvala + Vy_b(mu, nu) * Dvalb;
        gradient(2, atom_mu) -= Vz_a(mu, nu) * Dvala + Vz_b(mu, nu) * Dvalb;
      }
    }
  }
}

template 
void process_grid_block_gradient<0, R>(const Mat &D,
                                 const occ::gto::GTOValues &gto_vals,
                                 Eigen::Ref<const Mat3N> points_block,
                                 const Vec &weights_block,
                                 const std::vector<DensityFunctional> &funcs,
                                 Mat3N &gradient, double density_threshold,
                                 const qm::AOBasis &basis);

template 
void process_grid_block_gradient<1, R>(const Mat &D,
                                 const occ::gto::GTOValues &gto_vals,
                                 Eigen::Ref<const Mat3N> points_block,
                                 const Vec &weights_block,
                                 const std::vector<DensityFunctional> &funcs,
                                 Mat3N &gradient, double density_threshold,
                                 const qm::AOBasis &basis);

template 
void process_grid_block_gradient<2, R>(const Mat &D,
                                 const occ::gto::GTOValues &gto_vals,
                                 Eigen::Ref<const Mat3N> points_block,
                                 const Vec &weights_block,
                                 const std::vector<DensityFunctional> &funcs,
                                 Mat3N &gradient, double density_threshold,
                                 const qm::AOBasis &basis);
template 
void process_grid_block_gradient<0, U>(const Mat &D,
                                 const occ::gto::GTOValues &gto_vals,
                                 Eigen::Ref<const Mat3N> points_block,
                                 const Vec &weights_block,
                                 const std::vector<DensityFunctional> &funcs,
                                 Mat3N &gradient, double density_threshold,
                                 const qm::AOBasis &basis);
template 
void process_grid_block_gradient<1, U>(const Mat &D,
                                 const occ::gto::GTOValues &gto_vals,
                                 Eigen::Ref<const Mat3N> points_block,
                                 const Vec &weights_block,
                                 const std::vector<DensityFunctional> &funcs,
                                 Mat3N &gradient, double density_threshold,
                                 const qm::AOBasis &basis);
template 
void process_grid_block_gradient<2, U>(const Mat &D,
                                 const occ::gto::GTOValues &gto_vals,
                                 Eigen::Ref<const Mat3N> points_block,
                                 const Vec &weights_block,
                                 const std::vector<DensityFunctional> &funcs,
                                 Mat3N &gradient, double density_threshold,
                                 const qm::AOBasis &basis);

} // namespace occ::dft::kernels
