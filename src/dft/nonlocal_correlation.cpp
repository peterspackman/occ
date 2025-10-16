#include <cmath>
#include <occ/core/constants.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/parallel.h>
#include <occ/dft/molecular_grid.h>
#include <occ/dft/nonlocal_correlation.h>
#include <occ/gto/density.h>

namespace occ::dft {

using occ::qm::SpinorbitalKind;
using ArrayRef = Eigen::Ref<Eigen::ArrayXd>;
using ArrayConstRef = Eigen::Ref<const Eigen::ArrayXd>;
using PointsRef = Eigen::Ref<const Mat3N>;

NonLocalCorrelationFunctional::NonLocalCorrelationFunctional() {}

void NonLocalCorrelationFunctional::set_integration_grid(
    const qm::AOBasis &basis, const GridSettings &settings) {
  m_nlc_atom_grids.clear();
  MolecularGrid nlc_grid(basis, settings);

  for (size_t i = 0; i < basis.atoms().size(); i++) {
    m_nlc_atom_grids.push_back(nlc_grid.get_partitioned_atom_grid(i));
  }
  size_t num_grid_points = std::accumulate(
      m_nlc_atom_grids.begin(), m_nlc_atom_grids.end(), 0.0,
      [&](double tot, const auto &grid) { return tot + grid.points.cols(); });
  occ::log::info("finished calculating NLC atom grids ({} points)",
                 num_grid_points);
}

void NonLocalCorrelationFunctional::set_parameters(const Parameters &params) {
  m_params = params;
}

std::pair<Vec, Mat>
vv10_kernel(ArrayConstRef rho, ArrayConstRef grad_rho, PointsRef points,
            ArrayConstRef weights,
            const NonLocalCorrelationFunctional::Parameters &params) {

  using Array = Eigen::ArrayXd;
  constexpr double pi = occ::constants::pi<double>;
  constexpr double pi4_3 = 4 * pi / 3;

  const double kappa_factor =
      params.vv10_b * 1.5 * pi * (std::pow(9 * pi, -1.0 / 6));
  const double beta =
      std::pow(3.0 / (params.vv10_b * params.vv10_b), 0.75) / 32.0;

  // outer grid
  Array rho_weighted = rho * weights;
  Array tmp = grad_rho / (rho * rho);
  tmp = params.vv10_C * tmp * tmp;
  Array omega0 = (tmp + pi4_3 * rho).sqrt();

  Array domega0_dr = (0.5 * pi4_3 * rho - 2.0 * tmp) / omega0;
  Array domega0_dg = tmp * rho / (grad_rho * omega0);
  Array kappa = kappa_factor * (rho.pow(1. / 6.));
  Array dkappa_dr = (1. / 6.) * kappa;

  Array F = Array::Zero(rho.rows());
  Array U = Array::Zero(rho.rows());
  Array W = Array::Zero(rho.rows());

  occ::timing::start(occ::timing::category::dft_nlc);

  // Use TBB parallel_for instead of round-robin scheduling
  auto kernel = [&](size_t i) {
      // TODO lift screening outside loops
      // Refactor F, U, W to copy to avoid false sharing?
      if (rho(i) < 1e-8)
        return;

      double k = kappa(i);
      double f = rho_weighted(i) / (2 * k * k * k);
      double u = f * (1.0 / k + 1.0 / (2 * k));
      double w = 0.0;

      for (int j = 0; j < i; j++) {
        if (rho(j) < 1e-8)
          continue;
        double r2 = (points.col(i) - points.col(j)).squaredNorm();
        double gi = r2 * omega0(i) + kappa(i);
        double gj = r2 * omega0(j) + kappa(j);
        double t = 2 * rho_weighted(j) / (gi * gj * (gi + gj));
        f += t;
        t *= 1.0 / gi + 1.0 / (gi + gj);
        u += t;
        w += t * r2;
      }
      F(i) = f * -1.5;
      U(i) = u;
      W(i) = w;
  };
  occ::parallel::parallel_for(size_t(0), size_t(points.cols()), kernel);
  occ::timing::stop(occ::timing::category::dft_nlc);

  Vec exc = (beta + 0.5 * F);
  exc.array() *= weights;

  Mat vxc(exc.rows(), 2);
  vxc.col(0) =
      (beta + F + 1.5 * (U * dkappa_dr + W * domega0_dr)) * weights.array();
  vxc.col(1) = (1.5 * W * domega0_dg) * weights;
  return {exc, vxc};
}

struct VV10GradientData {
  Vec exc;
  Mat vxc;
};

VV10GradientData
vv10_kernel_gradient(ArrayConstRef rho, ArrayConstRef grad_rho, PointsRef points,
                    ArrayConstRef weights, 
                    const NonLocalCorrelationFunctional::Parameters &params) {

  using Array = Eigen::ArrayXd;
  constexpr double pi = occ::constants::pi<double>;
  constexpr double pi4_3 = 4 * pi / 3;

  const double kappa_factor =
      params.vv10_b * 1.5 * pi * (std::pow(9 * pi, -1.0 / 6));
  const double beta =
      std::pow(3.0 / (params.vv10_b * params.vv10_b), 0.75) / 32.0;

  // outer grid
  Array rho_weighted = rho * weights;
  Array tmp = grad_rho / (rho * rho);
  tmp = params.vv10_C * tmp * tmp;
  Array omega0 = (tmp + pi4_3 * rho).sqrt();

  Array domega0_dr = (0.5 * pi4_3 * rho - 2.0 * tmp) / omega0;
  Array domega0_dg = tmp * rho / (grad_rho * omega0);
  Array kappa = kappa_factor * (rho.pow(1. / 6.));
  Array dkappa_dr = (1. / 6.) * kappa;

  Array F = Array::Zero(rho.rows());
  Array U = Array::Zero(rho.rows());
  Array W = Array::Zero(rho.rows());

  occ::timing::start(occ::timing::category::dft_nlc);

  // Use TBB parallel_for instead of round-robin scheduling
  auto kernel = [&](size_t i) {
      if (rho(i) < 1e-8)
        return;

      double k = kappa(i);
      double f = rho_weighted(i) / (2 * k * k * k);
      double u = f * (1.0 / k + 1.0 / (2 * k));
      double w = 0.0;

      for (int j = 0; j < i; j++) {
        if (rho(j) < 1e-8)
          continue;
        
        double r2 = (points.col(i) - points.col(j)).squaredNorm();
        double gi = r2 * omega0(i) + kappa(i);
        double gj = r2 * omega0(j) + kappa(j);
        double t = 2 * rho_weighted(j) / (gi * gj * (gi + gj));
        f += t;
        t *= 1.0 / gi + 1.0 / (gi + gj);
        u += t;
        w += t * r2;
      }
      F(i) = f * -1.5;
      U(i) = u;
      W(i) = w;
  };
  occ::parallel::parallel_for(size_t(0), size_t(points.cols()), kernel);
  occ::timing::stop(occ::timing::category::dft_nlc);

  Vec exc = (beta + 0.5 * F);
  exc.array() *= weights;

  Mat vxc(exc.rows(), 2);
  vxc.col(0) =
      (beta + F + 1.5 * (U * dkappa_dr + W * domega0_dr)) * weights.array();
  vxc.col(1) = (1.5 * W * domega0_dg) * weights;
  
  return {exc, vxc};
}

Vec grad_rho(Eigen::Ref<const Mat> rho) {
  const auto gx = rho.col(1);
  const auto gy = rho.col(2);
  const auto gz = rho.col(3);
  return gx.array() * gx.array() + gy.array() * gy.array() +
         gz.array() * gz.array();
}

// Compute grid displacement gradient contributions for VV10
// Based on Vydrov & Van Voorhis, JCP 133, 244103 (2010), Eqs. 25-26
// This computes ∂Φ(r,r')/∂R_atom where Φ is the VV10 kernel
Mat3N compute_vv10_grid_displacement_gradient(
    ArrayConstRef rho, ArrayConstRef grad_rho, PointsRef points,
    ArrayConstRef weights, const std::vector<AtomGrid> &atom_grids,
    const NonLocalCorrelationFunctional::Parameters &params) {

  using Array = Eigen::ArrayXd;
  constexpr double pi = occ::constants::pi<double>;
  constexpr double pi4_3 = 4 * pi / 3;
  constexpr double rho_thresh = 1e-8;
  constexpr double distance_cutoff = 15.0; // Bohr - grid pairs beyond this are screened

  const double kappa_factor =
      params.vv10_b * 1.5 * pi * (std::pow(9 * pi, -1.0 / 6));
  const double beta =
      std::pow(3.0 / (params.vv10_b * params.vv10_b), 0.75) / 32.0;

  // Precompute VV10 parameters at each grid point
  Array rho_weighted = rho * weights;
  Array tmp = grad_rho / (rho * rho);
  tmp = params.vv10_C * tmp * tmp;
  Array omega0 = (tmp + pi4_3 * rho).sqrt();
  Array kappa = kappa_factor * (rho.pow(1. / 6.));

  const size_t natoms = atom_grids.size();
  const size_t npt = points.cols();

  Mat3N gradient = Mat3N::Zero(3, natoms);

  // Map grid points to atoms
  std::vector<size_t> grid_to_atom(npt);
  size_t offset = 0;
  for (size_t atom_id = 0; atom_id < natoms; atom_id++) {
    const size_t npt_atom = atom_grids[atom_id].num_points();
    for (size_t i = 0; i < npt_atom; i++) {
      grid_to_atom[offset + i] = atom_id;
    }
    offset += npt_atom;
  }

  occ::timing::start(occ::timing::category::dft_nlc);

  // Thread-local storage for gradients
  occ::parallel::thread_local_storage<Mat3N> gradients_local(
      [natoms]() { return Mat3N::Zero(3, natoms); });

  // Parallelize over grid points
  // For each point i, compute interactions with all points j
  occ::parallel::parallel_for(size_t(0), npt, [&](size_t i) {
    if (rho(i) < rho_thresh)
      return;

    const auto &r_i = points.col(i);
    const double w_i = weights(i);
    const double rho_i = rho(i);
    const double omega_i = omega0(i);
    const double kappa_i = kappa(i);
    const size_t atom_i = grid_to_atom[i];

    auto &local_grad = gradients_local.local();

    for (size_t j = 0; j < npt; j++) {
      if (i == j || rho(j) < rho_thresh)
        continue;

      const auto r_ij = r_i - points.col(j);
      const double r2 = r_ij.squaredNorm();
      const double r_dist = std::sqrt(r2);

      // Distance screening for performance
      if (r_dist > distance_cutoff)
        continue;

      const double w_j = weights(j);
      const double rho_j = rho(j);
      const double omega_j = omega0(j);
      const double kappa_j = kappa(j);
      const size_t atom_j = grid_to_atom[j];

      // VV10 kernel: Φ = -1.5 * Θ / (g_i * g_j * (g_i + g_j))
      // where g = r² * ω₀ + κ
      const double g_i = r2 * omega_i + kappa_i;
      const double g_j = r2 * omega_j + kappa_j;
      const double Theta = 2.0 * rho_i * w_i * rho_j * w_j;
      const double denom = g_i * g_j * (g_i + g_j);

      if (std::abs(denom) < 1e-20)
        continue;

      const double Phi = -1.5 * Theta / denom;

      // Derivative of Φ with respect to r_ij
      // ∂Φ/∂r_ij = -1.5 * Θ * ∂/∂r_ij[1/(g_i * g_j * (g_i + g_j))]
      //
      // Let D = g_i * g_j * (g_i + g_j)
      // ∂D/∂r_ij = ∂g_i/∂r_ij * [g_j*(g_i + g_j) + g_i*g_j]
      //          + ∂g_j/∂r_ij * [g_i*(g_i + g_j) + g_i*g_j]
      //
      // ∂g/∂r_ij = 2*r_ij * ω₀  (since g = r² * ω₀ + κ, and κ doesn't depend on r_ij)

      const double dg_i_dr = 2.0 * omega_i;  // ∂g_i/∂r²
      const double dg_j_dr = 2.0 * omega_j;  // ∂g_j/∂r²

      const double dD_dr2 = dg_i_dr * (g_j * (g_i + g_j) + g_i * g_j) +
                            dg_j_dr * (g_i * (g_i + g_j) + g_i * g_j);

      // ∂Φ/∂r² = -1.5 * Θ * (-1/D²) * ∂D/∂r²
      const double dPhi_dr2 = 1.5 * Theta * dD_dr2 / (denom * denom);

      // ∂Φ/∂r_ij = ∂Φ/∂r² * ∂r²/∂r_ij = ∂Φ/∂r² * 2*r_ij
      // Force on atom = -∂E/∂R = -∂Φ/∂r_ij
      //
      // When atom_i moves: r_ij changes by +dR, so ∂r_ij/∂R_i = +I
      // When atom_j moves: r_ij changes by -dR, so ∂r_ij/∂R_j = -I

      Vec3 force = 2.0 * dPhi_dr2 * r_ij;  // This is ∂Φ/∂r_ij

      // Add contribution to atomic gradients
      // Grid point i belongs to atom_i, grid point j belongs to atom_j
      // Force from this interaction acts on both atoms
      local_grad.col(atom_i) -= force;  // Force on atom containing grid i
      local_grad.col(atom_j) += force;  // Equal and opposite on atom containing grid j
    }
  });

  // Reduce thread-local gradients
  for (const auto &local_grad : gradients_local) {
    gradient += local_grad;
  }

  occ::timing::stop(occ::timing::category::dft_nlc);

  return gradient;
}

NonLocalCorrelationFunctional::Result
NonLocalCorrelationFunctional::vv10(const qm::AOBasis &basis,
                                    const qm::MolecularOrbitals &mo) {
  if (mo.kind != SpinorbitalKind::Restricted)
    throw std::runtime_error(
        "Only restricted NLC functional implemented right now");
  constexpr int derivative_order = 1;
  constexpr SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted;

  const Mat D2 = 2 * mo.D;

  /*
  size_t num_rows_factor = 1;
  if (spinorbital_kind == SpinorbitalKind::Unrestricted)
      num_rows_factor = 2;
  */

  Mat Vxc = Mat::Zero(D2.rows(), D2.cols());

  size_t npt = 0;
  for (const auto &atom_grid_a : m_nlc_atom_grids) {
    npt += atom_grid_a.num_points();
  }

  Mat3N points(3, npt);
  Vec weights(npt);
  npt = 0;
  for (const auto &atom_grid_a : m_nlc_atom_grids) {
    points.block(0, npt, 3, atom_grid_a.num_points()) = atom_grid_a.points;
    weights.block(npt, 0, atom_grid_a.num_points(), 1) = atom_grid_a.weights;
    npt += atom_grid_a.num_points();
  }

  Mat rho(npt, occ::density::num_components(derivative_order));
  auto gto_vals = occ::gto::evaluate_basis(basis, points, derivative_order);

  occ::density::evaluate_density<derivative_order, spinorbital_kind>(
      D2, gto_vals, rho);
  Vec g = grad_rho(rho);

  double etot = 0.0;
  const auto &phi = gto_vals.phi;
  const auto &phi_x = gto_vals.phi_x.array();
  const auto &phi_y = gto_vals.phi_y.array();
  const auto &phi_z = gto_vals.phi_z.array();

  Vec exc;
  Mat vxc;
  std::tie(exc, vxc) = vv10_kernel(rho.col(0), g, points, weights, m_params);
  etot += exc.dot(rho.col(0));
  const auto &vrho = vxc.col(0).array();
  const auto &vsigma = vxc.col(1).array();
  Mat ktmp =
      phi.transpose() * (0.5 * (phi.array().colwise() * vrho) +
                         2 * (phi_x.colwise() * (rho.col(1).array() * vsigma)) +
                         2 * (phi_y.colwise() * (rho.col(2).array() * vsigma)) +
                         2 * (phi_z.colwise() * (rho.col(3).array() * vsigma)))
                            .matrix();
  Vxc.noalias() += ktmp + ktmp.transpose();
  return {etot, Vxc};
}

NonLocalCorrelationFunctional::GradientResult
NonLocalCorrelationFunctional::vv10_gradient(const qm::AOBasis &basis,
                                            const qm::MolecularOrbitals &mo) const {
  // WARNING: This gradient implementation is experimental and not fully tested.
  // It assumes post-SCF VV10 correction and does not include SCF response.
  if (mo.kind != SpinorbitalKind::Restricted)
    throw std::runtime_error(
        "Only restricted NLC functional gradients implemented right now");
  constexpr int derivative_order = 1;
  constexpr SpinorbitalKind spinorbital_kind = SpinorbitalKind::Restricted;

  const Mat D2 = 2 * mo.D;
  const size_t natoms = basis.atoms().size();
  const size_t nbf = basis.nbf();

  Mat Vxc = Mat::Zero(D2.rows(), D2.cols());
  Mat3N gradient = Mat3N::Zero(3, natoms);

  size_t npt = 0;
  for (const auto &atom_grid_a : m_nlc_atom_grids) {
    npt += atom_grid_a.num_points();
  }

  Mat3N points(3, npt);
  Vec weights(npt);
  npt = 0;
  for (const auto &atom_grid_a : m_nlc_atom_grids) {
    points.block(0, npt, 3, atom_grid_a.num_points()) = atom_grid_a.points;
    weights.block(npt, 0, atom_grid_a.num_points(), 1) = atom_grid_a.weights;
    npt += atom_grid_a.num_points();
  }

  Mat rho(npt, occ::density::num_components(derivative_order));
  // Need second derivatives for gradient calculation
  auto gto_vals = occ::gto::evaluate_basis(basis, points, derivative_order + 1);

  occ::density::evaluate_density<derivative_order, spinorbital_kind>(
      D2, gto_vals, rho);
  Vec g = grad_rho(rho);

  double etot = 0.0;
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

  auto result = vv10_kernel_gradient(rho.col(0), g, points, weights, m_params);
  etot += result.exc.dot(rho.col(0));
  
  const Array vrho = result.vxc.col(0).array() * 0.5;
  const Array vsigma = result.vxc.col(1).array();
  const Array dx = rho.col(1).array();
  const Array dy = rho.col(2).array();
  const Array dz = rho.col(3).array();

  // Compute Vxc for energy
  Mat ktmp =
      p.transpose() * (0.5 * (p.array().colwise() * (2.0 * vrho)) +
                         2 * (px.array().colwise() * (dx * vsigma)) +
                         2 * (py.array().colwise() * (dy * vsigma)) +
                         2 * (pz.array().colwise() * (dz * vsigma)))
                            .matrix();
  Vxc.noalias() += ktmp + ktmp.transpose();

  // Compute gradient matrices Vx, Vy, Vz following the pattern in accumulate_gradient_contribution_R
  Mat Vx = Mat::Zero(nbf, nbf);
  Mat Vy = Mat::Zero(nbf, nbf);
  Mat Vz = Mat::Zero(nbf, nbf);

  // First contribution
  Mat aow = (p.array().colwise() * vrho);
  aow.array() += px.array().colwise() * (2.0 * vsigma * dx);
  aow.array() += py.array().colwise() * (2.0 * vsigma * dy);
  aow.array() += pz.array().colwise() * (2.0 * vsigma * dz);

  Vx.noalias() += px.transpose() * aow;
  Vy.noalias() += py.transpose() * aow;
  Vz.noalias() += pz.transpose() * aow;

  // X contribution
  aow = px.array().colwise() * vrho;
  aow.array() += pxx.array().colwise() * (2.0 * vsigma * dx);
  aow.array() += pxy.array().colwise() * (2.0 * vsigma * dy);
  aow.array() += pxz.array().colwise() * (2.0 * vsigma * dz);
  Vx.noalias() += (p.transpose() * aow).transpose();

  // Y contribution
  aow = py.array().colwise() * vrho;
  aow.array() += pxy.array().colwise() * (2.0 * vsigma * dx);
  aow.array() += pyy.array().colwise() * (2.0 * vsigma * dy);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma * dz);
  Vy.noalias() += (p.transpose() * aow).transpose();

  // Z contribution
  aow = pz.array().colwise() * vrho;
  aow.array() += pxz.array().colwise() * (2.0 * vsigma * dx);
  aow.array() += pyz.array().colwise() * (2.0 * vsigma * dy);
  aow.array() += pzz.array().colwise() * (2.0 * vsigma * dz);
  Vz.noalias() += (p.transpose() * aow).transpose();

  // Distribute to atoms using basis function to atom mapping (XC potential gradient)
  const auto &bf_to_atom = basis.bf_to_atom();
  for (size_t mu = 0; mu < nbf; mu++) {
    int atom_mu = bf_to_atom[mu];
    for (size_t nu = 0; nu < nbf; nu++) {
      double Dval = 2.0 * mo.D(mu, nu); // Factor of 2 for restricted case
      if (std::abs(Dval) < 1e-12)
        continue;

      // Add contributions to gradients
      gradient(0, atom_mu) -= Vx(mu, nu) * Dval;
      gradient(1, atom_mu) -= Vy(mu, nu) * Dval;
      gradient(2, atom_mu) -= Vz(mu, nu) * Dval;
    }
  }

  // Add grid displacement gradient contributions (∂Φ(r,r')/∂R terms)
  occ::log::debug("Computing VV10 grid displacement gradients");
  Mat3N grid_gradient = compute_vv10_grid_displacement_gradient(
      rho.col(0), g, points, weights, m_nlc_atom_grids, m_params);

  gradient += grid_gradient;

  return {etot, Vxc, gradient};
}

NonLocalCorrelationFunctional::Result
NonLocalCorrelationFunctional::operator()(const qm::AOBasis &basis,
                                          const qm::MolecularOrbitals &mo) {
  return vv10(basis, mo);
}

NonLocalCorrelationFunctional::GradientResult
NonLocalCorrelationFunctional::compute_gradient(const qm::AOBasis &basis,
                                               const qm::MolecularOrbitals &mo) const {
  return vv10_gradient(basis, mo);
}

} // namespace occ::dft
