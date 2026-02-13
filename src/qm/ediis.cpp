#include <LBFGSB.h>
#include <occ/qm/ediis.h>
#include <occ/qm/expectation.h>

namespace occ::qm {

EDIIS::EDIIS(size_t start, size_t diis_subspace)
    : m_error{1.0}, m_start{start}, m_max_subspace_size{diis_subspace} {}

void EDIIS::reset() {
  m_density_matrices.clear();
  m_fock_matrices.clear();
  m_energies.clear();
  m_iter = 0;
  m_error = 1.0;
  m_nskip = 0;
}

void EDIIS::minimize_coefficients(SpinorbitalKind kind) {
  using LBFGSpp::LBFGSBParam;
  using LBFGSpp::LBFGSBSolver;

  const size_t nvec = m_energies.size() - m_nskip;
  if (nvec == 0) {
    throw std::domain_error("EDIIS::minimize_coefficients: no vectors");
  }

  // Build the df matrix: df(i,j) = 2 * Tr(D_i * F_j)
  Mat df = Mat::Zero(nvec, nvec);
  for (size_t i = 0; i < nvec; i++) {
    for (size_t j = 0; j < nvec; j++) {
      df(i, j) = 2 * expectation(kind,
                                 m_density_matrices[i + m_nskip],
                                 m_fock_matrices[j + m_nskip]);
    }
  }

  // Build df2 matrix for the quadratic term
  Eigen::ArrayXd diagonal = df.diagonal();
  Mat df2 = -df - df.transpose();
  for (size_t i = 0; i < nvec; i++) {
    df2.row(i).array() += diagonal;
    df2.col(i).array() += diagonal;
  }

  // Energy vector
  Vec energies(nvec);
  for (size_t i = 0; i < nvec; i++) {
    energies(i) = m_energies[i + m_nskip];
  }

  // Set up L-BFGS-B optimizer
  LBFGSBParam<double> param;
  param.epsilon = 1e-9;
  param.max_iterations = 100;
  param.max_linesearch = 100;

  LBFGSBSolver<double> solver(param);
  Vec lb = Vec::Zero(nvec);
  Vec ub = Vec::Constant(nvec, 5.0);

  // Cost function: minimize E(c) = sum_i c_i * E_i - sum_ij c_i * c_j * df2(i,j)
  // Using squared parameterization: c_i = x_i^2 / sum(x^2) ensures sum(c) = 1, c >= 0
  auto func = [&](const Vec &x, Vec &grad) {
    Vec x2 = x.array() * x.array();
    double s = x2.sum();
    Vec c = x2 / s;

    // Compute gradient of energy w.r.t. c
    Vec dE_dc = energies;
    dE_dc.noalias() -= 2 * df2 * c;

    // Chain rule: grad_x = dE/dc * dc/dx
    // dc_i/dx_j = 2*x_i/s * (delta_ij - c_i)
    Mat dc_dx = Mat::Zero(nvec, nvec);
    for (size_t i = 0; i < nvec; i++) {
      for (size_t j = 0; j < nvec; j++) {
        double delta = (i == j) ? 1.0 : 0.0;
        dc_dx(i, j) = 2 * x(j) / s * (delta - c(i));
      }
    }
    grad = dc_dx.transpose() * dE_dc;

    // Compute function value
    double result = c.dot(energies) - c.dot(df2 * c);
    return result;
  };

  Vec x = Vec::Constant(nvec, 1.0);
  double fx;
  solver.minimize(func, x, fx, lb, ub);

  // Convert back to coefficients
  Vec x2 = x.array() * x.array();
  m_coeffs = x2 / x2.sum();
}

Mat EDIIS::update(SpinorbitalKind kind, const Mat &D, const Mat &F, double e) {
  m_iter++;

  // Manage subspace size
  if (m_energies.size() == m_max_subspace_size) {
    m_density_matrices.pop_front();
    m_fock_matrices.pop_front();
    m_energies.pop_front();
  }

  m_density_matrices.push_back(D);
  m_fock_matrices.push_back(F);
  m_energies.push_back(e);

  const size_t nvec = m_energies.size();

  // For first iteration or before start, just return F
  if (m_iter < m_start || nvec < 2) {
    m_coeffs = Vec::Ones(1);
    m_error = 1.0;
    return F;
  }

  // Try to minimize, skipping oldest vectors if ill-conditioned
  bool valid = false;
  m_nskip = 0;
  while (!valid && m_nskip < nvec - 1) {
    try {
      minimize_coefficients(kind);
      valid = true;
    } catch (const std::exception &) {
      m_nskip++;
    }
  }

  if (!valid) {
    // Fall back to just returning the latest Fock matrix
    m_coeffs = Vec::Zero(nvec);
    m_coeffs(nvec - 1) = 1.0;
  }

  // Build extrapolated Fock matrix
  Mat result = Mat::Zero(F.rows(), F.cols());
  for (size_t i = m_nskip; i < nvec; i++) {
    result += m_coeffs(i - m_nskip) * m_fock_matrices[i];
  }

  // Update error estimate based on coefficient distribution
  // Error is small when most weight is on latest vector
  m_error = 1.0 - m_coeffs(m_coeffs.size() - 1);

  return result;
}

} // namespace occ::qm
