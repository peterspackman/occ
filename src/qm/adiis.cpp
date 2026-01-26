#include <LBFGSB.h>
#include <fmt/core.h>
#include <occ/qm/adiis.h>
#include <occ/qm/expectation.h>
#include <occ/qm/opmatrix.h>

namespace occ::qm {

ADIIS::ADIIS(size_t start, size_t diis_subspace)
    : m_error{1.0}, m_start{start}, m_max_subspace_size{diis_subspace} {}

void ADIIS::reset() {
  m_density_matrices.clear();
  m_fock_matrices.clear();
  m_iter = 0;
  m_error = 1.0;
  m_nskip = 0;
}

void ADIIS::minimize_coefficients(SpinorbitalKind kind) {
  using LBFGSpp::LBFGSBParam;
  using LBFGSpp::LBFGSBSolver;

  const size_t nvec = m_density_matrices.size() - m_nskip;
  if (nvec == 0) {
    throw std::domain_error("ADIIS::minimize_coefficients: no vectors");
  }

  // Reference point is the most recent density/Fock
  const Mat &D_n = m_density_matrices.back();
  const Mat &F_n = m_fock_matrices.back();

  // Compute linear terms: e_i = Tr[(D_i - D_n) * F_n]
  Vec e_lin(nvec);
  for (size_t i = 0; i < nvec; i++) {
    Mat dD = m_density_matrices[i + m_nskip] - D_n;
    e_lin(i) = expectation(kind, dD, F_n);
  }

  // Compute quadratic terms: B_ij = Tr[(D_i - D_n) * (F_j - F_n)]
  Mat B = Mat::Zero(nvec, nvec);
  for (size_t i = 0; i < nvec; i++) {
    Mat dD_i = m_density_matrices[i + m_nskip] - D_n;
    for (size_t j = 0; j < nvec; j++) {
      Mat dF_j = m_fock_matrices[j + m_nskip] - F_n;
      B(i, j) = expectation(kind, dD_i, dF_j);
    }
  }

  // Set up L-BFGS-B optimizer
  LBFGSBParam<double> param;
  param.epsilon = 1e-9;
  param.max_iterations = 100;
  param.max_linesearch = 100;

  LBFGSBSolver<double> solver(param);
  Vec lb = Vec::Zero(nvec);
  Vec ub = Vec::Constant(nvec, 5.0);

  // ADIIS cost function:
  // f(c) = sum_i c_i * e_lin(i) + 0.5 * sum_ij c_i * c_j * B(i,j)
  // Using squared parameterization: c_i = x_i^2 / sum(x^2)
  auto func = [&](const Vec &x, Vec &grad) {
    Vec x2 = x.array() * x.array();
    double s = x2.sum();
    Vec c = x2 / s;

    // Function value
    double f = c.dot(e_lin) + 0.5 * c.dot(B * c);

    // Gradient w.r.t. c
    Vec df_dc = e_lin + B * c;

    // Chain rule: grad_x = df/dc * dc/dx
    // dc_i/dx_j = 2*x_j/s * (delta_ij - c_i)
    Vec grad_x = Vec::Zero(nvec);
    for (size_t j = 0; j < nvec; j++) {
      double sum = 0.0;
      for (size_t i = 0; i < nvec; i++) {
        double delta = (i == j) ? 1.0 : 0.0;
        sum += df_dc(i) * 2 * x(j) / s * (delta - c(i));
      }
      grad_x(j) = sum;
    }
    grad = grad_x;

    return f;
  };

  Vec x = Vec::Constant(nvec, 1.0);
  double fx;
  solver.minimize(func, x, fx, lb, ub);

  // Convert back to coefficients
  Vec x2 = x.array() * x.array();
  m_coeffs = x2 / x2.sum();
}

Mat ADIIS::update(SpinorbitalKind kind, const Mat &D, const Mat &F) {
  m_iter++;

  // Manage subspace size
  if (m_density_matrices.size() == m_max_subspace_size) {
    m_density_matrices.pop_front();
    m_fock_matrices.pop_front();
  }

  m_density_matrices.push_back(D);
  m_fock_matrices.push_back(F);

  const size_t nvec = m_density_matrices.size();

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
