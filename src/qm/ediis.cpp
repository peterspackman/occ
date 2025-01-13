#include <LBFGSB.h>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <occ/qm/ediis.h>
#include <occ/qm/expectation.h>
#include <occ/qm/opmatrix.h>

namespace occ::qm {

EDIIS::EDIIS(size_t start, size_t diis_subspace)
    : m_error{0}, m_start{start}, m_diis_subspace_size{diis_subspace},
      m_previous_coeffs(Vec::Zero(diis_subspace)) {}

void EDIIS::minimize_coefficients() {
  using LBFGSpp::LBFGSBParam;
  using LBFGSpp::LBFGSBSolver;

  LBFGSBParam<double> param; // New parameter class
  param.epsilon = 1e-9;
  param.max_iterations = 100;
  param.max_linesearch = 100;

  // Create solver and function object
  LBFGSBSolver<double> solver(param); // New solver class
  Vec lb = Vec::Zero(m_diis_subspace_size);
  Vec ub = Vec::Constant(m_diis_subspace_size, 5.0);

  Mat df = Mat::Zero(m_diis_subspace_size, m_diis_subspace_size);
  for (size_t i = m_nskip; i < m_energies.size(); i++) {
    for (size_t j = m_nskip; j < m_energies.size(); j++) {
      df(i, j) = 2 * expectation<SpinorbitalKind::Restricted>(
                         m_density_matrices[i], m_fock_matrices[j]);
    }
  }
  Eigen::ArrayXd diagonal = df.diagonal();
  Mat df2 = -df - df.transpose();
  for (size_t i = 0; i < m_diis_subspace_size; i++) {
    df2.row(i).array() += diagonal;
    df2.col(i).array() += diagonal;
  }

  auto func = [&](const Vec &coeffs, Vec &grad) {
    Vec x2 = coeffs.array() * coeffs.array();
    double s = x2.sum();
    Vec c = x2 / s;

    // grad
    Vec fc = Vec::Zero(m_diis_subspace_size);
    for (size_t i = m_nskip; i < m_energies.size(); i++) {
      fc(i) = m_energies[i];
    }
    fc -= 2 * c.transpose() * df2;
    Mat cx = (coeffs.array() * s).matrix().asDiagonal();
    cx -= (x2 * coeffs.transpose());
    cx *= 2 / (s * s);
    grad = fc.transpose() * cx;

    double result = 0.0;
    for (size_t i = 0; i < coeffs.rows(); i++) {
      result += c(i) * m_energies[i];
      result -= c(i) * df2.row(i).dot(c);
    }
    return result;
  };
  Vec x = Vec::Constant(m_diis_subspace_size, 1.0);
  double fx;
  solver.minimize(func, x, fx, lb, ub);
  Vec coeffs = x.array() * x.array();
  coeffs /= coeffs.sum();
  m_previous_coeffs = coeffs;
}

Mat EDIIS::update(const Mat &D, const Mat &F, double e) {
  if (m_energies.size() == m_diis_subspace_size) {
    m_density_matrices.pop_front();
    m_fock_matrices.pop_front();
    m_energies.pop_front();
  }
  m_density_matrices.push_back(D);
  m_fock_matrices.push_back(F);
  m_energies.push_back(e);

  bool valid_direction = false;
  m_nskip = 0;
  do {
    try {
      minimize_coefficients();
      valid_direction = true;
    } catch (std::logic_error &e) {
      m_nskip++;
    } catch (std::runtime_error &e) {
      m_nskip++;
    }
    if (m_nskip >= m_energies.size()) {
      throw std::domain_error("EDIIS::update: poorly-conditioned system");
    }
  } while (!valid_direction);

  Mat result = m_previous_coeffs(0) * m_fock_matrices[0];
  for (size_t i = 1; i < m_fock_matrices.size(); i++) {
    result += m_previous_coeffs(i) * m_fock_matrices[i];
  }
  return result;
}

} // namespace occ::qm
