#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/solvent/cosmo.h>
#include <occ/solvent/parameters.h>
#include <occ/xtb/cpcmx.h>
#include <stdexcept>

namespace occ::xtb {

namespace {

// Build the cavity → cavity COSMO response matrix A.
//   off-diag(i, j) = 1 / |r_i − r_j|
//   diag(i)        = 1.07 · √(4π / S_i)
// Same diagonal convention as `occ::solvent::COSMO`, so the LU is comparable.
Mat build_response_matrix(const Mat3N &points, const Vec &areas) {
  const Eigen::Index n = points.cols();
  Mat A(n, n);
  for (Eigen::Index i = 0; i < n; ++i) {
    for (Eigen::Index j = i + 1; j < n; ++j) {
      const double d = (points.col(i) - points.col(j)).norm();
      const double off = (d > 1e-6) ? 1.0 / d : 0.0;
      A(i, j) = off;
      A(j, i) = off;
    }
  }
  // 1.07 · √(4π / S_i). 1.07 · √(4π) ≈ 3.793051240937804.
  A.diagonal().array() = 3.793051240937804 / areas.array().sqrt();
  return A;
}

// Build the cavity → atoms Coulomb matrix B(i, a) = 1 / |r_i − R_a|.
Mat build_atom_cavity_coulomb(const Mat3N &surface_points,
                              const Mat3N &atom_positions) {
  const Eigen::Index ncav = surface_points.cols();
  const Eigen::Index natom = atom_positions.cols();
  Mat B(ncav, natom);
  for (Eigen::Index a = 0; a < natom; ++a) {
    for (Eigen::Index i = 0; i < ncav; ++i) {
      const double d = (surface_points.col(i) - atom_positions.col(a)).norm();
      B(i, a) = (d > 1e-6) ? 1.0 / d : 0.0;
    }
  }
  return B;
}

} // namespace

CpcmXSolvationModel::CpcmXSolvationModel(CpcmXOptions opts)
    : m_opts(std::move(opts)) {}

void CpcmXSolvationModel::initialize(const Mat3N &positions_bohr,
                                     const IVec &atomic_numbers) {
  const Eigen::Index natom = atomic_numbers.size();

  m_epsilon = (m_opts.dielectric_override > 0.0)
                  ? m_opts.dielectric_override
                  : occ::solvent::get_dielectric(m_opts.solvent);
  m_f_eps = (m_epsilon - 1.0) / (m_epsilon + m_opts.x);

  // CPCM cavity from atomic vdW radii. `solvation_radii` returns Bohr.
  Vec radii = occ::solvent::cosmo::solvation_radii(atomic_numbers);
  m_surface = occ::solvent::surface::solvent_surface(
      radii, atomic_numbers, positions_bohr, m_opts.probe_radius_angs);

  const Eigen::Index ncav =
      static_cast<Eigen::Index>(m_surface.areas.size());

  m_v_solv = Vec::Zero(natom);
  m_sigma = Vec();
  m_energy = 0.0;

  if (ncav == 0) {
    // Pathological — no cavity. Keep model functional but contribute nothing.
    occ::log::warn("CPCM-X: cavity has zero surface elements; solvation "
                   "contribution will be zero.");
    m_B.resize(0, natom);
    m_G.resize(0, natom);
    m_J_solv = Mat::Zero(natom, natom);
    return;
  }

  Mat A = build_response_matrix(m_surface.vertices, m_surface.areas);
  m_lu_A.compute(A);

  m_B = build_atom_cavity_coulomb(m_surface.vertices, positions_bohr);

  // G = −f(ε) · A^{-1} · B. Pre-solving avoids an LU back-substitution on
  // every SCC iteration: σ = G · q is then a plain GEMV.
  m_G = m_lu_A.solve(-m_f_eps * m_B);

  // J_solv = B^T · G. Symmetric (A is symmetric, so A^{-1} is too) and
  // negative-definite (f(ε) > 0 for ε > 1). Symmetrise explicitly to absorb
  // round-off so downstream code can assume exact symmetry.
  m_J_solv = m_B.transpose() * m_G;
  m_J_solv = 0.5 * (m_J_solv + m_J_solv.transpose()).eval();

  occ::log::debug("CPCM-X: solvent='{}' eps={:.4f} f(eps)={:.4f} "
                  "ncav={} natom={}",
                  m_opts.solvent, m_epsilon, m_f_eps, ncav, natom);
}

void CpcmXSolvationModel::update(const Vec &atomic_charges) {
  if (m_J_solv.rows() != atomic_charges.size()) {
    throw std::runtime_error(
        "CpcmXSolvationModel::update: atomic_charges length mismatch "
        "(model not initialized at this geometry?)");
  }
  if (m_G.rows() > 0) {
    m_sigma = m_G * atomic_charges;
  } else {
    m_sigma = Vec();
  }
  m_v_solv = m_J_solv * atomic_charges;
  m_energy = 0.5 * atomic_charges.dot(m_v_solv);
}

std::string CpcmXSolvationModel::name() const {
  return fmt::format("CPCM-X(solvent='{}', eps={:.3f})", m_opts.solvent,
                     m_epsilon);
}

} // namespace occ::xtb
