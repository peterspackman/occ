#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/solvent/cosmo.h>
#include <occ/solvent/parameters.h>
#include <occ/xtb/cosmo_response.h>
#include <occ/xtb/cpcmx.h>
#include <stdexcept>

namespace occ::xtb {

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
    occ::log::warn("CPCM-X: cavity has zero surface elements; solvation "
                   "contribution will be zero.");
  }

  auto resp = cosmo::build(positions_bohr, m_surface, m_epsilon, m_opts.x);
  m_G = std::move(resp.G);
  m_J_solv = std::move(resp.J_solv);

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
