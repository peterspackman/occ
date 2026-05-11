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

  m_atom_positions = positions_bohr;
  m_atomic_charges = Vec();

  m_epsilon = (m_opts.dielectric_override > 0.0)
                  ? m_opts.dielectric_override
                  : occ::solvent::get_dielectric(m_opts.solvent);
  m_f_eps = (m_epsilon - 1.0) / (m_epsilon + m_opts.x);

  // CPCM cavity from atomic vdW radii. `solvation_radii` returns Bohr.
  m_atom_radii = occ::solvent::cosmo::solvation_radii(atomic_numbers);
  m_surface = occ::solvent::surface::solvent_surface(
      m_atom_radii, atomic_numbers, positions_bohr, m_opts.probe_radius_angs,
      /*axis_aligned=*/false, m_opts.smoothing_width_bohr);

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
  m_B = std::move(resp.B);
  m_G = std::move(resp.G);
  m_J_solv = std::move(resp.J_solv);
  m_phi = Vec();

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
  m_atomic_charges = atomic_charges;
  if (m_G.rows() > 0) {
    m_sigma = m_G * atomic_charges;
    m_phi = m_B * atomic_charges;
  } else {
    m_sigma = Vec();
    m_phi = Vec();
  }
  m_v_solv = m_J_solv * atomic_charges;
  m_energy = 0.5 * atomic_charges.dot(m_v_solv);
}

Mat3N CpcmXSolvationModel::gradient() const {
  if (m_atom_positions.cols() == 0 || m_sigma.size() == 0 ||
      m_atomic_charges.size() == 0) {
    return Mat3N::Zero(3, m_atom_positions.cols());
  }
  return cosmo::gradient(m_atom_positions, m_surface, m_atomic_charges, m_sigma,
                         m_f_eps, m_atom_radii, m_opts.smoothing_width_bohr);
}

std::string CpcmXSolvationModel::name() const {
  return fmt::format("CPCM-X(solvent='{}', eps={:.3f})", m_opts.solvent,
                     m_epsilon);
}

std::optional<SolvationSurfaces> CpcmXSolvationModel::surfaces() const {
  if (m_surface.areas.size() == 0)
    return std::nullopt;
  SolvationSurface s;
  s.positions = m_surface.vertices;
  s.areas = m_surface.areas;
  s.atom_index = m_surface.atom_index;
  if (m_sigma.size() == s.areas.size() && m_phi.size() == s.areas.size()) {
    // Per-element ES energy ½ σ_i · φ_i (sums to ½ q · V_solv = E_solv).
    s.energies = 0.5 * m_sigma.array() * m_phi.array();
  } else {
    s.energies = Vec::Zero(s.areas.size());
  }
  SolvationSurfaces out;
  out.coulomb = std::move(s);
  return out;
}

} // namespace occ::xtb
