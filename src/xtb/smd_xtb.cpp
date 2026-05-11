#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/solvent/parameters.h>
#include <occ/solvent/smd.h>
#include <occ/xtb/cosmo_response.h>
#include <occ/xtb/smd_xtb.h>
#include <stdexcept>

namespace occ::xtb {

SmdSolvationModel::SmdSolvationModel(std::string solvent)
    : m_solvent(std::move(solvent)) {}

void SmdSolvationModel::initialize(const Mat3N &positions_bohr,
                                   const IVec &atomic_numbers) {
  m_params = occ::solvent::get_smd_parameters(m_solvent);
  m_epsilon = m_params.dielectric;

  // ----------------------------------------------------------------------
  // Electrostatic cavity: classical-COSMO over the SMD Coulomb surface.
  // ----------------------------------------------------------------------
  Vec es_radii =
      occ::solvent::smd::intrinsic_coulomb_radii(atomic_numbers, m_params);
  m_es_surface = occ::solvent::surface::solvent_surface(
      es_radii, atomic_numbers, positions_bohr, /*probe_radius_angs=*/0.4);

  // CPCM ideal-conductor convention (x = 0). SMD's electrostatic term has
  // traditionally been formulated as IEFPCM, but for atom-resolved xTB the
  // Klamt classical-COSMO solve is the natural drop-in — we adopt the same
  // f(ε) as the CPCM-X model so the two paths share a code path.
  auto resp =
      cosmo::build(positions_bohr, m_es_surface, m_epsilon, /*x=*/0.0);
  m_B = std::move(resp.B);
  m_G = std::move(resp.G);
  m_J_solv = std::move(resp.J_solv);
  m_phi = Vec();

  // ----------------------------------------------------------------------
  // CDS cavity: pure geometry. Atomic surface tensions live in cal/(mol·Å²);
  // areas convert Bohr² → Å²; result is divided by 1000·E_h↔kcal to land in
  // Hartree.
  // ----------------------------------------------------------------------
  Vec cds_r = occ::solvent::smd::cds_radii(atomic_numbers, m_params);
  m_cds_surface = occ::solvent::surface::solvent_surface(
      cds_r, atomic_numbers, positions_bohr, /*probe_radius_angs=*/0.4);

  Mat3N pos_angs = positions_bohr * occ::units::BOHR_TO_ANGSTROM;
  Vec sigma_atom = occ::solvent::smd::atomic_surface_tension(
      m_params, atomic_numbers, pos_angs);
  const double gamma_macro =
      occ::solvent::smd::molecular_surface_tension(m_params);
  const double area_conv =
      occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;
  const double scale_to_hartree =
      1.0 / (1000.0 * occ::units::AU_TO_KCAL_PER_MOL);

  const Eigen::Index ncds = m_cds_surface.areas.size();
  m_cds_energy_elements = Vec(ncds);
  for (Eigen::Index i = 0; i < ncds; ++i) {
    const int a = m_cds_surface.atom_index(i);
    const double area_angs = area_conv * m_cds_surface.areas(i);
    m_cds_energy_elements(i) =
        (sigma_atom(a) + gamma_macro) * area_angs * scale_to_hartree;
  }
  m_e_cds = m_cds_energy_elements.sum();

  // ----------------------------------------------------------------------
  // Reset per-update state. update(q) will refresh once SCC starts iterating.
  // ----------------------------------------------------------------------
  const Eigen::Index natom = atomic_numbers.size();
  m_v_solv = Vec::Zero(natom);
  m_sigma = Vec();
  m_e_es = 0.0;
  m_energy = m_e_cds; // until first update(), only CDS contributes

  occ::log::debug(
      "SMD-xtb: solvent='{}' eps={:.4f} ncav_es={} ncav_cds={} natom={} "
      "E_cds={:+.6f} Ha",
      m_solvent, m_epsilon, m_es_surface.areas.size(), ncds, natom, m_e_cds);
}

void SmdSolvationModel::update(const Vec &q) {
  if (m_J_solv.rows() != q.size()) {
    throw std::runtime_error(
        "SmdSolvationModel::update: atomic_charges length mismatch "
        "(model not initialised at this geometry?)");
  }
  if (m_G.rows() > 0) {
    m_sigma = m_G * q;
    m_phi = m_B * q;
  } else {
    m_sigma = Vec();
    m_phi = Vec();
  }
  m_v_solv = m_J_solv * q;
  m_e_es = 0.5 * q.dot(m_v_solv);
  m_energy = m_e_es + m_e_cds;
}

std::string SmdSolvationModel::name() const {
  return fmt::format("SMD-xtb(solvent='{}', eps={:.3f})", m_solvent,
                     m_epsilon);
}

std::optional<SolvationSurfaces> SmdSolvationModel::surfaces() const {
  SolvationSurfaces out;
  if (m_es_surface.areas.size() > 0) {
    SolvationSurface s;
    s.positions = m_es_surface.vertices;
    s.areas = m_es_surface.areas;
    s.atom_index = m_es_surface.atom_index;
    if (m_sigma.size() == s.areas.size() && m_phi.size() == s.areas.size()) {
      s.energies = 0.5 * m_sigma.array() * m_phi.array();
    } else {
      s.energies = Vec::Zero(s.areas.size());
    }
    out.coulomb = std::move(s);
  }
  if (m_cds_surface.areas.size() > 0) {
    SolvationSurface s;
    s.positions = m_cds_surface.vertices;
    s.areas = m_cds_surface.areas;
    s.atom_index = m_cds_surface.atom_index;
    s.energies = m_cds_energy_elements;
    out.cds = std::move(s);
  }
  if (!out.coulomb && !out.cds)
    return std::nullopt;
  return out;
}

} // namespace occ::xtb
