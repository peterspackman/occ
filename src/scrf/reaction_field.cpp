#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/scrf/reaction_field.h>
#include <occ/solvent/cosmo.h>
#include <occ/solvent/smd.h>
#include <stdexcept>

namespace occ::scrf {

namespace {

// CDS energy of a candidate geometry — used by initialize() to lay down the
// per-element baseline and by gradient() to FD over an explicit displacement.
double evaluate_cds_energy(const Mat3N &pos_bohr, const IVec &Z,
                           const occ::solvent::SMDSolventParameters &params,
                           double probe_radius_angs,
                           double smoothing_width_bohr) {
  Vec cds_r = occ::solvent::smd::cds_radii(Z, params);
  auto cds_surf =
      occ::solvent::surface::solvent_surface(cds_r, Z, pos_bohr,
                                             probe_radius_angs,
                                             /*axis_aligned=*/false,
                                             smoothing_width_bohr);
  Mat3N pos_angs = pos_bohr * occ::units::BOHR_TO_ANGSTROM;
  Vec sigma_atom =
      occ::solvent::smd::atomic_surface_tension(params, Z, pos_angs);
  const double gamma_macro =
      occ::solvent::smd::molecular_surface_tension(params);
  const double area_conv =
      occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;
  const double scale_to_hartree =
      1.0 / (1000.0 * occ::units::AU_TO_KCAL_PER_MOL);
  double e = 0.0;
  for (Eigen::Index i = 0; i < cds_surf.areas.size(); ++i) {
    const int a = cds_surf.atom_index(i);
    const double area_angs = area_conv * cds_surf.areas(i);
    e += (sigma_atom(a) + gamma_macro) * area_angs;
  }
  return e * scale_to_hartree;
}

} // namespace

ReactionFieldEngine::ReactionFieldEngine(Options opts) : m_opts(std::move(opts)) {}

void ReactionFieldEngine::initialize(const Mat3N &positions_bohr,
                                     const IVec &atomic_numbers) {
  const Eigen::Index natom = atomic_numbers.size();
  m_atom_positions = positions_bohr;
  m_atomic_numbers = atomic_numbers;

  // ---- Dielectric + SMD parameter lookup ---------------------------------
  if (m_opts.radii == Options::Radii::SmdIntrinsicCoulomb || m_opts.include_cds) {
    m_smd_params = occ::solvent::get_smd_parameters(m_opts.solvent);
    m_epsilon = m_smd_params.dielectric;
  } else {
    m_smd_params = occ::solvent::SMDSolventParameters{};
    m_epsilon = occ::solvent::get_dielectric(m_opts.solvent);
  }
  if (m_opts.dielectric_override > 0.0) {
    m_epsilon = m_opts.dielectric_override;
  }
  m_f_eps = (m_epsilon - 1.0) / (m_epsilon + m_opts.f_eps_x);

  // ---- ES cavity radii ---------------------------------------------------
  switch (m_opts.radii) {
  case Options::Radii::CosmoVdW:
    m_es_radii = occ::solvent::cosmo::solvation_radii(atomic_numbers);
    break;
  case Options::Radii::SmdIntrinsicCoulomb:
    m_es_radii =
        occ::solvent::smd::intrinsic_coulomb_radii(atomic_numbers, m_smd_params);
    break;
  case Options::Radii::Custom:
    if (m_opts.custom_es_radii_bohr.size() != natom) {
      throw std::runtime_error(
          "ReactionFieldEngine: Radii::Custom requires custom_es_radii_bohr "
          "of length N_atoms");
    }
    m_es_radii = m_opts.custom_es_radii_bohr;
    break;
  }

  // ---- ES cavity build + COSMO factor ------------------------------------
  m_es_surface = occ::solvent::surface::solvent_surface(
      m_es_radii, atomic_numbers, positions_bohr, m_opts.probe_radius_angs,
      /*axis_aligned=*/false, m_opts.smoothing_width_bohr);

  const Eigen::Index ncav = m_es_surface.areas.size();
  if (ncav == 0) {
    occ::log::warn("ReactionFieldEngine: ES cavity has zero surface elements; "
                   "solvation contribution will be zero.");
    m_response.B = Mat(0, natom);
    m_response.G = Mat(0, natom);
    m_response.J_solv = Mat::Zero(natom, natom);
    m_es_lu = Eigen::PartialPivLU<Mat>();
  } else {
    Mat A = detail::build_cosmo_A(m_es_surface.vertices, m_es_surface.areas);
    m_es_lu = Eigen::PartialPivLU<Mat>(A);
    m_response.B =
        detail::build_atom_cavity_coulomb(m_es_surface.vertices, positions_bohr);
    m_response.G = m_es_lu.solve(-m_f_eps * m_response.B);
    Mat J = m_response.B.transpose() * m_response.G;
    m_response.J_solv = 0.5 * (J + J.transpose()).eval();
  }

  // ---- CDS branch (SMD only) ---------------------------------------------
  if (m_opts.include_cds) {
    rebuild_cds_branch();
  } else {
    m_cds_surface = occ::solvent::surface::Surface{};
    m_cds_energy_elements = Vec();
    m_e_cds = 0.0;
  }

  // ---- Reset per-update state --------------------------------------------
  m_have_atom_charges = false;
  m_atomic_charges = Vec();
  m_sigma = Vec();
  m_phi = Vec();
  m_v_solv = Vec::Zero(natom);
  m_e_es = 0.0;

  occ::log::debug("ReactionFieldEngine: solvent='{}' eps={:.4f} f(eps)={:.4f} "
                  "ncav_es={} ncav_cds={} natom={} cds={} E_cds={:+.6f} Ha",
                  m_opts.solvent, m_epsilon, m_f_eps, ncav,
                  m_cds_surface.areas.size(), natom, m_opts.include_cds, m_e_cds);
}

void ReactionFieldEngine::rebuild_cds_branch() {
  Vec cds_r = occ::solvent::smd::cds_radii(m_atomic_numbers, m_smd_params);
  m_cds_surface = occ::solvent::surface::solvent_surface(
      cds_r, m_atomic_numbers, m_atom_positions, m_opts.probe_radius_angs,
      /*axis_aligned=*/false, m_opts.smoothing_width_bohr);

  Mat3N pos_angs = m_atom_positions * occ::units::BOHR_TO_ANGSTROM;
  Vec sigma_atom = occ::solvent::smd::atomic_surface_tension(
      m_smd_params, m_atomic_numbers, pos_angs);
  const double gamma_macro =
      occ::solvent::smd::molecular_surface_tension(m_smd_params);
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
}

void ReactionFieldEngine::solve_asc(const Vec &phi_at_cavity) {
  const Eigen::Index ncav = m_es_surface.areas.size();
  if (phi_at_cavity.size() != ncav) {
    throw std::runtime_error(
        fmt::format("ReactionFieldEngine::solve_asc: phi length {} != ncav {}",
                    phi_at_cavity.size(), ncav));
  }
  if (ncav == 0) {
    m_sigma = Vec();
    m_phi = Vec();
    m_e_es = 0.0;
    return;
  }
  m_phi = phi_at_cavity;
  m_sigma = m_es_lu.solve(-m_f_eps * phi_at_cavity);
  m_e_es = 0.5 * m_sigma.dot(m_phi);
  // Atom-resolved state stays stale — caller is using the Eulerian path.
  m_have_atom_charges = false;
  m_atomic_charges = Vec();
  m_v_solv = Vec::Zero(m_atom_positions.cols());
}

void ReactionFieldEngine::update_from_atom_charges(const Vec &atom_charges) {
  if (atom_charges.size() != m_response.J_solv.rows()) {
    throw std::runtime_error(
        "ReactionFieldEngine::update_from_atom_charges: q length mismatch "
        "(engine not initialised at this geometry?)");
  }
  m_atomic_charges = atom_charges;
  m_have_atom_charges = true;
  if (m_response.G.rows() > 0) {
    m_sigma = m_response.G * atom_charges;
    m_phi = m_response.B * atom_charges;
  } else {
    m_sigma = Vec();
    m_phi = Vec();
  }
  m_v_solv = m_response.J_solv * atom_charges;
  m_e_es = 0.5 * atom_charges.dot(m_v_solv);
}

SolvationSurfaces ReactionFieldEngine::surfaces() const {
  SolvationSurfaces out;
  if (m_es_surface.areas.size() > 0) {
    SolvationSurface s;
    s.positions = m_es_surface.vertices;
    s.areas = m_es_surface.areas;
    s.atom_index = m_es_surface.atom_index;
    if (m_sigma.size() == s.areas.size() && m_phi.size() == s.areas.size()) {
      // Per-element ES energy ½ σ_i · φ_i (sums to ½ q · V_solv = E_es).
      s.energies = 0.5 * m_sigma.array() * m_phi.array();
    } else {
      s.energies = Vec::Zero(s.areas.size());
    }
    out.coulomb = std::move(s);
  }
  if (m_opts.include_cds && m_cds_surface.areas.size() > 0) {
    SolvationSurface s;
    s.positions = m_cds_surface.vertices;
    s.areas = m_cds_surface.areas;
    s.atom_index = m_cds_surface.atom_index;
    s.energies = m_cds_energy_elements;
    out.cds = std::move(s);
  }
  return out;
}

Mat3N ReactionFieldEngine::gradient() const {
  const Eigen::Index natom = m_atom_positions.cols();
  Mat3N grad = Mat3N::Zero(3, natom);
  if (natom == 0 || !m_have_atom_charges)
    return grad;

  // ES branch — closed-form frozen-cavity gradient.
  if (m_sigma.size() > 0 && m_atomic_charges.size() > 0) {
    grad += detail::cosmo_gradient_frozen(
        m_atom_positions, m_es_surface.vertices, m_es_surface.areas,
        m_es_surface.atom_index, m_atomic_charges, m_sigma, m_f_eps,
        m_es_radii, m_opts.smoothing_width_bohr);
  }

  // CDS branch — FD over the geometry-only CDS energy. Rebuilding the cavity
  // at each displaced geometry is necessary with smooth masking (weights, and
  // hence per-atom areas, depend on geometry). For a boolean cavity it's
  // slightly wasteful but produces the same answer.
  if (m_opts.include_cds) {
    const double h = 1e-4; // Bohr
    for (Eigen::Index a = 0; a < natom; ++a) {
      for (int k = 0; k < 3; ++k) {
        Mat3N pos_p = m_atom_positions;
        pos_p(k, a) += h;
        Mat3N pos_m = m_atom_positions;
        pos_m(k, a) -= h;
        const double e_p =
            evaluate_cds_energy(pos_p, m_atomic_numbers, m_smd_params,
                                m_opts.probe_radius_angs,
                                m_opts.smoothing_width_bohr);
        const double e_m =
            evaluate_cds_energy(pos_m, m_atomic_numbers, m_smd_params,
                                m_opts.probe_radius_angs,
                                m_opts.smoothing_width_bohr);
        grad(k, a) += (e_p - e_m) / (2.0 * h);
      }
    }
  }

  return grad;
}

} // namespace occ::scrf
