#include <Eigen/LU>
#include <fmt/core.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/scrf/reaction_field.h>
#include <occ/solvent/cosmo.h>
#include <occ/solvent/smd.h>
#include <stdexcept>

namespace occ::scrf {

namespace {

// Analytic CDS gradient at a candidate geometry. E_cds = Σ_k (σ_a + γ) a_k
// over the CDS cavity elements k on atom a: the surface tensions σ_a depend on
// the interatomic distances and, on a smooth cavity, the areas a_k on the
// geometry too.
Mat3N evaluate_cds_gradient(const Mat3N &pos_bohr, const IVec &Z,
                            const occ::solvent::SMDSolventParameters &params,
                            double probe_radius_angs,
                            double smoothing_width_bohr) {
  const Eigen::Index natom = Z.size();
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
  // Hartree per Bohr² of area per unit surface tension.
  const double scale = occ::units::BOHR_TO_ANGSTROM *
                       occ::units::BOHR_TO_ANGSTROM /
                       (1000.0 * occ::units::AU_TO_KCAL_PER_MOL);

  Vec atom_area = Vec::Zero(natom);
  for (Eigen::Index k = 0; k < cds_surf.areas.size(); ++k)
    atom_area(cds_surf.atom_index(k)) += cds_surf.areas(k);
  Mat3N grad = (scale * occ::units::BOHR_TO_ANGSTROM) *
               occ::solvent::smd::atomic_surface_tension_gradient(
                   params, Z, pos_angs, atom_area);

  if (smoothing_width_bohr > 0.0) {
    const detail::CavityGeometry cavity{pos_bohr, cds_surf.vertices,
                                        cds_surf.areas, cds_surf.atom_index,
                                        cds_r, smoothing_width_bohr};
    Vec c(cds_surf.areas.size());
    for (Eigen::Index k = 0; k < c.size(); ++k)
      c(k) = scale * (sigma_atom(cds_surf.atom_index(k)) + gamma_macro) *
             cds_surf.areas(k);
    const Vec g = detail::cavity_log_weight_gradient(cavity).transpose() * c;
    grad += Eigen::Map<const Mat3N>(g.data(), 3, natom);
  }
  return grad;
}

} // namespace

Options Options::conductor(double probe_radius_angs, int angular_points) {
  Options o;
  o.backend = Options::Backend::CPCM;
  o.radii = Options::Radii::CosmoVdW;
  o.dielectric_override = 1.0e10;
  o.f_eps_x = 0.0;
  o.probe_radius_angs = probe_radius_angs;
  o.smoothing_width_bohr = 0.0;
  o.angular_points = angular_points;
  o.include_cds = false;
  return o;
}

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
      /*axis_aligned=*/false, m_opts.smoothing_width_bohr,
      m_opts.angular_points);

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
  m_atomic_dipoles = Mat3N();
  m_atomic_quadrupoles = Mat();
  m_sigma = Vec();
  m_phi = Vec();
  m_v_solv = Vec::Zero(natom);
  m_v_dipole = Mat3N();
  m_v_quad = Mat();
  m_dE_drco = Vec();
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

void ReactionFieldEngine::solve_asc(const Vec &phi_at_cavity,
                                   std::optional<double> total_solute_charge) {
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
  if (total_solute_charge) {
    const Vec y = m_es_lu.solve(Vec::Ones(ncav));
    const double denominator = y.sum();
    if (std::abs(denominator) > 1e-14) {
      const double target = -m_f_eps * (*total_solute_charge);
      const double lambda = (m_sigma.sum() - target) / denominator;
      m_sigma -= lambda * y;
    }
  }
  m_e_es = 0.5 * m_sigma.dot(m_phi);
  // Atom-resolved state stays stale — caller is using the Eulerian path.
  m_have_atom_charges = false;
  m_atomic_charges = Vec();
  m_atomic_dipoles = Mat3N();
  m_atomic_quadrupoles = Mat();
  m_v_solv = Vec::Zero(m_atom_positions.cols());
  m_v_dipole = Mat3N();
  m_v_quad = Mat();
  m_dE_drco = Vec();
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
  m_v_dipole = Mat3N();
  m_v_quad = Mat();
  m_dE_drco = Vec();
}

void ReactionFieldEngine::set_multipole_damping(const Vec &rco_bohr,
                                               double kdmp3, double kdmp5) {
  m_damping_rco = rco_bohr;
  m_kdmp3 = kdmp3;
  m_kdmp5 = kdmp5;
}

void ReactionFieldEngine::update_from_atom_multipoles(const Vec &atom_charges,
                                                      const Mat3N &dipoles,
                                                      const Mat &quadrupoles) {
  const Eigen::Index natom = m_atom_positions.cols();
  if (atom_charges.size() != natom || dipoles.rows() != 3 ||
      dipoles.cols() != natom || quadrupoles.rows() != 6 ||
      quadrupoles.cols() != natom) {
    throw std::runtime_error(fmt::format(
        "ReactionFieldEngine::update_from_atom_multipoles: expected q({}), "
        "dipoles(3 x {}), quadrupoles(6 x {}); got q({}), dipoles({} x {}), "
        "quadrupoles({} x {})",
        natom, natom, natom, atom_charges.size(), dipoles.rows(),
        dipoles.cols(), quadrupoles.rows(), quadrupoles.cols()));
  }
  m_atomic_charges = atom_charges;
  m_have_atom_charges = true;
  m_atomic_dipoles = dipoles;
  m_atomic_quadrupoles = quadrupoles;
  m_v_solv = Vec::Zero(natom);
  m_v_dipole = Mat3N::Zero(3, natom);
  m_v_quad = Mat::Zero(6, natom);
  m_dE_drco = Vec();

  const Eigen::Index ncav = m_es_surface.areas.size();
  if (ncav == 0) {
    m_sigma = Vec();
    m_phi = Vec();
    m_e_es = 0.0;
    return;
  }

  const bool damped = m_damping_rco.size() == natom;
  const auto damping = [&](Eigen::Index a) {
    return detail::MultipoleDamping{damped ? m_damping_rco(a) : 0.0, m_kdmp3,
                                    m_kdmp5};
  };

  m_phi = m_response.B * atom_charges;
  for (Eigen::Index i = 0; i < ncav; ++i) {
    double phi = 0.0;
    for (Eigen::Index a = 0; a < natom; ++a) {
      const auto k = detail::multipole_kernel(
          m_es_surface.vertices.col(i) - m_atom_positions.col(a), damping(a));
      phi += k.t1.dot(dipoles.col(a));
      for (int p = 0; p < 6; ++p)
        phi += k.t2[p] * quadrupoles(p, a);
    }
    m_phi(i) += phi;
  }

  m_sigma = m_es_lu.solve(-m_f_eps * m_phi);

  // E_es = ½ φᵀKφ with K = −f(ε)A⁻¹ symmetric, so ∂E/∂X = (∂φ/∂X)ᵀσ for every
  // moment: each conjugate is just that moment's own kernel contracted with σ.
  m_v_solv = m_response.B.transpose() * m_sigma;
  if (damped)
    m_dE_drco = Vec::Zero(natom);
  for (Eigen::Index i = 0; i < ncav; ++i) {
    const double s = m_sigma(i);
    for (Eigen::Index a = 0; a < natom; ++a) {
      const auto k = detail::multipole_kernel(
          m_es_surface.vertices.col(i) - m_atom_positions.col(a), damping(a));
      const double phi_d = k.t1.dot(dipoles.col(a));
      double phi_q = 0.0;
      for (int p = 0; p < 6; ++p)
        phi_q += k.t2[p] * quadrupoles(p, a);
      m_v_dipole.col(a) += s * k.t1;
      for (int p = 0; p < 6; ++p)
        m_v_quad(p, a) += s * k.t2[p];
      if (damped) {
        // t1 and t2 already carry f3 and f5, so ∂φ/∂rco is φ times the
        // logarithmic derivative of the factor that scaled it.
        m_dE_drco(a) += s * (phi_d * k.df3_drco / k.f3 +
                             phi_q * k.df5_drco / k.f5);
      }
    }
  }
  m_e_es = 0.5 * m_sigma.dot(m_phi);
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
        m_es_radii, m_opts.smoothing_width_bohr, m_atomic_dipoles,
        m_atomic_quadrupoles, m_damping_rco, m_kdmp3, m_kdmp5);
  }

  if (m_opts.include_cds)
    grad += cds_gradient();

  return grad;
}

Mat3N ReactionFieldEngine::es_gradient(const Vec &sigma,
                                       const Vec &atom_charges) const {
  return detail::cosmo_gradient_frozen(
      m_atom_positions, m_es_surface.vertices, m_es_surface.areas,
      m_es_surface.atom_index, atom_charges, sigma, m_f_eps, m_es_radii,
      m_opts.smoothing_width_bohr);
}

ReactionFieldEngine::EsDerivatives
ReactionFieldEngine::es_derivatives(const Vec &sigma,
                                    const Vec &atom_charges) const {
  const detail::CavityGeometry cavity{
      m_atom_positions,         m_es_surface.vertices,
      m_es_surface.areas,       m_es_surface.atom_index,
      m_es_radii,               m_opts.smoothing_width_bohr};
  const Mat dlnw = detail::cavity_log_weight_gradient(cavity);
  EsDerivatives out;
  out.dphi = detail::atom_cavity_potential_derivatives(cavity, atom_charges);
  out.dA_sigma = detail::cosmo_A_derivative_product(cavity, dlnw, sigma);
  out.hessian =
      detail::atom_cavity_potential_hessian(cavity, atom_charges, sigma) +
      (0.5 / m_f_eps) * detail::cosmo_A_second_derivative(cavity, dlnw, sigma);
  return out;
}

Mat3N ReactionFieldEngine::cds_gradient() const {
  if (!m_opts.include_cds)
    return Mat3N::Zero(3, m_atom_positions.cols());
  return evaluate_cds_gradient(m_atom_positions, m_atomic_numbers,
                               m_smd_params, m_opts.probe_radius_angs,
                               m_opts.smoothing_width_bohr);
}

Mat ReactionFieldEngine::cds_hessian() const {
  const Eigen::Index n3 = 3 * m_atom_positions.cols();
  Mat H = Mat::Zero(n3, n3);
  if (!m_opts.include_cds)
    return H;
  const double h = 1e-4; // Bohr
  for (Eigen::Index x = 0; x < n3; ++x) {
    Mat3N pos_p = m_atom_positions, pos_m = m_atom_positions;
    pos_p(x % 3, x / 3) += h;
    pos_m(x % 3, x / 3) -= h;
    const Mat3N gp =
        evaluate_cds_gradient(pos_p, m_atomic_numbers, m_smd_params,
                              m_opts.probe_radius_angs,
                              m_opts.smoothing_width_bohr);
    const Mat3N gm =
        evaluate_cds_gradient(pos_m, m_atomic_numbers, m_smd_params,
                              m_opts.probe_radius_angs,
                              m_opts.smoothing_width_bohr);
    H.col(x) = (Eigen::Map<const Vec>(gp.data(), n3) -
                Eigen::Map<const Vec>(gm.data(), n3)) /
               (2.0 * h);
  }
  return 0.5 * (H + H.transpose());
}

} // namespace occ::scrf
