#include <fmt/core.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <occ/core/atom.h>
#include <occ/core/eeq.h>
#include <occ/core/element.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/solvent/draco.h>
#include <occ/solvent/parameters.h>
#include <occ/solvent/smd.h>
#include <occ/solvent/solvation_correction.h>
#include <occ/solvent/surface.h>

namespace occ::solvent {

namespace {

occ::scrf::Options build_engine_options(const std::string &solvent) {
  occ::scrf::Options o;
  o.backend = occ::scrf::Options::Backend::CPCM;
  // Caller supplies the radii (DRACO-scaled if enabled, intrinsic Coulomb
  // otherwise). Set in initialize_surfaces() via Options::Custom.
  o.radii = occ::scrf::Options::Radii::Custom;
  o.solvent = solvent;
  o.f_eps_x = 0.0;
  o.probe_radius_angs = 0.0;       // DFT/HF SMD historically uses no probe
  o.smoothing_width_bohr = 0.0;    // boolean cavity (energy-only — gradients
                                   // go through the SCF response, not the
                                   // frozen-cavity engine path)
  o.include_cds = true;            // SMD CDS always on
  return o;
}

} // namespace

ContinuumSolvationModel::ContinuumSolvationModel(
    const std::vector<occ::core::Atom> &atoms, const std::string &solvent,
    double charge, bool scale_radii)
    : m_charge(charge), m_atomic_charges(Vec::Zero(atoms.size())),
      m_solvent_name(solvent), m_nuclear_positions(3, atoms.size()),
      m_nuclear_charges(atoms.size()), m_engine(build_engine_options(solvent)),
      m_scale_radii(scale_radii) {
  occ::log::debug("Number of atoms for continuum solvation model = {}",
                  atoms.size());
  for (size_t i = 0; i < atoms.size(); i++) {
    m_nuclear_positions(0, i) = atoms[i].x;
    m_nuclear_positions(1, i) = atoms[i].y;
    m_nuclear_positions(2, i) = atoms[i].z;
    m_nuclear_charges(i) = atoms[i].atomic_number;
  }
  set_solvent(m_solvent_name);
}

Vec ContinuumSolvationModel::compute_es_radii() {
  IVec nums = m_nuclear_charges.cast<int>();
  if (m_scale_radii) {
    m_atomic_charges = occ::core::charges::eeq_partial_charges(
        nums, m_nuclear_positions * occ::units::BOHR_TO_ANGSTROM, m_charge);
    occ::log::warn("DRACO implementation currently assumes EEQ charges");
    occ::log::warn("Predicted EEQ charges (net = {})", m_charge);
    for (int i = 0; i < m_atomic_charges.size(); i++) {
      occ::log::warn("Atom {}: {:.5f}", i, m_atomic_charges(i));
    }
    return occ::solvent::draco::smd_coulomb_radii(
        m_atomic_charges, nums, m_nuclear_positions, m_params);
  }
  return occ::solvent::smd::intrinsic_coulomb_radii(nums, m_params);
}

void ContinuumSolvationModel::initialize_surfaces() {
  IVec nums = m_nuclear_charges.cast<int>();
  Vec radii = compute_es_radii();

  // Log the Coulomb radii in a per-element table (matches legacy output).
  ankerl::unordered_dense::map<int, double> element_radii;
  for (int i = 0; i < nums.rows(); i++) {
    int nuc = m_nuclear_charges(i);
    if (!element_radii.contains(nuc))
      element_radii[nuc] = radii(i);
  }
  occ::log::info("Intrinsic coulomb radii ({} atom types)",
                 element_radii.size());
  for (const auto &x : element_radii) {
    auto el = occ::core::Element(x.first);
    occ::log::info("{:<7s} {: 12.6f}", el.symbol(), x.second);
  }

  // Re-configure the engine with the freshly-computed ES radii and (re)build
  // both cavities + factor A.
  auto opts = m_engine.options();
  opts.solvent = m_solvent_name;
  opts.custom_es_radii_bohr = radii;
  m_engine = occ::scrf::ReactionFieldEngine(opts);
  m_engine.initialize(m_nuclear_positions, nums);

  // CDS radii log (informational — engine builds its own CDS cavity).
  Vec cds_r = occ::solvent::smd::cds_radii(nums, m_params);
  element_radii.clear();
  for (int i = 0; i < nums.rows(); i++) {
    int nuc = m_nuclear_charges(i);
    if (!element_radii.contains(nuc))
      element_radii[nuc] = cds_r(i);
  }
  occ::log::info("CDS radii");
  for (const auto &x : element_radii) {
    auto el = occ::core::Element(x.first);
    occ::log::info("{:<7s} {: 12.6f}", el.symbol(), x.second);
  }

  m_surface_potential = Vec::Zero(m_engine.num_es_surface_points());
  m_asc = Vec::Zero(m_engine.num_es_surface_points());
  m_asc_needs_update = true;

  const double au2_to_ang2 =
      occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;
  occ::log::info("solvent surface:");
  occ::log::info("total surface area (coulomb) = {:10.3f} Angstroms^2, {} "
                 "finite elements",
                 m_engine.es_cavity().areas.sum() * au2_to_ang2,
                 m_engine.num_es_surface_points());
  occ::log::info("total surface area (cds)     = {:10.3f} Angstroms^2, {} "
                 "finite elements",
                 m_engine.cds_cavity().areas.sum() * au2_to_ang2,
                 m_engine.num_cds_surface_points());
}

void ContinuumSolvationModel::write_surface_file(const std::string &filename) {
  auto output = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                               fmt::file::CREATE);
  const auto &surf = m_engine.es_cavity();
  output.print("{}\natom_idx x y z area q asc\n", surf.areas.rows());
  for (int i = 0; i < surf.areas.rows(); i++) {
    output.print(
        "{:4d} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.8f} {:12.8f}\n",
        surf.atom_index(i), surf.vertices(0, i), surf.vertices(1, i),
        surf.vertices(2, i), surf.areas(i), m_surface_potential(i), m_asc(i));
  }
}

void ContinuumSolvationModel::set_surface_potential(const Vec &potential) {
  m_surface_potential = potential;
  m_asc_needs_update = true;
}

void ContinuumSolvationModel::set_solvent(const std::string &solvent) {
  m_solvent_name = solvent;
  m_params = occ::solvent::get_smd_parameters(m_solvent_name);
  occ::log::info("Using SMD solvent '{}'", m_solvent_name);
  occ::log::info("Parameters:");
  occ::log::info("Dielectric                    {: 9.4f}", m_params.dielectric);
  if (!m_params.is_water) {
    occ::log::info("Surface Tension               {: 9.4f}", m_params.gamma);
    occ::log::info("Acidity                       {: 9.4f}", m_params.acidity);
    occ::log::info("Basicity                      {: 9.4f}", m_params.basicity);
    occ::log::info("Aromaticity                   {: 9.4f}",
                   m_params.aromaticity);
    occ::log::info("Electronegative Halogenicity  {: 9.4f}",
                   m_params.electronegative_halogenicity);
  }
  initialize_surfaces();
}

const Vec &ContinuumSolvationModel::apparent_surface_charge() {
  if (m_asc_needs_update) {
    m_engine.solve_asc(m_surface_potential);
    m_asc = m_engine.surface_charges();
    m_asc_needs_update = false;
  }
  return m_asc;
}

double ContinuumSolvationModel::surface_polarization_energy() {
  return -0.5 * m_surface_potential.dot(m_asc);
}

Vec ContinuumSolvationModel::surface_polarization_energy_elements() const {
  return -0.5 * m_asc.array() * m_surface_potential.array();
}

} // namespace occ::solvent
