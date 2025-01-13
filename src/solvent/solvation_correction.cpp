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

ContinuumSolvationModel::ContinuumSolvationModel(
    const std::vector<occ::core::Atom> &atoms, const std::string &solvent,
    double charge, bool scale_radii)
    : m_charge(charge), m_atomic_charges(Vec::Zero(atoms.size())),
      m_solvent_name(solvent), m_nuclear_positions(3, atoms.size()),
      m_nuclear_charges(atoms.size()), m_cosmo(78.39),
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

void ContinuumSolvationModel::update_radii() {
  IVec nums = m_nuclear_charges.cast<int>();
  if (m_scale_radii) {
    m_atomic_charges = occ::core::charges::eeq_partial_charges(
        nums, m_nuclear_positions * occ::units::BOHR_TO_ANGSTROM, m_charge);
    occ::log::warn("DRACO implementation currently assumes EEQ charges");
    occ::log::warn("Predicted EEQ charges (net = {})", m_charge);
    for (int i = 0; i < m_atomic_charges.size(); i++) {
      occ::log::warn("Atom {}: {:.5f}", i, m_atomic_charges(i));
    }
    m_coulomb_radii = occ::solvent::draco::smd_coulomb_radii(
        m_atomic_charges, nums, m_nuclear_positions, m_params);
  } else {
    m_coulomb_radii =
        occ::solvent::smd::intrinsic_coulomb_radii(nums, m_params);
  }
  m_cds_radii = occ::solvent::smd::cds_radii(nums, m_params);
}

void ContinuumSolvationModel::initialize_surfaces() {
  update_radii();
  IVec nums = m_nuclear_charges.cast<int>();

  ankerl::unordered_dense::map<int, double> element_radii;

  for (int i = 0; i < nums.rows(); i++) {
    int nuc = m_nuclear_charges(i);
    if (!element_radii.contains(nuc))
      element_radii[nuc] = m_coulomb_radii(i);
  }

  occ::log::info("Intrinsic coulomb radii ({} atom types)",
                 element_radii.size());
  for (const auto &x : element_radii) {
    auto el = occ::core::Element(x.first);
    occ::log::info("{:<7s} {: 12.6f}", el.symbol(), x.second);
  }
  auto s = occ::solvent::surface::solvent_surface(m_coulomb_radii, nums,
                                                  m_nuclear_positions, 0.0);
  m_surface_positions_coulomb = s.vertices;
  m_surface_areas_coulomb = s.areas;
  m_surface_atoms_coulomb = s.atom_index;
  element_radii.clear();
  for (int i = 0; i < nums.rows(); i++) {
    int nuc = m_nuclear_charges(i);
    if (!element_radii.contains(nuc))
      element_radii[nuc] = m_cds_radii[i];
  }

  occ::log::info("CDS radii");
  for (const auto &x : element_radii) {
    auto el = occ::core::Element(x.first);
    occ::log::info("{:<7s} {: 12.6f}", el.symbol(), x.second);
  }

  auto s_cds = occ::solvent::surface::solvent_surface(m_cds_radii, nums,
                                                      m_nuclear_positions, 0.0);
  m_surface_positions_cds = s_cds.vertices;
  m_surface_areas_cds = s_cds.areas;
  m_surface_atoms_cds = s_cds.atom_index;
  m_surface_potential = Vec::Zero(m_surface_areas_coulomb.rows());
  m_asc = Vec::Zero(m_surface_areas_coulomb.rows());
  double area_coulomb = m_surface_areas_coulomb.sum();
  double area_cds = m_surface_areas_cds.sum();
  double au2_to_ang2 =
      occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;
  occ::log::info("solvent surface:");
  occ::log::info("total surface area (coulomb) = {:10.3f} Angstroms^2, {} "
                 "finite elements",
                 area_coulomb * au2_to_ang2, m_surface_areas_coulomb.rows());
  occ::log::info("total surface area (cds)     = {:10.3f} Angstroms^2, {} "
                 "finite elements",
                 area_cds * au2_to_ang2, m_surface_areas_cds.rows());
}

void ContinuumSolvationModel::write_surface_file(const std::string &filename) {
  auto output = fmt::output_file(filename, fmt::file::WRONLY | O_TRUNC |
                                               fmt::file::CREATE);
  output.print("{}\natom_idx x y z area q asc\n",
               m_surface_areas_coulomb.rows());
  for (int i = 0; i < m_surface_areas_coulomb.rows(); i++) {
    output.print(
        "{:4d} {:12.6f} {:12.6f} {:12.6f} {:12.6f} {:12.8f} {:12.8f}\n",
        m_surface_atoms_coulomb(i), m_surface_positions_coulomb(0, i),
        m_surface_positions_coulomb(1, i), m_surface_positions_coulomb(2, i),
        m_surface_areas_coulomb(i), m_surface_potential(i), m_asc(i));
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
  m_cosmo = COSMO(m_params.dielectric);
  initialize_surfaces();
}

const Vec &ContinuumSolvationModel::apparent_surface_charge() {
  if (m_asc_needs_update) {
    auto result = m_cosmo(m_surface_positions_coulomb, m_surface_areas_coulomb,
                          m_surface_potential);
    m_asc = result.converged;
    m_asc_needs_update = false;
  }

  return m_asc;
}

double ContinuumSolvationModel::surface_polarization_energy() {
  return -0.5 * m_surface_potential.dot(m_asc);
}

double ContinuumSolvationModel::smd_cds_energy() const {
  Mat3N pos_angs = m_nuclear_positions * occ::units::BOHR_TO_ANGSTROM;
  IVec nums = m_nuclear_charges.cast<int>();
  Vec at = occ::solvent::smd::atomic_surface_tension(m_params, nums, pos_angs);
  Vec surface_areas_per_atom_angs = Vec::Zero(nums.rows());
  const double conversion_factor =
      occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;
  for (int i = 0; i < m_surface_areas_cds.rows(); i++) {
    surface_areas_per_atom_angs(m_surface_atoms_cds(i)) +=
        conversion_factor * m_surface_areas_cds(i);
  }

  occ::log::debug("Surface area per atom:");
  for (int i = 0; i < surface_areas_per_atom_angs.rows(); i++) {
    occ::log::debug("{:<7d} {:10.3f}", static_cast<int>(m_nuclear_charges(i)),
                    surface_areas_per_atom_angs(i));
  }
  double total_area = surface_areas_per_atom_angs.array().sum();
  double atomic_term = surface_areas_per_atom_angs.dot(at) / 1000 /
                       occ::units::AU_TO_KCAL_PER_MOL;
  double molecular_term =
      total_area * occ::solvent::smd::molecular_surface_tension(m_params) /
      1000 / occ::units::AU_TO_KCAL_PER_MOL;

  occ::log::info("CDS energy: {:.4f}", (molecular_term + atomic_term) *
                                           occ::units::AU_TO_KCAL_PER_MOL);
  occ::log::info("CDS energy (molecular): {:.4f}",
                 molecular_term * occ::units::AU_TO_KCAL_PER_MOL);
  return molecular_term + atomic_term;
}

Vec ContinuumSolvationModel::surface_cds_energy_elements() const {
  Vec result(m_surface_areas_cds.rows());
  Mat3N pos_angs = m_nuclear_positions * occ::units::BOHR_TO_ANGSTROM;
  IVec nums = m_nuclear_charges.cast<int>();
  Vec at = occ::solvent::smd::atomic_surface_tension(m_params, nums, pos_angs);
  Vec surface_areas_per_atom_angs = Vec::Zero(nums.rows());

  const double conversion_factor =
      occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;
  const double molecular_term =
      occ::solvent::smd::molecular_surface_tension(m_params);

  for (int i = 0; i < m_surface_areas_cds.rows(); i++) {
    result(i) =
        at(m_surface_atoms_cds(i)) * conversion_factor * m_surface_areas_cds(i);
    result(i) += molecular_term * m_surface_areas_cds(i);
  }
  result /= (1000 * occ::units::AU_TO_KCAL_PER_MOL);
  return result;
}

Vec ContinuumSolvationModel::surface_polarization_energy_elements() const {
  return -0.5 * m_asc.array() * m_surface_potential.array();
}

} // namespace occ::solvent
