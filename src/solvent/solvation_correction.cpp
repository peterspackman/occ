#include <occ/solvent/solvation_correction.h>
#include <occ/solvent/parameters.h>
#include <occ/solvent/surface.h>
#include <occ/core/logger.h>
#include <occ/solvent/smd.h>
#include <occ/core/units.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

namespace occ::solvent {

ContinuumSolvationModel::ContinuumSolvationModel(const std::vector<libint2::Atom> &atoms, const std::string &solvent) :
    m_nuclear_positions(3, atoms.size()), m_nuclear_charges(atoms.size()), m_solvent_name(solvent), m_cosmo(78.39)
{
    for(size_t i = 0; i < atoms.size(); i++)
    {
        m_nuclear_positions(0, i) = atoms[i].x;
        m_nuclear_positions(1, i) = atoms[i].y;
        m_nuclear_positions(2, i) = atoms[i].z;
        m_nuclear_charges(i) = atoms[i].atomic_number;
    }
    IVec nums = m_nuclear_charges.cast<int>();

    set_solvent(m_solvent_name);
    Vec coulomb_radii = occ::solvent::smd::intrinsic_coulomb_radii(nums, m_params);
    auto s = occ::solvent::surface::solvent_surface(coulomb_radii, nums, m_nuclear_positions, 0.0);
    m_surface_positions_coulomb = s.vertices;
    m_surface_areas_coulomb = s.areas;
    m_surface_atoms_coulomb = s.atom_index;

    Vec cds_radii = occ::solvent::smd::cds_radii(nums, m_params);
    auto s_cds = occ::solvent::surface::solvent_surface(cds_radii, nums, m_nuclear_positions, 0.0);
    m_surface_positions_cds = s_cds.vertices;
    m_surface_areas_cds = s_cds.areas;
    m_surface_atoms_cds = s_cds.atom_index;
    m_surface_potential = Vec::Zero(m_surface_areas_coulomb.rows());
    m_asc = Vec::Zero(m_surface_areas_coulomb.rows());
    double area_coulomb = m_surface_areas_coulomb.sum();
    double area_cds = m_surface_areas_cds.sum();
    double au2_to_ang2 = occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;
    fmt::print("\nsolvent surface\n");
    fmt::print("total surface area (coulomb) = {:10.3f} Angstroms^2, {} finite elements\n",
                area_coulomb * au2_to_ang2, m_surface_areas_coulomb.rows());
    fmt::print("total surface area (cds)     = {:10.3f} Angstroms^2, {} finite elements\n\n",
                area_cds * au2_to_ang2, m_surface_areas_cds.rows());

}

void ContinuumSolvationModel::set_surface_potential(const Vec &potential)
{
    m_surface_potential = potential;
    m_asc_needs_update = true;
}

void ContinuumSolvationModel::set_solvent(const std::string &solvent)
{
    m_solvent_name = solvent;
    if(!occ::solvent::smd_solvent_parameters.contains(m_solvent_name)) throw std::runtime_error(fmt::format("Unknown solvent '{}'", m_solvent_name));
    m_params = occ::solvent::smd_solvent_parameters[m_solvent_name];
    fmt::print("Using SMD solvent '{}'\n", m_solvent_name);
    fmt::print("Parameters:\n");
    fmt::print("Dielectric                    {: 9.4f}\n", m_params.dielectric);
    if(!m_params.is_water)
    {
        fmt::print("Surface Tension               {: 9.4f}\n", m_params.gamma);
        fmt::print("Acidity                       {: 9.4f}\n", m_params.acidity);
        fmt::print("Basicity                      {: 9.4f}\n", m_params.basicity);
        fmt::print("Aromaticity                   {: 9.4f}\n", m_params.aromaticity);
        fmt::print("Electronegative Halogenicity  {: 9.4f}\n", m_params.electronegative_halogenicity);
    }
    m_cosmo = COSMO(m_params.dielectric);
}


const Vec& ContinuumSolvationModel::apparent_surface_charge()
{
    if(m_asc_needs_update)
    {
      auto result = m_cosmo(m_surface_positions_coulomb, m_surface_areas_coulomb, m_surface_potential);
      m_asc = result.converged;
      m_asc_needs_update = false;
    }

    return m_asc;
}

double ContinuumSolvationModel::surface_polarization_energy()
{
    return 0.5 * m_surface_potential.dot(m_asc);
}


double ContinuumSolvationModel::smd_cds_energy() const
{
    Mat3N pos_angs = m_nuclear_positions * occ::units::BOHR_TO_ANGSTROM;
    IVec nums = m_nuclear_charges.cast<int>();
    Vec at = occ::solvent::smd::atomic_surface_tension(m_params, nums, pos_angs);
    Vec surface_areas_per_atom_angs = Vec::Zero(nums.rows());
    const double conversion_factor = occ::units::BOHR_TO_ANGSTROM * occ::units::BOHR_TO_ANGSTROM;
    for(int i = 0; i < m_surface_areas_cds.rows(); i++)
    {
        surface_areas_per_atom_angs(m_surface_atoms_cds(i)) += conversion_factor * m_surface_areas_cds(i);
    }

    /*
    fmt::print("Surface area per atom:\n");
    for(int i = 0; i < surface_areas_per_atom_angs.rows(); i++)
    {
        fmt::print("{:<7d} {:10.3f}\n", static_cast<int>(m_nuclear_charges(i)), surface_areas_per_atom_angs(i));
    }
    */
    double total_area = surface_areas_per_atom_angs.array().sum();
    double atomic_term = surface_areas_per_atom_angs.dot(at) / 1000 / occ::units::AU_TO_KCAL_PER_MOL;
    double molecular_term = total_area * occ::solvent::smd::molecular_surface_tension(m_params) / 1000 / occ::units::AU_TO_KCAL_PER_MOL;
    /*
    fmt::print("Coulomb cavity surface area: {:.3f} Ang**2\n", m_surface_areas_coulomb.array().sum() * conversion_factor);
    fmt::print("CDS cavity surface area: {:.3f} Ang**2 ({:.3f})\n", total_area, m_surface_areas_cds.array().sum() * conversion_factor);
    fmt::print("CDS energy: {:.4f}\n", (molecular_term + atomic_term) * occ::units::AU_TO_KCAL_PER_MOL);
    fmt::print("CDS energy (molecular): {:.4f}\n", molecular_term * occ::units::AU_TO_KCAL_PER_MOL);
    */
    return molecular_term + atomic_term;
}

}
