#include <occ/solvent/solvation_correction.h>
#include <occ/solvent/parameters.h>
#include <occ/solvent/surface.h>
#include <occ/core/logger.h>
#include <occ/solvent/smd.h>
#include <occ/core/units.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

namespace occ::solvent {

namespace impl {

#ifdef USING_PCMSolver
void initialize_pcminput(struct PCMInput *input)
{

    std::memset(reinterpret_cast<void*>(input), 0, sizeof *input);
    snprintf(input->cavity_type, sizeof(input->cavity_type), "%s", "gepol");
    input->patch_level = 2;
    input->coarsity = 0.5;
    input->area = 0.2;
    input->min_distance = 0.1;
    input->der_order = 1;
    input->scaling = true;
    snprintf(input->radii_set, sizeof(input->radii_set), "%s", "bondi");
    //snprintf(input->restart_name, sizeof(input->restart_name), "%s", "pcm_restart.npz");
    input->min_radius = 100.0;
    snprintf(input->solver_type, sizeof(input->solver_type), "%s", "iefpcm");
    snprintf(input->solvent, sizeof(input->solvent), "%s", "water");
    snprintf(input->equation_type, sizeof(input->equation_type), "%s", "secondkind");
    input->correction = 0.0;
    input->probe_radius = 1.385;

    snprintf(input->inside_type, sizeof(input->inside_type), "%s", "vacuum");
    input->outside_epsilon = 1.776;
    snprintf(input->outside_type, sizeof(input->outside_type), "%s", "uniformdielectric");

}

void host_writer(const char * message) { occ::log::debug("PCMSolver\n\n{}\n\n", message); }

#endif

}

ContinuumSolvationModel::ContinuumSolvationModel(const std::vector<libint2::Atom> &atoms, const std::string &solvent) :
    m_nuclear_positions(3, atoms.size()), m_nuclear_charges(atoms.size()), m_solvent_name(solvent)
#ifndef USING_PCMSolver
    , m_cosmo(78.39)
#endif
{
    for(size_t i = 0; i < atoms.size(); i++)
    {
        m_nuclear_positions(0, i) = atoms[i].x;
        m_nuclear_positions(1, i) = atoms[i].y;
        m_nuclear_positions(2, i) = atoms[i].z;
        m_nuclear_charges(i) = atoms[i].atomic_number;
    }
    IVec nums = m_nuclear_charges.cast<int>();

#ifdef USING_PCMSolver
    struct PCMInput input;
    impl::initialize_pcminput(&input);
    int symmetry_info[4] = {0, 0, 0, 0};
    m_pcm_context = pcmsolver_new(
        PCMSOLVER_READER_HOST,
        m_nuclear_charges.rows(),
        m_nuclear_charges.data(),
        m_nuclear_positions.data(),
        symmetry_info,
        &input,
        impl::host_writer
    );
    set_solvent(m_solvent_name);
    Vec coulomb_radii = occ::solvent::smd::intrinsic_coulomb_radii(nums, m_params);
    pcmsolver_print(m_pcm_context);
    m_surface_atoms_coulomb = occ::solvent::surface::nearest_atom_index(m_nuclear_positions, m_surface_positions_coulomb);
#else
    set_solvent(m_solvent_name);
    Vec coulomb_radii = occ::solvent::smd::intrinsic_coulomb_radii(nums, m_params);
    auto s = occ::solvent::surface::solvent_surface(coulomb_radii, nums, m_nuclear_positions, 0.0);
    m_surface_positions_coulomb = s.vertices;
    m_surface_areas_coulomb = s.areas;
    m_surface_atoms_coulomb = s.atom_index;
#endif
    Vec cds_radii = occ::solvent::smd::cds_radii(nums, m_params);
    auto s_cds = occ::solvent::surface::solvent_surface(cds_radii, nums, m_nuclear_positions, 0.0);
    m_surface_positions_cds = s_cds.vertices;
    m_surface_areas_cds = s_cds.areas;
    m_surface_atoms_cds = s_cds.atom_index;
    m_surface_potential = Vec::Zero(m_surface_areas_coulomb.rows());
    m_asc = Vec::Zero(m_surface_areas_coulomb.rows());

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
#if USING_PCMSolver
    pcmsolver_set_string_option(m_pcm_context, "solvent", solvent.c_str());
    pcmsolver_refresh(m_pcm_context);
    int grid_size = pcmsolver_get_cavity_size(m_pcm_context);
    int irr_grid_size = pcmsolver_get_irreducible_cavity_size(m_pcm_context);
    m_surface_areas_coulomb = Vec(grid_size);
    m_surface_positions_coulomb = Mat3N(3, grid_size);
    pcmsolver_get_centers(m_pcm_context, m_surface_positions_coulomb.data());
    pcmsolver_get_areas(m_pcm_context, m_surface_areas_coulomb.data());
    m_asc_needs_update = true;
#else
    m_cosmo = COSMO(m_params.dielectric);
#endif 
}


const Vec& ContinuumSolvationModel::apparent_surface_charge()
{
    if(m_asc_needs_update)
    {
#ifdef USING_PCMSolver
      int irrep = 0;
      int grid_size = static_cast<int>(m_surface_potential.rows());
      pcmsolver_set_surface_function(m_pcm_context, grid_size, m_surface_potential.data(), m_surface_potential_label);
      pcmsolver_compute_asc(m_pcm_context, m_surface_potential_label, m_asc_label, irrep);
      pcmsolver_get_surface_function(m_pcm_context, grid_size, m_asc.data(), m_asc_label);
#else
      auto result = m_cosmo(m_surface_positions_coulomb, m_surface_areas_coulomb, m_surface_potential);
      m_asc = result.converged;
#endif
      m_asc_needs_update = false;
    }

    return m_asc;
}

double ContinuumSolvationModel::surface_polarization_energy()
{
#ifdef USING_PCMSolver
    return pcmsolver_compute_polarization_energy(m_pcm_context, m_surface_potential_label, m_asc_label);
#else
    return 0.5 * m_surface_potential.dot(m_asc);
#endif
}

ContinuumSolvationModel::~ContinuumSolvationModel()
{
#ifdef USING_PCMSolver
    pcmsolver_delete(m_pcm_context);
#endif 
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

    fmt::print("Surface area per atom:\n");
    for(int i = 0; i < surface_areas_per_atom_angs.rows(); i++)
    {
        fmt::print("{:<7d} {:10.3f}\n", static_cast<int>(m_nuclear_charges(i)), surface_areas_per_atom_angs(i));
    }
    double total_area = surface_areas_per_atom_angs.array().sum();
    double atomic_term = surface_areas_per_atom_angs.dot(at) / 1000 / occ::units::AU_TO_KCAL_PER_MOL;
    double molecular_term = total_area * occ::solvent::smd::molecular_surface_tension(m_params) / 1000 / occ::units::AU_TO_KCAL_PER_MOL;
    fmt::print("Coulomb cavity surface area: {:.3f} Ang**2\n", m_surface_areas_coulomb.array().sum() * conversion_factor);
    fmt::print("CDS cavity surface area: {:.3f} Ang**2 ({:.3f})\n", total_area, m_surface_areas_cds.array().sum() * conversion_factor);
    fmt::print("CDS energy: {:.4f}\n", (molecular_term + atomic_term) * occ::units::AU_TO_KCAL_PER_MOL);
    fmt::print("CDS energy (molecular): {:.4f}\n", molecular_term * occ::units::AU_TO_KCAL_PER_MOL);
    return molecular_term + atomic_term;
}

}
