#include <occ/solvent/solvation_correction.h>
#include <fmt/core.h>

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
    input->der_order = 4;
    input->scaling = true;
    snprintf(input->radii_set, sizeof(input->radii_set), "%s", "bondi");
    snprintf(input->restart_name, sizeof(input->restart_name), "%s", "pcm_restart.npz");
    input->min_radius = 100.0;
    snprintf(input->solver_type, sizeof(input->solver_type), "%s", "iefpcm");
    snprintf(input->solvent, sizeof(input->solvent), "%s", "water");
    snprintf(input->equation_type, sizeof(input->equation_type), "%s", "secondkind");
    input->correction = 0.0;
    input->probe_radius = 1.0;

    snprintf(input->inside_type, sizeof(input->inside_type), "%s", "vacuum");
    input->outside_epsilon = 1.0;
    snprintf(input->outside_type, sizeof(input->outside_type), "%s", "uniformdielectric");

}

void host_writer(const char * message) { fmt::print("{}\n", message); }

#endif

}

ContinuumSolvationModel::ContinuumSolvationModel(const std::vector<libint2::Atom> &atoms) :
    m_nuclear_positions(3, atoms.size()), m_nuclear_charges(atoms.size())
{
    for(size_t i = 0; i < atoms.size(); i++)
    {
        m_nuclear_positions(0, i) = atoms[i].x;
        m_nuclear_positions(1, i) = atoms[i].y;
        m_nuclear_positions(2, i) = atoms[i].z;
        m_nuclear_charges(i) = atoms[i].atomic_number;
    }

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
    pcmsolver_print(m_pcm_context);
    int grid_size = pcmsolver_get_cavity_size(m_pcm_context);
    int irr_grid_size = pcmsolver_get_irreducible_cavity_size(m_pcm_context);
    m_surface_areas = Vec(grid_size);
    m_surface_positions = Mat3N(3, grid_size);
    pcmsolver_get_centers(m_pcm_context, m_surface_positions.data());
    pcmsolver_get_areas(m_pcm_context, m_surface_areas.data());
    m_surface_potential = Vec::Zero(m_surface_areas.rows());
    m_asc = Vec::Zero(m_surface_areas.rows());
#endif

}

void ContinuumSolvationModel::set_surface_potential(const Vec &potential)
{
    m_surface_potential = potential;
    m_asc_needs_update = true;
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
#endif
}

ContinuumSolvationModel::~ContinuumSolvationModel()
{
#ifdef USING_PCMSolver
    pcmsolver_delete(m_pcm_context);
#endif 
}
}
