#include <algorithm>
#include <occ/core/log.h>
#include <occ/core/timings.h>
#include <occ/dft/grid_utils.h>
#include <occ/dft/molecular_grid.h>

namespace occ::dft {

MolecularGrid::MolecularGrid(const AOBasis &basis, const GridSettings &settings,
                             RadialGridMethod method)
    : m_settings(settings), m_radial_method(method) {

  occ::timing::start(occ::timing::category::grid_init);

  // Validate and adjust settings
  ensure_settings();

  // Initialize from basis
  initialize_from_basis(basis);
  initialize_default_radii();

  occ::timing::stop(occ::timing::category::grid_init);
}

MolecularGrid::MolecularGrid(const Mat3N &positions, const IVec &atomic_numbers,
                             const GridSettings &settings,
                             RadialGridMethod method)
    : m_settings(settings), m_radial_method(method) {

  occ::timing::start(occ::timing::category::grid_init);

  // Validate and adjust settings
  ensure_settings();

  // Initialize from atomic positions and numbers
  initialize_from_atoms(positions, atomic_numbers);
  initialize_default_radii();

  occ::timing::stop(occ::timing::category::grid_init);
}

size_t MolecularGrid::n_atoms() const { return m_atomic_numbers.size(); }

AtomGrid MolecularGrid::get_partitioned_atom_grid(size_t atom_idx) const {
  occ::timing::start(occ::timing::category::grid_points);

  // Find the corresponding grid for this atomic number
  const size_t atomic_number = m_atomic_numbers(atom_idx);
  AtomGrid grid;
  grid.atomic_number = -1;

  for (const auto &ugrid : m_unique_atom_grids) {
    if (ugrid.atomic_number == atomic_number) {
      grid = ugrid;
      break;
    }
  }

  if (grid.atomic_number < 0) {
    throw std::runtime_error("Unique atom grid not found for atomic number " +
                             std::to_string(atomic_number));
  }

  // Apply center position to grid points
  Vec3 center = m_positions.col(atom_idx);
  grid.points.colwise() += center;

  PartitionMethod partition{PartitionFunction::Becke,
                            m_settings.treutler_alrichs_adjustment};
  // Calculate weights using Becke partition scheme
  Mat weights = calculate_atomic_grid_weights(
      partition, grid.points, m_positions, m_atomic_radii, m_dists);

  // Apply weights to grid with normalization
  grid.weights.array() *=
      weights.col(atom_idx).array() / weights.array().rowwise().sum();

  occ::timing::stop(occ::timing::category::grid_points);
  return grid;
}

AtomGrid MolecularGrid::get_atom_grid(size_t atomic_number) const {

  if (m_use_custom_radii) {
    return generate_atom_grid(atomic_number, m_settings, m_radial_method);
    // Custom grid generation
  } else {
    // Find basis information for this atomic number
    for (size_t i = 0; i < m_atomic_numbers.size(); i++) {
      if (m_atomic_numbers(i) == atomic_number) {
        return generate_atom_grid(atomic_number, m_settings, m_radial_method,
                                  m_alpha_max(i), m_l_max(i),
                                  m_alpha_min.col(i));
      }
    }

    // If no basis information is found, use default parameters
    occ::log::warn("No basis information found for atomic number {}. "
                   "Using default parameters.",
                   atomic_number);
    return generate_atom_grid(atomic_number, m_settings, m_radial_method);
  }
}

const GridSettings &MolecularGrid::settings() const { return m_settings; }

void MolecularGrid::populate_molecular_grid_points() {
  if (m_grid_initialized) {
    return;
  }

  occ::timing::start(occ::timing::category::grid_points);
  occ::log::debug("Generating molecular grid points...");

  // Generate partitioned atom grids
  std::vector<AtomGrid> atom_grids;
  atom_grids.reserve(n_atoms());

  for (size_t i = 0; i < n_atoms(); i++) {
    atom_grids.push_back(get_partitioned_atom_grid(i));
  }

  // Initialize molecular grid points
  m_grid_points.initialize_from_atom_grids(atom_grids);
  m_grid_initialized = true;

  occ::log::debug("Generated {} total grid points", m_grid_points.num_points());
  occ::timing::stop(occ::timing::category::grid_points);
}

const MolecularGridPoints &MolecularGrid::get_molecular_grid_points() const {
  if (!m_grid_initialized) {
    // Lazily initialize grid points if not done yet
    const_cast<MolecularGrid *>(this)->populate_molecular_grid_points();
  }

  return m_grid_points;
}

void MolecularGrid::ensure_settings() {
  // Validate and adjust angular grid settings
  // Skip this check if using ORCA-style angular regions (they have their own scheme)
  if (!m_settings.has_angular_regions() &&
      m_settings.max_angular_points < m_settings.min_angular_points) {
    m_settings.max_angular_points = m_settings.min_angular_points + 1;
    occ::log::warn(
        "Invalid maximum angular grid points < minimum angular grid points "
        "- will be set equal to the minimum + 1 ({} points)",
        m_settings.max_angular_points);
  }

  // Ensure max_angular_points is a valid Lebedev grid level
  int new_maximum =
      nearest_grid_level_at_or_above(m_settings.max_angular_points);
  if (new_maximum != m_settings.max_angular_points) {
    occ::log::debug("Clamping max angular grid points to next grid "
                    "level ({} -> {})",
                    m_settings.max_angular_points, new_maximum);
    m_settings.max_angular_points = new_maximum;
  }

  // Ensure min_angular_points is a valid Lebedev grid level
  int new_minimum =
      nearest_grid_level_at_or_above(m_settings.min_angular_points);
  if (new_minimum != m_settings.min_angular_points) {
    occ::log::debug("Clamping min angular grid points to next grid "
                    "level ({} -> {})",
                    m_settings.min_angular_points, new_minimum);
    m_settings.min_angular_points = new_minimum;
  }

  occ::log::debug("DFT molecular grid settings:");
  occ::log::debug("  max_angular_points = {}", m_settings.max_angular_points);
  occ::log::debug("  min_angular_points = {}", m_settings.min_angular_points);
  occ::log::debug("  radial_points = {}", m_settings.radial_points);
  occ::log::debug("  radial_precision = {:.3g}", m_settings.radial_precision);
  occ::log::debug("  reduced_first_row_element_grid = {}",
                  m_settings.reduced_first_row_element_grid ? "true" : "false");
}

void MolecularGrid::initialize_from_basis(const AOBasis &basis) {
  size_t natom = basis.atoms().size();

  // Initialize arrays
  m_atomic_numbers.resize(natom);
  m_positions.resize(3, natom);
  m_l_max.resize(natom);
  m_alpha_max.resize(natom);
  m_alpha_min.resize(basis.l_max() + 1, natom);

  // Keep track of unique atoms
  std::vector<int> unique_atoms;
  const auto atom_map = basis.atom_to_shell();

  // Extract information from basis
  for (size_t i = 0; i < natom; i++) {
    m_atomic_numbers(i) = basis.atoms()[i].atomic_number;
    unique_atoms.push_back(basis.atoms()[i].atomic_number);

    m_positions(0, i) = basis.atoms()[i].x;
    m_positions(1, i) = basis.atoms()[i].y;
    m_positions(2, i) = basis.atoms()[i].z;

    // Find min/max exponents and max angular momentum
    std::vector<double> atom_min_alpha;
    double max_alpha = 0.0;
    int max_l = basis.l_max();

    for (const auto &shell_idx : atom_map[i]) {
      const auto &shell = basis[shell_idx];

      for (int j = atom_min_alpha.size(); j < max_l + 1; j++) {
        atom_min_alpha.push_back(std::numeric_limits<double>::max());
      }

      atom_min_alpha[shell.l] =
          std::min(shell.exponents.minCoeff(), atom_min_alpha[shell.l]);
      max_alpha = std::max(max_alpha, shell.exponents.maxCoeff());
    }

    for (int l = 0; l <= max_l; l++) {
      m_alpha_min(l, i) = atom_min_alpha[l];
    }

    m_alpha_max(i) = max_alpha;
    m_l_max(i) = max_l;
  }

  // Generate unique atom grids
  std::sort(unique_atoms.begin(), unique_atoms.end());
  unique_atoms.erase(std::unique(unique_atoms.begin(), unique_atoms.end()),
                     unique_atoms.end());

  for (const auto &atomic_number : unique_atoms) {
    // Find an atom with this atomic number to get its information
    size_t atom_idx = 0;
    for (size_t i = 0; i < natom; i++) {
      if (m_atomic_numbers(i) == atomic_number) {
        atom_idx = i;
        break;
      }
    }

    // Generate atom grid for this atomic number
    AtomGrid grid = generate_atom_grid(
        atomic_number, m_settings, m_radial_method, m_alpha_max(atom_idx),
        m_l_max(atom_idx), m_alpha_min.col(atom_idx));

    m_unique_atom_grids.push_back(grid);
  }

  // Calculate interatomic distances
  m_dists = calculate_interatomic_distances(m_positions);
}

void MolecularGrid::initialize_from_atoms(const Mat3N &positions,
                                          const IVec &atomic_numbers) {
  size_t natom = positions.cols();

  // Validate input
  if (atomic_numbers.size() != natom) {
    throw std::invalid_argument(
        "Number of positions must match number of atomic numbers");
  }

  // Initialize arrays
  m_atomic_numbers = atomic_numbers;
  m_positions = positions;

  // Generate unique atom grids
  std::vector<int> unique_atoms(atomic_numbers.data(),
                                atomic_numbers.data() + atomic_numbers.size());
  std::sort(unique_atoms.begin(), unique_atoms.end());
  unique_atoms.erase(std::unique(unique_atoms.begin(), unique_atoms.end()),
                     unique_atoms.end());

  for (const auto &atomic_number : unique_atoms) {
    // Generate atom grid for this atomic number
    AtomGrid grid =
        generate_atom_grid(atomic_number, m_settings, m_radial_method);
    m_unique_atom_grids.push_back(grid);
  }

  // Calculate interatomic distances
  m_dists = calculate_interatomic_distances(m_positions);
}

void MolecularGrid::initialize_default_radii() {
  m_atomic_radii = Vec::Ones(m_atomic_numbers.rows());
  for (int i = 0; i < m_atomic_radii.rows(); i++) {
    m_atomic_radii(i) = get_atomic_radius(m_atomic_numbers(i));
  }
}

void MolecularGrid::set_atomic_radii(const Vec &radii) {
  m_atomic_radii = radii;
}

} // namespace occ::dft
