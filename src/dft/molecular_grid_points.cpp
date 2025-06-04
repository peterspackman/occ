#include <occ/dft/molecular_grid_points.h>

namespace occ::dft {

MolecularGridPoints::MolecularGridPoints(
    const Mat3N &points, const Vec &weights,
    const std::vector<std::pair<size_t, size_t>> &atom_blocks)
    : m_points(points), m_weights(weights), m_atom_blocks(atom_blocks) {}

MolecularGridPoints::MolecularGridPoints(
    const std::vector<AtomGrid> &atom_grids) {
  initialize_from_atom_grids(atom_grids);
}

void MolecularGridPoints::initialize_from_atom_grids(
    const std::vector<AtomGrid> &atom_grids) {
  // Calculate total number of grid points
  size_t total_points = 0;
  for (const auto &grid : atom_grids) {
    total_points += grid.num_points();
  }

  // Resize containers to hold all points
  m_points.resize(3, total_points);
  m_weights.resize(total_points);
  m_atom_blocks.clear();
  m_atom_blocks.reserve(atom_grids.size());

  // Copy data from each atom grid
  size_t offset = 0;
  for (size_t i = 0; i < atom_grids.size(); i++) {
    const auto &grid = atom_grids[i];
    size_t npt = grid.num_points();

    if (npt > 0) {
      // Copy points and weights
      m_points.middleCols(offset, npt) = grid.points;
      m_weights.segment(offset, npt) = grid.weights;

      // Record the offset and size for this atom
      m_atom_blocks.push_back({offset, npt});

      // Update the offset for the next atom
      offset += npt;
    } else {
      // Empty grid for this atom
      m_atom_blocks.push_back({offset, 0});
    }
  }
}

size_t MolecularGridPoints::num_points() const { return m_weights.size(); }

size_t MolecularGridPoints::num_atoms() const { return m_atom_blocks.size(); }

const Mat3N &MolecularGridPoints::points() const { return m_points; }

const Vec &MolecularGridPoints::weights() const { return m_weights; }

const std::vector<std::pair<size_t, size_t>> &
MolecularGridPoints::atom_blocks() const {
  return m_atom_blocks;
}

Eigen::Ref<const Mat3N>
MolecularGridPoints::points_for_atom(size_t atom_idx) const {
  if (atom_idx >= m_atom_blocks.size()) {
    throw std::out_of_range("Atom index out of range");
  }

  const auto &[offset, size] = m_atom_blocks[atom_idx];
  return m_points.middleCols(offset, size);
}

Eigen::Ref<const Vec>
MolecularGridPoints::weights_for_atom(size_t atom_idx) const {
  if (atom_idx >= m_atom_blocks.size()) {
    throw std::out_of_range("Atom index out of range");
  }

  const auto &[offset, size] = m_atom_blocks[atom_idx];

  // Handle the case of an empty grid
  if (size == 0) {
    // Return an empty vector reference
    static const Vec empty_vec(0);
    return empty_vec;
  }

  return m_weights.segment(offset, size);
}

AtomGrid MolecularGridPoints::get_atom_grid(size_t atom_idx,
                                            uint_fast8_t atomic_number) const {
  if (atom_idx >= m_atom_blocks.size()) {
    throw std::out_of_range("Atom index out of range");
  }

  const auto &[offset, size] = m_atom_blocks[atom_idx];
  AtomGrid grid(size);

  if (size > 0) {
    // Copy points and weights
    grid.points = m_points.middleCols(offset, size);
    grid.weights = m_weights.segment(offset, size);
  }

  // Set the atomic number
  grid.atomic_number = atomic_number;

  return grid;
}

} // namespace occ::dft
