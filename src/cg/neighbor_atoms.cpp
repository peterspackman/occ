#include <occ/cg/solvation_types.h>
#include <occ/core/units.h>

namespace occ::cg {

NeighborAtoms::NeighborAtoms(
    const crystal::CrystalDimers::MoleculeNeighbors &neighbors) {
  size_t num_atoms = 0;
  for (const auto &[n, unique_index] : neighbors) {
    num_atoms += n.b().size();
  }
  positions = Mat3N(3, num_atoms);
  molecule_index = IVec(num_atoms), atomic_numbers = IVec(num_atoms);
  vdw_radii = Vec(num_atoms);

  size_t current_idx = 0;
  size_t i = 0;
  for (const auto &[n, unique_index] : neighbors) {
    const auto &mol = n.b();
    size_t N = mol.size();
    molecule_index.block(current_idx, 0, N, 1).array() = i;
    atomic_numbers.block(current_idx, 0, N, 1).array() =
        mol.atomic_numbers().array();
    vdw_radii.block(current_idx, 0, N, 1).array() =
        mol.vdw_radii().array() / units::BOHR_TO_ANGSTROM;
    positions.block(0, current_idx, 3, N) =
        mol.positions() / units::BOHR_TO_ANGSTROM;
    current_idx += N;
    i++;
  }
}

} // namespace occ::cg
