#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/interaction/wavefunction_transform.h>

namespace occ::interaction::transform {

double
WavefunctionTransformer::compute_position_rmsd(const Mat3N &pos_ref,
                                               const Mat3N &pos_transformed) {
  return (pos_transformed - pos_ref).norm();
}

TransformResult
WavefunctionTransformer::calculate_transform(const qm::Wavefunction &wfn,
                                             const core::Molecule &mol,
                                             const crystal::Crystal &crystal) {

  using occ::crystal::SymmetryOperation;
  TransformResult result;

  Mat3N pos_mol = mol.positions();
  ankerl::unordered_dense::set<int> symops_tested;

  for (int i = 0; i < mol.size(); i++) {
    int sint = mol.asymmetric_unit_symop()(i);
    if (symops_tested.contains(sint))
      continue;

    symops_tested.insert(sint);
    SymmetryOperation symop(sint);
    Mat3N positions = wfn.positions() * units::BOHR_TO_ANGSTROM;

    result.rotation = crystal.unit_cell().direct() * symop.rotation() *
                      crystal.unit_cell().inverse();

    result.translation =
        (mol.centroid() - (result.rotation * positions).rowwise().mean()) /
        units::BOHR_TO_ANGSTROM;

    qm::Wavefunction tmp = wfn;
    tmp.apply_transformation(result.rotation, result.translation);

    Mat3N transformed_pos = tmp.positions() * units::BOHR_TO_ANGSTROM;
    result.rmsd = compute_position_rmsd(pos_mol, transformed_pos);

    occ::log::debug("Test transform (symop={}) RMSD = {}\n", symop.to_string(),
                    result.rmsd);

    // Debug: print positions for every transformation attempt
    occ::log::debug("Symop {} - Reference molecule positions (target):",
                    symop.to_string());
    for (int j = 0; j < pos_mol.cols(); j++) {
      occ::log::debug("  Atom {}: element={}, pos=[{:.6f}, {:.6f}, {:.6f}]", j,
                      mol.atomic_numbers()(j), pos_mol(0, j), pos_mol(1, j),
                      pos_mol(2, j));
    }
    occ::log::debug("Symop {} - Transformed wavefunction positions:",
                    symop.to_string());
    auto wfn_atomic_numbers = tmp.atomic_numbers();
    for (int j = 0; j < transformed_pos.cols(); j++) {
      occ::log::debug("  Atom {}: element={}, pos=[{:.6f}, {:.6f}, {:.6f}]", j,
                      wfn_atomic_numbers(j), transformed_pos(0, j),
                      transformed_pos(1, j), transformed_pos(2, j));
    }

    if (result.rmsd < 1e-3) {
      occ::log::debug("Symop found: RMSD = {:12.5f}", result.rmsd);
      result.wfn = tmp;
      return result;
    }
  }

  throw std::runtime_error(
      "Unable to determine symmetry operation to transform wavefunction");
}

} // namespace occ::interaction::transform
