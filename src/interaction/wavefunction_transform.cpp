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

    if (result.rmsd < 1e-3) {
      occ::log::debug("Symop found: RMSD = {:12.5f}", result.rmsd);
      return result;
    }
  }

  throw std::runtime_error(
      "Unable to determine symmetry operation to transform wavefunction");
}

} // namespace occ::interaction::transform
