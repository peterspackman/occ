#pragma once
#include <occ/crystal/crystal.h>
#include <occ/qm/wavefunction.h>

namespace occ::interaction::transform {

struct TransformResult {
  Mat3 rotation{Mat3::Identity()};
  Vec3 translation{Vec3::Zero()};
  double rmsd{0.0};
};

class WavefunctionTransformer {
public:
  static TransformResult calculate_transform(const qm::Wavefunction &wfn,
                                             const core::Molecule &mol,
                                             const crystal::Crystal &crystal);

private:
  static double compute_position_rmsd(const Mat3N &pos_ref,
                                      const Mat3N &pos_transformed);
};

} // namespace occ::interaction::transform
