#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::interaction {

struct WolfParams {
    double cutoff{16.0}; // Angstroms;
    double eta{0.2};     // 1/Angstroms
};

double wolf_coulomb_energy(double qi, const Vec3 &pi, Eigen::Ref<const Vec> qj,
                           Eigen::Ref<const Mat3N> pj,
                           const WolfParams &params = {});

} // namespace occ::interaction
