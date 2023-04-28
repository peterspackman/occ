#pragma once
#include <occ/core/dimer.h>
#include <occ/core/linear_algebra.h>

namespace occ::interaction {

double coulomb_energy(Eigen::Ref<const Vec> charges,
                      Eigen::Ref<const Mat3N> positions);

double coulomb_pair_energy(Eigen::Ref<const Vec> charges_a,
                           Eigen::Ref<const Mat3N> positions_a,
                           Eigen::Ref<const Vec> charges_b,
                           Eigen::Ref<const Mat3N> positions_b);

Vec3 coulomb_efield(Eigen::Ref<const Vec> charges,
                    Eigen::Ref<const Mat3N> positions,
                    Eigen::Ref<const Vec3> point);

std::pair<Mat3N, Mat3N> coulomb_pair_efield(
    Eigen::Ref<const Vec> charges_a, Eigen::Ref<const Mat3N> positions_a,
    Eigen::Ref<const Vec> charges_b, Eigen::Ref<const Mat3N> positions_b);

double
coulomb_interaction_energy_asym_charges(const occ::core::Dimer &,
                                        Eigen::Ref<const Vec> charges_asym);

std::pair<Mat3N, Mat3N>
coulomb_efield_asym_charges(const occ::core::Dimer &,
                            Eigen::Ref<const Vec> charges_asym);

double coulomb_self_energy_asym_charges(const occ::core::Molecule &,
                                        Eigen::Ref<const Vec> charges);

} // namespace occ::interaction
