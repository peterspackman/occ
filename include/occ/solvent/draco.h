#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/solvent/smd_parameters.h>

namespace occ::solvent::draco {

Vec coordination_numbers(const IVec &nums, const Mat3N &pos_bohr);
Vec smd_coulomb_radii(const Vec &charges, const IVec &nums,
                      const Mat3N &pos_bohr, const SMDSolventParameters &);

} // namespace occ::solvent::draco
