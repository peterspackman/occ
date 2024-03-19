#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/solvent/parameters.h>
#include <occ/core/atom.h>
#include <vector>

namespace occ::solvent::smd {

Vec atomic_surface_tension(const SMDSolventParameters &, const IVec &,
                           const Mat3N &);

Vec atomic_surface_tension(const SMDSolventParameters &, const IVec &,
                           const Mat3N &, const Vec &);
Vec intrinsic_coulomb_radii(const IVec &nums, const SMDSolventParameters &);
Vec cds_radii(const IVec &nums, const SMDSolventParameters &);

Vec intrinsic_coulomb_radii(const std::vector<core::Atom> &, const SMDSolventParameters &);
Vec cds_radii(const std::vector<core::Atom> &, const SMDSolventParameters &);

double molecular_surface_tension(const SMDSolventParameters &);

} // namespace occ::solvent::smd
