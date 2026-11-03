#pragma once
#include <occ/core/atom.h>
#include <occ/solvent/parameters.h>
#include <vector>

namespace occ::solvent::smd {

Vec atomic_surface_tension(const SMDSolventParameters &, const IVec &,
                           const Mat3N &);

Vec atomic_surface_tension(const SMDSolventParameters &, const IVec &,
                           const Mat3N &, const Vec &);
/// Nuclear gradient of sum_i w_i sigma_i for the atomic surface tensions
/// sigma_i of `atomic_surface_tension` at fixed weights w; positions and the
/// gradient in Angstrom.
Mat3N atomic_surface_tension_gradient(const SMDSolventParameters &,
                                      const IVec &, const Mat3N &,
                                      const Vec &weights);

Vec intrinsic_coulomb_radii(const IVec &nums, const SMDSolventParameters &);
Vec cds_radii(const IVec &nums, const SMDSolventParameters &);

Vec intrinsic_coulomb_radii(const std::vector<core::Atom> &,
                            const SMDSolventParameters &);
Vec cds_radii(const std::vector<core::Atom> &, const SMDSolventParameters &);

double molecular_surface_tension(const SMDSolventParameters &);

} // namespace occ::solvent::smd
