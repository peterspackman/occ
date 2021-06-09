#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/solvent/parameters.h>

namespace occ::solvent::smd{

Vec atomic_surface_tension(const SMDSolventParameters&, const IVec&, const Mat3N&);
double molecular_surface_tension(const SMDSolventParameters&);

}
