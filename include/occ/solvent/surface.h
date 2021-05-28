#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::solvent::surface
{

struct Surface
{
    occ::Mat3N vertices;
    occ::Vec areas;
    occ::IVec atom_index;
};

Surface solvent_surface(const occ::Vec &radii, const occ::IVec &atomic_numbers, const occ::Mat3N &positions);

}
