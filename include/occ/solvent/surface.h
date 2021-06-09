#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::solvent::surface
{

struct Surface
{
    Mat3N vertices;
    Vec areas;
    IVec atom_index;
};

Surface solvent_surface(const Vec &radii, const IVec &atomic_numbers, const Mat3N &positions);

IVec nearest_atom_index(const Mat3N &atom_positions, const Mat3N &element_centers);
}
