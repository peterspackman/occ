#pragma once
#include <tonto/core/linear_algebra.h>

namespace tonto::solvent::surface
{

struct Surface
{
    tonto::Mat3N vertices;
    tonto::Vec areas;
    tonto::IVec atom_index;
};

Surface solvent_surface(const tonto::Vec &radii, const tonto::IVec &atomic_numbers, const tonto::Mat3N &positions);


}
