#pragma once
#include <cstdint>
#include <occ/core/linear_algebra.h>

namespace occ::crystal {

struct MillerIndex
{
    int h, k, l;
    inline double d(const Mat3 &lattice) const
    {
        Vec3 result = h * lattice.col(0) + k * lattice.col(1) + l * lattice.col(2);
        return result.norm();
    }
};

}
