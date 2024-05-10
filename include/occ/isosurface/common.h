#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/units.h>
#include <occ/core/interpolator.h>
#include <vector>

namespace occ::isosurface {

namespace impl {

template<class Func>
void remap_vertices(const Func &f, const std::vector<float> &v, std::vector<float> &dest) {
    dest.resize(v.size());
    for(int i = 0; i < v.size(); i += 3) {
       dest[i] = occ::units::BOHR_TO_ANGSTROM * v[i];
       dest[i + 1] = occ::units::BOHR_TO_ANGSTROM * v[i + 1];
       dest[i + 2] = occ::units::BOHR_TO_ANGSTROM * v[i + 2];
    }
}

}

struct AxisAlignedBoundingBox {
    Eigen::Vector3f lower;
    Eigen::Vector3f upper;

    inline bool inside(const Eigen::Vector3f &point) const {
        return (lower.array() <= point.array()).all() &&
               (upper.array() >= point.array()).all();
    }

};

using LinearInterpolatorFloat =
    occ::core::Interpolator1D<float, occ::core::DomainMapping::Linear>;

struct AtomInterpolator {
    LinearInterpolatorFloat interpolator;
    Eigen::Matrix<float, 3, Eigen::Dynamic> positions;
    float threshold{144.0};
    int interior{0};
};

struct InterpolatorParams {
    int num_points{8192};
    float domain_lower{0.04};
    float domain_upper{144.0};
};


}


