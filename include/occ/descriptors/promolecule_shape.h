#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/units.h>
#include <occ/core/interpolator.h>


namespace occ::descriptors {

struct PromoleculeDensityShape {
    using Interpolator =
	occ::core::Interpolator1D<float, occ::core::DomainMapping::Linear>;

    struct AtomInterpolator {
	Interpolator interpolator;
	FMat3N positions;
	float threshold{144.0};
    };

    struct InterpolatorParameters {
	int num_points{8192};
	float domain_lower{0.04};
	float domain_upper{144.0};
    };



};

}
