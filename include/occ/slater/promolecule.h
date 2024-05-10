#pragma once
#include <occ/isosurface/common.h>
#include <ankerl/unordered_dense.h>
#include <occ/core/molecule.h>
#include <occ/core/interpolator.h>

namespace occ::slater {


namespace impl {
    using Interpolator = occ::core::Interpolator1D<float, occ::core::DomainMapping::Linear>;
}

struct AtomInterpolator {
    impl::Interpolator interpolator;
    FMat3N positions;
    float threshold{144.0};
};

struct InterpolatorParams {
    int num_points{8192};
    float domain_lower{0.04};
    float domain_upper{144.0};
};


class PromoleculeDensity {
public:
    PromoleculeDensity(Eigen::Ref<const IVec>, Eigen::Ref<const FMat3N>,
		       const InterpolatorParams& = {});
    PromoleculeDensity(const occ::core::Molecule &mol, const InterpolatorParams &params = {});


    OCC_ALWAYS_INLINE float operator()(const FVec3 &pos) const {
        float result{0.0};
        for (const auto &[interp, interp_positions, threshold] :
             m_atom_interpolators) {
            for (int i = 0; i < interp_positions.cols(); i++) {
                float r = (interp_positions.col(i) - pos).squaredNorm();
                if (r > threshold)
                    continue;
                float rho = interp(r);
                result += rho;
            }
        }
        return result;
    }

    OCC_ALWAYS_INLINE FVec3 gradient(const FVec3 &pos) const {
        FVec3 grad = FVec3::Zero();
        for (const auto &[interp, interp_positions, threshold] :
             m_atom_interpolators) {
            for (int i = 0; i < interp_positions.cols(); i++) {
                FVec3 v = interp_positions.col(i) - pos;
                float r = v.squaredNorm();
                if (r > threshold)
                    continue;
                float grad_rho = interp.gradient(r);
                grad.array() += 2 * v.array() * grad_rho;
            }
        }
        return grad;
    }

    OCC_ALWAYS_INLINE std::pair<float, FVec3> density_and_gradient(const FVec3 &pos) const {
        double result{0.0};
        FVec3 grad = FVec3::Zero();
        for (const auto &[interp, interp_positions, threshold] :
             m_atom_interpolators) {
            for (int i = 0; i < interp_positions.cols(); i++) {
                FVec3 v = interp_positions.col(i) - pos;
                float r = v.squaredNorm();
                if (r > threshold)
                    continue;
                float rho = interp(r);
                float grad_rho = interp.gradient(r);
                result += rho;
                grad.array() += 2 * v.array() * grad_rho;
            }
        }
        return {result, grad};
    }

    float maximum_distance_heuristic(float value, float buffer = 1.0f) const;

private:
    void initialize_interpolators(Eigen::Ref<const IVec> numbers, Eigen::Ref<const FMat3N> positions);
    InterpolatorParams m_interpolator_params;
    std::vector<AtomInterpolator> m_atom_interpolators;
};

}
