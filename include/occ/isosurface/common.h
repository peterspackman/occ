#pragma once
#include <occ/core/linear_algebra.h>
#include <occ/core/units.h>
#include <occ/core/interpolator.h>
#include <vector>

namespace occ::isosurface {

namespace impl {
struct FillParams {
    Eigen::Vector3f origin;
    float side_length{0.0};
    float separation{0.0};
    float isovalue{0.0};
};

template<class Func>
void fill_layer(Func &f, const FillParams &params, 
		float offset, Eigen::Ref<Eigen::MatrixXf> layer) {
    Mat3N pos(3, layer.size());
    for(int x = 0, idx = 0; x < layer.rows(); x++) {
	for(int y = 0; y < layer.cols(); y++) {
	    pos(0, idx) = x * params.separation + params.origin(0);
	    pos(1, idx) = y * params.separation + params.origin(1);
	    idx++;
	}
    }
    pos.row(2).setConstant(offset * params.side_length + params.origin(2));

    auto values = f(pos);

    for(int x = 0, idx = 0; x < layer.rows(); x++) {
	for(int y = 0; y < layer.cols(); y++) {
	    layer(x, y) = params.isovalue - values(idx);
	    idx++;
	}
    }
}

template<class Func>
void remap_vertices(const Func &f, const std::vector<float> &v, std::vector<float> &dest) {
    const auto &length = f.side_length();
    const auto &origin = f.origin();
    dest.resize(v.size());
    for(int i = 0; i < v.size(); i += 3) {
       dest[i] = occ::units::BOHR_TO_ANGSTROM * (v[i] * length(0) + origin(0));
       dest[i + 1] = occ::units::BOHR_TO_ANGSTROM * (v[i + 1] * length(1) + origin(1));
       dest[i + 2] = occ::units::BOHR_TO_ANGSTROM * (v[i + 2] * length(2) + origin(2));
    }
}

inline Eigen::Vector3f remap_point(float x, float y, float z, const Eigen::Vector3f &cube, const Eigen::Vector3f &o) noexcept {
    return Eigen::Vector3f{
	x * cube(0) + o(0),
	y * cube(1) + o(1),
	z * cube(2) + o(2)
    };
}

template<class Func>
void fill_normals(Func &f, const FillParams &params,
		  const std::vector<float> &vertices,
		  std::vector<float> &normals) {
    auto cube_pos = Eigen::Map<const Eigen::Matrix3Xf>(vertices.data(), 3, vertices.size() / 3);

    Mat3N pos(cube_pos.rows(), cube_pos.cols());
    for(int i = 0; i < cube_pos.cols(); i++) {
	pos(0, i) = cube_pos(0, i) * params.side_length + params.origin(0);
	pos(1, i) = cube_pos(1, i) * params.side_length + params.origin(1);
	pos(2, i) = cube_pos(2, i) * params.side_length + params.origin(2);
    }

    Mat3N grad = f(pos);
    for(int i = 0; i < grad.cols(); i++) {
	Vec3 normal = -grad.col(i).normalized();
	normals.push_back(normal(0));
	normals.push_back(normal(1));
	normals.push_back(normal(2));
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


