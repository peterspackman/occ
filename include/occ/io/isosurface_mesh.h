#pragma once
#include <ankerl/unordered_dense.h>
#include <occ/core/linear_algebra.h>
#include <cstdint>

namespace occ::io {

struct IsosurfaceMesh {
    IsosurfaceMesh() {}
    IsosurfaceMesh(size_t num_vertices, size_t num_faces)
        : vertices(3 * num_vertices), faces(3 * num_faces),
          normals(3 * num_vertices) {}
    std::vector<float> vertices;
    std::vector<uint32_t> faces;
    std::vector<float> normals;
    std::vector<float> mean_curvature;
    std::vector<float> gaussian_curvature;
};



struct VertexProperties {
    using FloatVertexProperties = ankerl::unordered_dense::map<std::string, std::vector<float>>;
    using IntVertexProperties = ankerl::unordered_dense::map<std::string, std::vector<int>>;


    template<typename T>
    void add_property(const std::string &name, const std::vector<T> &values) {
	if constexpr(std::is_same<T, float>::value) {
	    fprops[name] = values;
	}
	else {
	    iprops[name] = values;
	}
    }

    void add_property(const std::string &name, const FVec &values) {
	fprops[name] = std::vector<float>(values.data(), values.data() + values.size());
    }

    void add_property(const std::string &name, const IVec &values) {
	iprops[name] = std::vector<int>(values.data(), values.data() + values.size());
    }


    FloatVertexProperties fprops;
    IntVertexProperties iprops;
};

}
