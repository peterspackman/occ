#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::io {
struct IsosurfaceMesh {
    IsosurfaceMesh() {}
    IsosurfaceMesh(size_t num_vertices, size_t num_faces)
        : vertices(3, num_vertices), faces(3, num_faces),
          normals(3, num_vertices) {}
    Eigen::Matrix3Xf vertices;
    Eigen::Matrix3Xi faces;
    Eigen::Matrix3Xf normals;
};

struct VertexProperties {
    VertexProperties() {}
    VertexProperties(int size)
        : de(size), di(size), de_idx(size), di_idx(size), de_norm(size),
          di_norm(size), de_norm_idx(size), di_norm_idx(size), dnorm(size) {}
    Eigen::VectorXf de;
    Eigen::VectorXf di;
    Eigen::VectorXi de_idx;
    Eigen::VectorXi di_idx;
    Eigen::VectorXf de_norm;
    Eigen::VectorXf di_norm;
    Eigen::VectorXi de_norm_idx;
    Eigen::VectorXi di_norm_idx;
    Eigen::VectorXf dnorm;
};


void write_obj_file(const std::string &filename, const IsosurfaceMesh &mesh,
                    const VertexProperties &properties);

}
