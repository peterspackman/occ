#pragma once
#include <array>
#include <occ/core/linear_algebra.h>
#include <occ/core/timings.h>
#include <occ/geometry/index_cache.h>
#include <type_traits>
#include <vector>

namespace occ::geometry::mc {

namespace impl {

template <typename T, typename = void>
struct has_batch_evaluate : std::false_type {};

template <typename T>
struct has_batch_evaluate<T, std::void_t<decltype(std::declval<T>().batch(
                                 std::declval<Eigen::Ref<const FMat3N>>(),
                                 std::declval<Eigen::Ref<FVec>>()))>>
    : std::true_type {};

// Detects an optional FMat3 basis_transform() method on the functor. When
// present, the functor is sampled in a non-cartesian basis (e.g. fractional
// for crystal voids) and the returned matrix is the inverse-transpose of the
// local-to-world Jacobian, used to map gradients/Hessians to cartesian.
template <typename T, typename = void>
struct has_basis_transform : std::false_type {};

template <typename T>
struct has_basis_transform<
    T, std::void_t<decltype(std::declval<T>().basis_transform())>>
    : std::true_type {};

} // namespace impl

namespace tables {
extern const std::array<std::array<uint_fast8_t, 3>, 8> CORNERS;
extern const std::array<std::array<uint_fast8_t, 2>, 12> EDGE_CONNECTION;
extern const std::array<std::array<int_fast8_t, 16>, 256> TRIANGLE_CONNECTION;
} // namespace tables

namespace impl {

template <typename E>
void march_cube(const std::array<float, 8> &values, E &edge_func) {
  using namespace tables;
  uint32_t cube_index = 0;
  for (size_t i = 0; i < 8; i++) {
    if (values[i] <= 0.0) {
      cube_index |= (1 << i);
    }
  }
  const auto triangles = TRIANGLE_CONNECTION[cube_index];

  for (size_t i = 0; i < 5; i++) {
    if (triangles[3 * i] < 0)
      break;
    for (size_t j = 0; j < 3; j++) {
      size_t edge = triangles[3 * i + j];
      edge_func(edge);
    }
  }
}

inline constexpr float get_offset(float a, float b) {
  float delta = b - a;
  if (delta == 0.0)
    return 0.5;
  return -a / delta;
}

template <typename T> T interpolate(T a, T b, float t) {
  return a * (1.0 - t) + b * t;
}

} // namespace impl

struct MarchingCubes {
  size_t size_x, size_y, size_z;

  FVec3 origin{0.0f, 0.0f, 0.0f};
  FVec3 lengths{1.0f, 1.0f, 1.0f};
  FVec3 scale{1.0f, 1.0f, 1.0f};
  float isovalue = 0.0f;
  FMat3N layer_positions;

  bool flip_normals{false};

  std::array<FMat, 4> layers;

  inline void set_origin_and_side_lengths(const FVec3 &o, const FVec3 &l) {
    origin = o;
    lengths = l;
    scale(0) = lengths(0) / (size_x);
    scale(1) = lengths(1) / (size_y);
    scale(2) = lengths(2) / (size_z);
    populate_layer_positions();
  }

  // Sample size_x points spanning [origin, origin+lengths] inclusive of both
  // endpoints; suitable for grids that must align with cell faces (e.g. void
  // surfaces in fractional coordinates so adjacent cells stitch).
  inline void set_origin_and_lengths_inclusive(const FVec3 &o, const FVec3 &l) {
    origin = o;
    lengths = l;
    scale(0) = lengths(0) / float(size_x - 1);
    scale(1) = lengths(1) / float(size_y - 1);
    scale(2) = lengths(2) / float(size_z - 1);
    populate_layer_positions();
  }

  inline void populate_layer_positions() {
    layer_positions = FMat3N(3, size_x * size_y);
    for (size_t y = 0, idx = 0; y < size_y; y++) {
      for (size_t x = 0; x < size_x; x++, idx++) {
        layer_positions(0, idx) = x * scale(0) + origin(0);
        layer_positions(1, idx) = y * scale(1) + origin(1);
      }
    }
  }

  MarchingCubes(size_t s) : size_x(s), size_y(s), size_z(s) {
    for (int i = 0; i < layers.size(); i++) {
      layers[i] = FMat::Zero(size_x, size_y);
    }
    set_origin_and_side_lengths(origin, lengths);
  }

  MarchingCubes(size_t x, size_t y, size_t z)
      : size_x(x), size_y(y), size_z(z) {
    for (int i = 0; i < layers.size(); i++) {
      layers[i] = FMat::Zero(size_x, size_y);
    }
    set_origin_and_side_lengths(origin, lengths);
  }

  template <typename S>
  void extract(const S &source, std::vector<float> &vertices,
               std::vector<uint32_t> &indices) {
    auto fn = [&vertices](const FVec3 &vertex, const FVec3 &gradient,
                          const FMat3 &hessian) {
      vertices.push_back(vertex(0));
      vertices.push_back(vertex(1));
      vertices.push_back(vertex(2));
    };

    extract_impl(source, fn, indices);
  }

  template <typename S>
  void extract_with_normals(const S &source, std::vector<float> &vertices,
                            std::vector<uint32_t> &indices,
                            std::vector<float> &normals) {
    int sign = flip_normals ? 1 : -1;
    auto fn = [sign, &vertices, &source, &normals](const FVec3 &vertex,
                                                   const FVec3 &gradient,
                                                   const FMat3 &hessian) {
      vertices.push_back(vertex[0]);
      vertices.push_back(vertex[1]);
      vertices.push_back(vertex[2]);

      occ::timing::start(occ::timing::isosurface_normals);
      // Normalize the gradient and use it as the normal
      FVec3 normal = sign * gradient.normalized();
      normals.push_back(normal[0]);
      normals.push_back(normal[1]);
      normals.push_back(normal[2]);
      occ::timing::stop(occ::timing::isosurface_normals);
    };

    occ::timing::stop(occ::timing::isosurface_normals);
  }

  template <typename S>
  void extract_with_curvature(const S &source, std::vector<float> &vertices,
                              std::vector<uint32_t> &indices,
                              std::vector<float> &normals,
                              std::vector<float> &curvatures) {
    int sign = flip_normals ? 1 : -1;

    // For affine sampling (e.g. fractional grids on a non-orthogonal lattice),
    // the gradient/Hessian computed in MC's local basis must be transformed
    // to cartesian before normals/curvatures are derived.
    FMat3 J_inv_T = FMat3::Identity();
    if constexpr (impl::has_basis_transform<S>::value) {
      J_inv_T = source.basis_transform();
    }

    auto fn = [sign, J_inv_T, &vertices, &normals, &curvatures](
                  const FVec3 &vertex, const FVec3 &gradient,
                  const FMat3 &hessian) {
      vertices.push_back(vertex(0));
      vertices.push_back(vertex(1));
      vertices.push_back(vertex(2));

      FVec3 g = J_inv_T * gradient;
      FMat3 h = J_inv_T * hessian * J_inv_T.transpose();
      // Normalize the gradient and use it as the normal
      float l = g.norm();
      FVec3 normal = g / l;

      normals.push_back(sign * normal[0]);
      normals.push_back(sign * normal[1]);
      normals.push_back(sign * normal[2]);

      // Evaluate surface tangents u, v
      FVec3 u(-sign * normal[1], sign * normal[0], 0.0f);
      if (u.isZero()) {
        u = FVec3(sign * normal[2], 0.0f, -sign * normal[0]);
      }
      u.normalize();
      FVec3 v = normal.cross(u);

      // Construct the UV matrix
      Eigen::Matrix<float, 3, 2> UV;
      UV.col(0) = u;
      UV.col(1) = v;

      // Calculate the shape operator matrix S
      Eigen::Matrix2f shape = (UV.transpose() * h * UV) / l;

      // Calculate mean and Gaussian curvatures
      float mean_curvature = shape.trace() / 2;
      float gaussian_curvature = shape.determinant();

      // Store the curvatures
      curvatures.push_back(mean_curvature);
      curvatures.push_back(gaussian_curvature);
    };

    extract_impl(source, fn, indices);
  }

private:
  template <typename S, typename E>
  void extract_impl(const S &source, E &extract_fn,
                    std::vector<uint32_t> &indices) {
    using namespace tables;
    using namespace impl;

    const size_t size_less_one_x = size_x - 1;
    const size_t size_less_one_y = size_y - 1;
    const size_t size_less_one_z = size_z - 1;

    std::array<FVec3, 8> corners{FVec3::Zero()};

    std::array<float, 8> values{0.0f};
    std::array<FVec3, 8> vertex_gradients;
    std::array<FMat3, 8> vertex_hessians;

    IndexCache index_cache(size_x, size_y);
    uint32_t index = 0;

    for (size_t z = 0; z < size_less_one_z; z++) {
      occ::timing::start(occ::timing::isosurface_function);

      if constexpr (impl::has_batch_evaluate<S>::value) {

        if (z == 0) {
          layer_positions.row(2).setConstant(-scale(2) + origin(2));
          source.batch(layer_positions,
                       Eigen::Map<FVec>(layers[0].data(), layers[0].size()));
          layer_positions.row(2).setConstant(origin(2));
          source.batch(layer_positions,
                       Eigen::Map<FVec>(layers[1].data(), layers[1].size()));
          layer_positions.row(2).setConstant(scale(2) + origin(2));
          source.batch(layer_positions,
                       Eigen::Map<FVec>(layers[2].data(), layers[2].size()));
          layer_positions.row(2).setConstant(2 * scale(2) + origin(2));
          source.batch(layer_positions,
                       Eigen::Map<FVec>(layers[3].data(), layers[3].size()));
        } else {
          layers[0] = layers[1];
          layers[1] = layers[2];
          layers[2] = layers[3];
          layer_positions.row(2).setConstant((z + 2) * scale(2) + origin(2));
          source.batch(layer_positions,
                       Eigen::Map<FVec>(layers[3].data(), layers[3].size()));
        }
      } else {
        if (z == 0) {
          for (size_t y = 0; y < size_y; y++) {
            for (size_t x = 0; x < size_x; x++) {
              FVec3 pos;
              pos = {x * scale(0) + origin(0), y * scale(1) + origin(1),
                     origin(2) - scale(2)};
              layers[0](x, y) = source(pos);
              pos = {x * scale(0) + origin(0), y * scale(1) + origin(1),
                     origin(2)};
              layers[1](x, y) = source(pos);
              pos = {x * scale(0) + origin(0), y * scale(1) + origin(1),
                     origin(2) + scale(2)};
              layers[2](x, y) = source(pos);
              pos = {x * scale(0) + origin(0), y * scale(1) + origin(1),
                     origin(2) + 2 * scale(2)};
              layers[3](x, y) = source(pos);
            }
          }
        } else {
          layers[0] = layers[1];
          layers[1] = layers[2];
          layers[2] = layers[3];
          FVec3 pos;
          for (size_t y = 0; y < size_y; y++) {
            for (size_t x = 0; x < size_x; x++) {
              pos = {x * scale(0) + origin(0), y * scale(1) + origin(1),
                     (z + 2) * scale(2) + origin(2)};
              layers[3](x, y) = source(pos);
            }
          }
        }
      }
      occ::timing::stop(occ::timing::isosurface_function);

      for (size_t y = 0; y < size_less_one_y; y++) {
        for (size_t x = 0; x < size_less_one_x; x++) {
          const float fac_x = 2.0 / (scale(0));
          const float fac_y = 2.0 / (scale(1));
          const float fac_z = 2.0 / (scale(2));

          for (size_t i = 0; i < 8; i++) {
            const auto corner = CORNERS[i];
            const auto cx = corner[0], cy = corner[1], cz = corner[2];
            corners[i] = {(x + cx) * scale(0) + origin(0),
                          (y + cy) * scale(1) + origin(1),
                          (z + cz) * scale(2) + origin(2)};

            const int idx = (cz == 0) ? 1 : 2;
            const FMat &layer = layers[idx];
            values[i] = layer(x + cx, y + cy);

            // Calculate gradient and Hessian
            const size_t xp = std::min(x + cx + 1, size_x - 1);
            const size_t xm = (x + cx > 0) ? x + cx - 1 : 0;
            const size_t yp = std::min(y + cy + 1, size_y - 1);
            const size_t ym = (y + cy > 0) ? y + cy - 1 : 0;

            const Eigen::MatrixXf &layerp = (cz == 0) ? layers[2] : layers[3];
            const Eigen::MatrixXf &layerm = (cz == 0) ? layers[0] : layers[1];

            const float fx_plus = layer(xp, y + cy);
            const float fx_minus = layer(xm, y + cy);
            const float fy_plus = layer(x + cx, yp);
            const float fy_minus = layer(x + cx, ym);
            const float fz_plus = layerp(x + cx, y + cy);
            const float fz_minus = layerm(x + cx, y + cy);

            vertex_gradients[i] = FVec3((fx_plus - fx_minus) * fac_x,
                                        (fy_plus - fy_minus) * fac_y,
                                        (fz_plus - fz_minus) * fac_z);

            vertex_hessians[i] = FMat3::Zero();
            vertex_hessians[i](0, 0) =
                (fx_plus + fx_minus - 2.0f * values[i]) * fac_x * fac_x;
            vertex_hessians[i](1, 1) =
                (fy_plus + fy_minus - 2.0f * values[i]) * fac_y * fac_y;
            vertex_hessians[i](2, 2) =
                (fz_plus + fz_minus - 2.0f * values[i]) * fac_z * fac_z;
            vertex_hessians[i](1, 0) = vertex_hessians[i](0, 1) =
                ((layer(xp, yp) - layer(xp, ym) - layer(xm, yp) +
                  layer(xm, ym)) *
                 0.25f) *
                fac_x * fac_y;
            vertex_hessians[i](2, 0) = vertex_hessians[i](0, 2) =
                ((layerp(xp, y + cy) - layerp(xm, y + cy) - layerm(xp, y + cy) +
                  layerm(xm, y + cy)) *
                 0.25f) *
                fac_x * fac_z;
            vertex_hessians[i](2, 1) = vertex_hessians[i](1, 2) =
                ((layerp(x + cx, yp) - layerp(x + cx, ym) - layerm(x + cx, yp) +
                  layerm(x + cx, ym)) *
                 0.25f) *
                fac_y * fac_z;

            // Form an SDF based on isovalue
            // Won't affect gradients as it's a constant shift
            values[i] = values[i] - isovalue;
          }
          auto fn = [&](size_t edge) {
            const uint32_t cached_index = index_cache.get(x, y, edge);
            if (cached_index > 0) {
              indices.push_back(cached_index);
            } else {
              size_t u = EDGE_CONNECTION[edge][0];
              size_t v = EDGE_CONNECTION[edge][1];

              index_cache.put(x, y, edge, index);
              indices.push_back(index);
              index += 1;

              float offset = get_offset(values[u], values[v]);
              FVec3 vertex = interpolate(corners[u], corners[v], offset);
              FVec3 gradient =
                  interpolate(vertex_gradients[u], vertex_gradients[v], offset);
              FMat3 hessian =
                  interpolate(vertex_hessians[u], vertex_hessians[v], offset);
              extract_fn(vertex, gradient, hessian);
            }
          };

          march_cube(values, fn);
          index_cache.advance_cell();
        }
        index_cache.advance_row();
      }
      index_cache.advance_layer();
    }
  }
};

} // namespace occ::geometry::mc
