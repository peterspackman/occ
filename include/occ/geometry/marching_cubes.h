#pragma once
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <ankerl/unordered_dense.h>
#include <array>
#include <cmath>
#include <occ/core/linear_algebra.h>
#include <occ/core/timings.h>
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

// Detects a scalar point-evaluation operator()(FVec3) on the functor. When
// present it is used for cheap single-point sampling during edge root finding;
// functors that only expose batch() fall back to a 1-column batch call.
template <typename T, typename = void>
struct has_point_evaluate : std::false_type {};

template <typename T>
struct has_point_evaluate<
    T, std::void_t<decltype(std::declval<const T &>()(
           std::declval<const FVec3 &>()))>> : std::true_type {};

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

// Single-point evaluation of the source field, preferring a scalar
// operator()(FVec3) when available and otherwise falling back to a 1-column
// batch() call (so it works for every marching-cubes functor).
template <typename S>
inline float eval_point(const S &source, const FVec3 &p) {
  if constexpr (has_point_evaluate<S>::value) {
    return source(p);
  } else {
    FMat3N pos(3, 1);
    pos.col(0) = p;
    FVec out(1);
    source.batch(pos, out);
    return out(0);
  }
}

// A deferred edge crossing: everything needed to place one vertex once its
// refined offset along the edge is known. Recorded during the topology march
// so that all crossings can be root-found together in one batched pass.
struct EdgeCrossing {
  FVec3 cu, cv;             // edge endpoint positions (MC-local coordinates)
  float gu, gv;             // (field - isovalue) at the endpoints
  FVec3 grad_u, grad_v;     // finite-difference gradients at the endpoints
  FMat3 hess_u, hess_v;     // finite-difference Hessians at the endpoints
};

// Batched edge root finding: refine every crossing's offset with `steps`
// synchronized safeguarded regula-falsi (Illinois) iterations, evaluating all
// candidate points for a given step in a single batch() call (falling back to
// scalar operator() for functors that only expose point evaluation). The
// initial offset equals get_offset() exactly, so step 0 reproduces classic
// linear interpolation and each extra step strictly improves on it. Vertices
// are then emitted in index order via extract_fn.
template <typename S, typename E>
void refine_and_emit(const S &source, const std::vector<EdgeCrossing> &crossings,
                     float isovalue, int steps, E &extract_fn) {
  const size_t N = crossings.size();
  if (N == 0)
    return;

  std::vector<float> off(N), a(N, 0.0f), b(N, 1.0f), ga(N), gb(N);
  for (size_t i = 0; i < N; i++) {
    ga[i] = crossings[i].gu;
    gb[i] = crossings[i].gv;
    const float denom = gb[i] - ga[i];
    off[i] = (denom == 0.0f) ? 0.5f : (-ga[i] / denom); // == get_offset()
  }

  FMat3N pts(3, N);
  FVec vals(N);
  for (int s = 0; s < steps; s++) {
    for (size_t i = 0; i < N; i++)
      pts.col(i) = interpolate(crossings[i].cu, crossings[i].cv, off[i]);

    if constexpr (has_batch_evaluate<S>::value) {
      source.batch(pts, vals);
    } else {
      for (size_t i = 0; i < N; i++) {
        FVec3 p = pts.col(i);
        vals(i) = source(p);
      }
    }

    for (size_t i = 0; i < N; i++) {
      const float gt = vals(i) - isovalue;
      if (gt == 0.0f)
        continue;
      if (std::signbit(gt) == std::signbit(ga[i])) {
        a[i] = off[i];
        ga[i] = gt;
        gb[i] *= 0.5f; // Illinois down-weight of the stale endpoint
      } else {
        b[i] = off[i];
        gb[i] = gt;
        ga[i] *= 0.5f;
      }
      const float denom = gb[i] - ga[i];
      if (denom != 0.0f)
        off[i] = (a[i] * gb[i] - b[i] * ga[i]) / denom;
    }
  }

  for (size_t i = 0; i < N; i++) {
    const auto &c = crossings[i];
    FVec3 vertex = interpolate(c.cu, c.cv, off[i]);
    FVec3 gradient = interpolate(c.grad_u, c.grad_v, off[i]);
    FMat3 hessian = interpolate(c.hess_u, c.hess_v, off[i]);
    extract_fn(vertex, gradient, hessian);
  }
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

  // Number of regula-falsi refinement steps applied to each edge crossing.
  // 0 (default) keeps the classic linear interpolation; >0 root-finds the
  // true field along the edge for more accurate vertex placement.
  int edge_refinement_steps{0};

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
    for (size_t i = 0; i < layers.size(); i++) {
      layers[i] = FMat::Zero(size_x, size_y);
    }
    set_origin_and_side_lengths(origin, lengths);
  }

  MarchingCubes(size_t x, size_t y, size_t z)
      : size_x(x), size_y(y), size_z(z) {
    for (size_t i = 0; i < layers.size(); i++) {
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

    // Map each canonical grid edge (lower node's linear index * 3 + axis) to
    // the vertex placed on it, so every shared edge yields exactly one vertex
    // and the output is watertight by construction. (The old rolling IndexCache
    // shared only a subset of edges, duplicating ~half the vertices.)
    ankerl::unordered_dense::map<uint64_t, uint32_t> edge_vertices;
    uint32_t index = 0;

    auto grid_edge_key = [&](size_t cx, size_t cy, size_t cz,
                             size_t edge) -> uint64_t {
      const auto &eu = CORNERS[EDGE_CONNECTION[edge][0]];
      const auto &ev = CORNERS[EDGE_CONNECTION[edge][1]];
      const size_t ux = cx + eu[0], uy = cy + eu[1], uz = cz + eu[2];
      const size_t vx = cx + ev[0], vy = cy + ev[1], vz = cz + ev[2];
      const size_t lx = ux < vx ? ux : vx, ly = uy < vy ? uy : vy,
                   lz = uz < vz ? uz : vz;
      const uint64_t dir = (ux != vx) ? 0 : ((uy != vy) ? 1 : 2);
      const uint64_t node =
          (static_cast<uint64_t>(lz) * size_y + ly) * size_x + lx;
      return node * 3 + dir;
    };

    // Only populated when edge_refinement_steps > 0; one record per unique
    // vertex (in index order) for the batched root-finding pass below.
    std::vector<EdgeCrossing> crossings;

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
            const uint64_t key = grid_edge_key(x, y, z, edge);
            if (auto it = edge_vertices.find(key); it != edge_vertices.end()) {
              indices.push_back(it->second);
              return;
            }
            const size_t u = EDGE_CONNECTION[edge][0];
            const size_t v = EDGE_CONNECTION[edge][1];

            edge_vertices.emplace(key, index);
            indices.push_back(index);
            index += 1;

            if (edge_refinement_steps == 0) {
              float offset = get_offset(values[u], values[v]);
              FVec3 vertex = interpolate(corners[u], corners[v], offset);
              FVec3 gradient =
                  interpolate(vertex_gradients[u], vertex_gradients[v], offset);
              FMat3 hessian =
                  interpolate(vertex_hessians[u], vertex_hessians[v], offset);
              extract_fn(vertex, gradient, hessian);
            } else {
              // Defer geometry; root-find all crossings together after the
              // march so expensive functors evaluate in batches.
              crossings.push_back(EdgeCrossing{
                  corners[u], corners[v], values[u], values[v],
                  vertex_gradients[u], vertex_gradients[v],
                  vertex_hessians[u], vertex_hessians[v]});
            }
          };

          march_cube(values, fn);
        }
      }
    }

    if (edge_refinement_steps > 0)
      refine_and_emit(source, crossings, isovalue, edge_refinement_steps,
                      extract_fn);
  }
};

} // namespace occ::geometry::mc
