#pragma once
#include <ankerl/unordered_dense.h>
#include <mutex>
#include <occ/3rdparty/concurrentqueue.h>
#include <occ/core/timings.h>
#include <occ/geometry/linear_hashed_octree.h>
#include <occ/geometry/marching_cubes.h>

namespace occ::geometry::mc {

namespace impl {
const size_t CUBE_REMAP[8] = {2, 3, 1, 0, 6, 7, 5, 4};
constexpr float sqrt_3 = 1.73205080757;

struct Edge {
  MIndex from, to;
  Edge(MIndex u, MIndex v) : from(u > v ? v : u), to(v > u ? v : u) {}
  bool operator==(const Edge &other) const {
    return (from == other.from) && (to == other.to);
  }
};

struct EdgeHash {
  uint64_t operator()(const Edge &x) const noexcept {
    return ankerl::unordered_dense::detail::wyhash::hash(&x, sizeof(x));
  }
};

} // namespace impl

struct LinearHashedMarchingCubes {
  using VertexMap = ankerl::unordered_dense::map<MIndex, uint64_t, MIndexHash>;
  using EdgeMap =
      ankerl::unordered_dense::map<impl::Edge, uint64_t, impl::EdgeHash>;
  size_t max_depth{6};
  size_t min_depth{2};
  float tolerance{1e-6};

  LinearHashedMarchingCubes(uint64_t depth) : max_depth(depth) {}

  template <typename S>
  void extract(const S &source, std::vector<float> &vertices,
               std::vector<uint32_t> &indices) {
    auto fn = [&vertices](const Eigen::Vector3f &vertex) {
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
    auto fn = [&vertices, &source, &normals](const Eigen::Vector3f &vertex) {
      auto normal = source.normal(vertex(0), vertex(1), vertex(2));
      vertices.push_back(vertex[0]);
      vertices.push_back(vertex[1]);
      vertices.push_back(vertex[2]);
      normals.push_back(normal[0]);
      normals.push_back(normal[1]);
      normals.push_back(normal[2]);
    };

    extract_impl(source, fn, indices);
  }

private:
  inline float diagonal(float distance) const {
    using namespace impl;
    return distance * sqrt_3;
  }

  template <typename S> LinearHashedOctree<float> build_octree(S &source) {
    LinearHashedOctree<float> octree;
    auto should_refine = [this](const MIndex &key, float distance) {
      const auto level = key.level();
      const auto size = key.size();
      return (level < min_depth) ||
             ((level < max_depth) && (fabs(distance) <= diagonal(size)));
    };

    auto construct_node = [&](MIndex key) {
      auto p = key.center();
      return source(p.x, p.y, p.z);
    };
    octree.build(should_refine, construct_node);
    return octree;
  }

  template <typename S, typename E>
  void extract_impl(const S &source, E &extract_fn,
                    std::vector<uint32_t> &indices) {

    occ::timing::start(occ::timing::category::mc_octree);
    auto octree = build_octree(source);
    occ::log::info("Octree nodes    {}", octree.nodes.size());
    occ::timing::stop(occ::timing::category::mc_octree);

    occ::timing::start(occ::timing::category::mc_primal);
    auto primal_vertices = compute_primal_vertices(octree);
    occ::log::info("Primal vertices {}", primal_vertices.size());
    occ::timing::stop(occ::timing::category::mc_primal);
    size_t base_index = 0;

    occ::timing::start(occ::timing::category::mc_surface);
    extract_surface(octree, primal_vertices, indices, base_index, extract_fn);
    occ::timing::stop(occ::timing::category::mc_surface);
  }

  VertexMap compute_primal_vertices(const LinearHashedOctree<float> &octree) {
    VertexMap primal_vertices;
    for (const auto &key : octree.leaves) {
      const auto level = key.level();
      std::array<MIndex, 8> primals = key.primals(level);
      for (uint_fast8_t i = 0; i < 8; i++) {
        auto vertex = primals[i];
        if (vertex.code != 0) {
          const auto level = vertex.level();
          auto kv = primal_vertices.find(vertex);
          if (kv != primal_vertices.end()) {
            if (level > kv->second) {
              kv->second = level;
            }
          } else {
            primal_vertices.insert(kv, {vertex, level});
          }
        }
      }
    }
    return primal_vertices;
  }

  template <typename E>
  void extract_surface(const LinearHashedOctree<float> &octree,
                       const VertexMap &primal_vertices,
                       std::vector<uint32_t> &indices, size_t &base_index,
                       E &extract_fn) {
    using namespace impl;
    EdgeMap index_map;
    std::array<MIndex, 8> duals{};
    std::array<float, 8> dual_distances{};
    for (auto &m : duals)
      m.code = 0;

    MIndex key;
    size_t level;
    const MIndex one{};
    // this can technically be parallelised, but in order
    // to achieve good performance would need substantial rewrite
    // in its logic
    for (const auto &[key, level] : primal_vertices.values()) {
      key.fill_duals(level, duals);
      for (uint_fast8_t i = 0; i < 8; i++) {
        MIndex m = duals[i];
        while (m > one) {
          auto distance = octree.nodes.find(m);
          if (distance != octree.nodes.end()) {
            duals[i] = m;
            dual_distances[i] = distance->second;
            break;
          }
          m = m.parent();
        }
      }

      march_one_cube(duals, dual_distances, index_map, indices, base_index,
                     extract_fn);
    }
  }

  template <typename E>
  void march_one_cube(const std::array<MIndex, 8> &nodes,
                      const std::array<float, 8> &dual_distances,
                      EdgeMap &index_map, std::vector<uint32_t> &indices,
                      size_t &base_index, E &extract_fn) {
    using namespace impl;
    using namespace tables;

    std::array<MIndex, 8> reordered_nodes{};
    std::array<Eigen::Vector3f, 8> corners{};
    std::array<float, 8> values;
    for (uint_fast8_t i = 0; i < 8; i++) {
      auto key = nodes[CUBE_REMAP[i]];
      auto distance = dual_distances[CUBE_REMAP[i]];

      reordered_nodes[i] = key;
      auto center = key.center();
      corners[i] = {static_cast<float>(center.x), static_cast<float>(center.y),
                    static_cast<float>(center.z)};
      values[i] = distance;
    }

    auto edge_fn = [&values, &corners, &reordered_nodes, &index_map, &indices,
                    &base_index, &extract_fn](size_t edge) {
      auto u = EDGE_CONNECTION[edge][0];
      auto v = EDGE_CONNECTION[edge][1];

      auto edge_key = Edge(reordered_nodes[u], reordered_nodes[v]);
      auto search = index_map.find(edge_key);
      if (search != index_map.end()) {
        indices.push_back(search->second);
      } else {
        auto index = base_index;
        base_index++;
        index_map.insert(search, {edge_key, index});
        indices.push_back(index);
        auto offset = get_offset(values[u], values[v]);
        auto vertex = interpolate(corners[u], corners[v], offset);
        extract_fn(vertex);
      }
    };
    march_cube(values, edge_fn);
  }
};

} // namespace occ::geometry::mc
