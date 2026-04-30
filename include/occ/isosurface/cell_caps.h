#pragma once
#include <cstdint>
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::isosurface {

// Per-vertex classification produced by add_cell_caps:
//   0 = interior (vertex is not on any cell face)
//   1 = on a cell edge or corner (vertex lies on >= 2 face planes)
//   2 = on exactly one cell face (cap vertex / boundary curve vertex)
//
// Per-face classification:
//   0 = natural void surface (from marching cubes)
//   1 = cap triangle added by add_cell_caps
struct CellCapClassification {
  IVec vertex_class; // size = number of vertices in the final mesh
  IVec face_class;   // size = number of triangles in the final mesh
};

// Append cap geometry to an open void surface mesh sampled in fractional
// coordinates over [0, 1]^3. For each of the six cell faces, runs a 2D
// marching-squares-style triangulation of the void region (where density <
// iso) on that face, appending new vertices, normals, curvatures (set to 0
// for caps), and triangle indices in place.
//
// Vertex deduplication happens within each face (so adjacent 2D cells share
// corner / edge-crossing vertices). Vertices on the cell rim are shared
// between two faces' caps via the same dedup map. Cap vertices are *not*
// deduplicated against the existing 3D MC boundary curve vertices (some
// duplicate positions remain on the cell rim where the natural surface meets
// the cap; visually invisible, mesh has a T-junction).
//
// Adjacent unit cells produce identical cap geometry on shared faces because
// the input density values on those faces are identical; the same vertex
// positions are emitted in the same triangulation order.
//
// The functor must support:
//   - operator()(FVec3 frac) -> float       (density at a fractional point)
//   - cubes_per_side() -> Eigen::Vector3i   (per-axis sample counts)
//   - basis_transform() -> FMat3            (frac-grad → cart-grad; used for
//                                            cap normals)
template <typename F>
CellCapClassification
add_cell_caps(const F &functor, float iso,
              std::vector<float> &vertices_frac,  // [3 * num_vertices]
              std::vector<float> &normals,        // [3 * num_vertices], cart
              std::vector<float> &curvatures,     // [2 * num_vertices]
              std::vector<uint32_t> &indices,
              float on_face_tol = 1.0e-4f);

} // namespace occ::isosurface

#include <occ/isosurface/cell_caps_impl.h>
