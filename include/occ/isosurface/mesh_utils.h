#pragma once
#include <ankerl/unordered_dense.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace occ::isosurface {

// Weld coincident vertices of a triangle mesh held in flat buffers (3 floats
// per position and normal, 2 per curvature, 3 uint32 per face). Marching cubes
// now shares vertices across cells (see marching_cubes.h), so its output is
// already deduplicated and watertight; this is a cheap safety pass that merges
// any exactly-coincident vertices and drops degenerate faces, e.g. for external
// meshes. The tolerance is deliberately tight so it never merges genuinely
// distinct (near-coincident sliver) vertices. Done in place; faces are remapped
// and degenerate faces dropped.
inline void weld_vertices(std::vector<float> &V, std::vector<float> &N,
                          std::vector<float> &C, std::vector<uint32_t> &F) {
  struct ArrHash {
    size_t operator()(const std::array<int64_t, 3> &a) const {
      return ankerl::unordered_dense::detail::wyhash::hash(a.data(), sizeof(a));
    }
  };

  constexpr double q = 1.0e5; // quantise to 1e-5 local units
  ankerl::unordered_dense::map<std::array<int64_t, 3>, uint32_t, ArrHash> wmap;
  const size_t nin = V.size() / 3;
  std::vector<uint32_t> remap(nin);
  std::vector<float> Vw, Nw, Cw;
  Vw.reserve(V.size());
  Nw.reserve(N.size());
  Cw.reserve(C.size());
  for (size_t i = 0; i < nin; i++) {
    std::array<int64_t, 3> key{std::llround(V[3 * i] * q),
                               std::llround(V[3 * i + 1] * q),
                               std::llround(V[3 * i + 2] * q)};
    auto it = wmap.find(key);
    if (it != wmap.end()) {
      remap[i] = it->second;
      continue;
    }
    const uint32_t ni = Vw.size() / 3;
    wmap.emplace(key, ni);
    remap[i] = ni;
    Vw.insert(Vw.end(), {V[3 * i], V[3 * i + 1], V[3 * i + 2]});
    Nw.insert(Nw.end(), {N[3 * i], N[3 * i + 1], N[3 * i + 2]});
    Cw.insert(Cw.end(), {C[2 * i], C[2 * i + 1]});
  }
  // Remap faces, dropping any that became degenerate (two of their vertices
  // welded together) -- these would otherwise be empty/zero-area triangles.
  std::vector<uint32_t> Fw;
  Fw.reserve(F.size());
  for (size_t f = 0; f < F.size() / 3; f++) {
    const uint32_t a = remap[F[3 * f]], b = remap[F[3 * f + 1]],
                   c = remap[F[3 * f + 2]];
    if (a != b && b != c && a != c)
      Fw.insert(Fw.end(), {a, b, c});
  }
  V.swap(Vw);
  N.swap(Nw);
  C.swap(Cw);
  F.swap(Fw);
}

} // namespace occ::isosurface
