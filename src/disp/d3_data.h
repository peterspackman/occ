#pragma once
#include <array>
#include <occ/core/linear_algebra.h>
#include <vector>

namespace occ::disp::d3_data {

inline constexpr int N_ELEMENTS = 94; // D3 references go up to Z=94
inline constexpr int MAX_REF = 5;

// Per-pair reference C6 block (5 × 5).
using RefC6Block = std::array<std::array<double, MAX_REF>, MAX_REF>;

struct ReferenceData {
  // Number of references per element (length N_ELEMENTS, 1-indexed: index 0
  // unused, indices 1..N_ELEMENTS valid).
  std::array<int, N_ELEMENTS + 1> nref{};
  // Reference covalent CN per (element, reference) — [Z][iref], 1-indexed.
  std::array<std::array<double, MAX_REF>, N_ELEMENTS + 1> ref_cn{};
  // Per-pair reference C6 table. Pair index follows xtb's convention:
  //   ipair(i, j) = j*(j-1)/2 + i  for i ≤ j (1-indexed).
  // c6ab[ipair][iref_i][iref_j]. Length matches max_elem*(max_elem+1)/2.
  std::vector<RefC6Block> c6ab;

  // Helpers for pair indexing.
  static inline int pair_index(int Zi, int Zj) {
    int i = std::min(Zi, Zj), j = std::max(Zi, Zj);
    return (j * (j - 1)) / 2 + i; // 1-indexed
  }
};

const ReferenceData &reference_data();

} // namespace occ::disp::d3_data
