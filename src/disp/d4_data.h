#pragma once
#include <array>
#include <occ/core/linear_algebra.h>

namespace occ::disp::d4_data {

inline constexpr int N_FREQ = 23;
inline constexpr int MAX_REF = 7;
inline constexpr int N_ELEMENTS = 118; // Z = 1..118, stored at index Z
inline constexpr int N_SECONDARY = 17;  // index 1..17

// Per-element DFT-D4 reference data. The active number of references is `refn`;
// trailing slots in the fixed-length arrays are unused.
struct ElementRefs {
  int refn{0};
  std::array<int, MAX_REF> refsys{};        // secondary atom index (1..17)
  std::array<double, MAX_REF> refcovcn{};   // reference covalent CN
  std::array<double, MAX_REF> refq{};       // GFN2-xTB reference charge
  std::array<double, MAX_REF> refh{};       // GFN2-xTB reference Hubbard offset
  std::array<double, MAX_REF> ascale{};     // ref-α scale
  std::array<double, MAX_REF> hcount{};     // # of secondary atoms to subtract
  // Ref polarizability α(iω) on the 23-point Casimir-Polder grid.
  std::array<std::array<double, N_FREQ>, MAX_REF> alphaiw{};
};

struct ReferenceData {
  // Casimir-Polder integration grid (length N_FREQ).
  std::array<double, N_FREQ> casimir_polder_weights{};
  // Per-element scalars; indexed by atomic number Z (1..N_ELEMENTS).
  // Index 0 is unused/zero.
  std::array<double, N_ELEMENTS + 1> zeff{};
  std::array<double, N_ELEMENTS + 1> gam{};
  std::array<double, N_ELEMENTS + 1> sqrt_zr4r2{};
  // Secondary-atom records (index 1..N_SECONDARY).
  std::array<double, N_SECONDARY + 1> secq{};
  std::array<double, N_SECONDARY + 1> sscale{};
  std::array<std::array<double, N_FREQ>, N_SECONDARY + 1> secaiw{};
  // Per-element references.
  std::array<ElementRefs, N_ELEMENTS + 1> elements{};
};

// Get the cached reference data, lazily loading it from share/dftd4/refdata.json
// on first call. Throws std::runtime_error if the file cannot be located/parsed.
const ReferenceData &reference_data();

} // namespace occ::disp::d4_data
