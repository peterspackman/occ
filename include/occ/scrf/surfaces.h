#pragma once
#include <occ/core/linear_algebra.h>
#include <optional>

namespace occ::scrf {

/// Per-element solvation surface data — the unified Phase-3 shape consumed by
/// `occ::cg` energy decomposition and Phase-7D-style per-element reporting.
///
/// All quantities are in atomic units: positions in Bohr, areas in Bohr²,
/// energies in Hartree. `atom_index(i)` is the atomic index this element was
/// generated on (0-based, < num_atoms).
///
/// `energies(i)` is the per-element contribution to whatever branch this
/// surface represents:
///   • Coulomb (ES) branch:  energies(i) = ½ σ_i · φ_i   (sums to ½ q·V_solv)
///   • CDS branch:           energies(i) = (σ_atom + γ_macro)·A_i / scale
struct SolvationSurface {
  Mat3N positions;
  Vec areas;
  IVec atom_index;
  Vec energies;

  size_t size() const { return static_cast<size_t>(areas.size()); }
  double total_energy() const { return energies.sum(); }
  double total_area() const { return areas.sum(); }
};

/// Bundle of optional surfaces — `coulomb` is the electrostatic cavity,
/// `cds` is the SMD cavitation-dispersion-solvent-rearrangement cavity.
/// CPCM-X populates `coulomb` only; SMD populates both; gas-phase / "null"
/// returns no surfaces.
struct SolvationSurfaces {
  std::optional<SolvationSurface> coulomb;
  std::optional<SolvationSurface> cds;

  double total_energy() const {
    double e = 0.0;
    if (coulomb)
      e += coulomb->total_energy();
    if (cds)
      e += cds->total_energy();
    return e;
  }
};

} // namespace occ::scrf
