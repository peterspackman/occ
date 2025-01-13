#pragma once
#include <occ/core/linear_algebra.h>

namespace occ::cg {

struct SolventSurface {
  occ::Mat3N positions;
  occ::Vec energies;
  occ::Vec areas;

  [[nodiscard]] double total_energy() const;
  [[nodiscard]] double total_area() const;
  [[nodiscard]] size_t size() const;
};

struct SMDSolventSurfaces {
  SolventSurface coulomb;
  SolventSurface cds;
  occ::Vec electronic_energies;

  // Energy components
  double total_solvation_energy{0.0};
  double electronic_contribution{0.0};
  double gas_phase_contribution{0.0};
  double free_energy_correction{0.0};

  [[nodiscard]] double total_energy() const;
};

} // namespace occ::cg
