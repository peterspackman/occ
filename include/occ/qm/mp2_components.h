#pragma once

namespace occ::qm {

struct MP2OrbitalSpec {
  size_t n_frozen_core = 0;
  size_t n_active_occ = 0;
  size_t n_active_virt = 0;
  size_t n_total_occ = 0;
  size_t n_total_virt = 0;
  double e_min = 0.0;
  double e_max = 0.0;
};

struct MP2OrbitalInfo {
  size_t n_frozen_core = 0;
  size_t n_active_occ = 0;
  size_t n_active_virt = 0;
  size_t n_total_occ = 0;
  size_t n_total_virt = 0;
  double e_min_used = 0.0;
  double e_max_used = 0.0;
};

struct MP2Components {
  double total_correlation = 0.0;
  double same_spin_correlation = 0.0;
  double opposite_spin_correlation = 0.0;
  MP2OrbitalInfo orbital_info;
};

} // namespace occ::qm