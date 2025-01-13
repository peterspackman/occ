#pragma once
#include <occ/core/log.h>
#include <vector>

namespace occ::qm {

template <typename HighLevel, typename LowLevel> struct Oniom {
  std::vector<HighLevel> subsystems_high;
  std::vector<LowLevel> subsystems_low;
  LowLevel system;

  double compute_scf_energy() {
    double high = 0.0, low = 0.0;
    occ::log::info("Computing full system at low level");
    double complete = system.compute_scf_energy();

    for (auto &subsys : subsystems_high) {
      occ::log::info("Computing fragment at high level");
      high += subsys.compute_scf_energy();
    }

    for (auto &subsys : subsystems_low) {
      occ::log::info("Computing fragment at low level");
      low += subsys.compute_scf_energy();
    }

    return complete + high - low;
  }
};

} // namespace occ::qm
