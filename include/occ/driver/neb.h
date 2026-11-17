#pragma once
#include <occ/core/molecule.h>
#include <occ/core/vibration.h>
#include <occ/io/occ_input.h>
#include <occ/qm/wavefunction.h>
#include <optional>
#include <vector>

namespace occ::driver {

struct NebResult {
  std::vector<core::Molecule> path; // endpoints included
  Vec energies;                     // Hartree
  int highest{0};                   // index in path of the highest image
  bool converged{false};
  // with neb_ts: the saddle point refined from the highest image
  std::optional<qm::Wavefunction> transition_state;
  core::VibrationalModes ts_modes;
};

/// Nudged elastic band from the input geometry to the product in
/// config.optimization.neb_product, with a climbing image, and with neb_ts
/// a transition state search from the highest image along the band's
/// tangent, then frequencies. Writes <name>_neb.xyz (the path, rewritten
/// every iteration) and, with neb_ts, the saddle point search outputs as
/// <name>_ts_opt.xyz, <name>_ts_freq.json and so on.
NebResult nudged_elastic_band(const io::OccInput &config);

} // namespace occ::driver
