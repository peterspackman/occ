#pragma once
#include <occ/solvent/sigma_activity.h>
#include <string>

namespace occ::solvent::sigma {

/// A σ-profile as stored on disk, in the `.sigma` layout used by the
/// published COSMO-SAC databases: a `# meta:` JSON header followed by
/// `sigma psigmaA` rows, with the three H-bond classes concatenated in the
/// order NHB, OH, OT when present.
struct ProfileFile {
  std::string name;
  Component component; ///< profile (Å² per bin) and cavity volume (Å³)
  double area{0.0};    ///< cavity area from the header, Å²
};

ProfileFile read_sigma_profile(const std::string &path);

void write_sigma_profile(const std::string &path, const std::string &name,
                         const Profile &profile, const Parameters &params,
                         double area, double volume);

} // namespace occ::solvent::sigma
