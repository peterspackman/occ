#pragma once
#include <occ/solvent/cosmors.h>
#include <string>
#include <vector>

namespace occ::solvent::cosmors {

/// A solvent's segment ensemble as stored on disk, in the `.rsseg` layout: a
/// `# meta:` JSON header followed by
/// `sigma sigma_orth area atomic_number` rows.
///
/// The COSMO-SAC `.sigma` format cannot carry this. It stores a σ-histogram,
/// while the openCOSMO-RS kernel needs each segment's σ and σ⊥ together —
/// binning them separately would decorrelate the pair the misfit term
/// multiplies. The atomic number rides along for the cavity term.
struct ComponentFile {
  std::string name;
  Component component;
  /// Averaging radii the descriptors were built on, checked on load against
  /// the parameters in use.
  double r_av{0.0};
  double r_corr{0.0};
  double sigma_orth_factor{0.0};
  /// Per-segment atomic numbers, for `segment_cavity_energies`.
  IVec atomic_numbers;
  /// What the cavity was computed with. Mixing a solute and solvent built at
  /// different levels is silently wrong, so the provenance travels with the
  /// ensemble rather than being assumed.
  std::string method;
  std::string basis;
};

ComponentFile read_segments(const std::string &path);

void write_segments(const std::string &path, const std::string &name,
                       const Component &component,
                       const IVec &atomic_numbers,
                       const Parameters &params,
                       const std::string &method = {},
                       const std::string &basis = {});

/// Resolves a solvent name to its segment ensemble, read from `<name>.rsseg`
/// under the search paths. Ensembles are produced by `occ sigma`; nothing is
/// computed here.
class SegmentStore {
public:
  explicit SegmentStore(std::vector<std::string> search_paths);

  /// `$OCC_DATA_PATH/solvent/cosmors`, then the working directory.
  static SegmentStore standard();

  [[nodiscard]] bool contains(const std::string &name) const;

  /// Throws naming the paths searched and how to generate the ensemble.
  [[nodiscard]] ComponentFile get(const std::string &name) const;

  /// Names of every ensemble found, sorted.
  [[nodiscard]] std::vector<std::string> available() const;

  [[nodiscard]] const std::vector<std::string> &search_paths() const {
    return m_search_paths;
  }

private:
  std::vector<std::string> m_search_paths;
};

} // namespace occ::solvent::cosmors
