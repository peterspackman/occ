#pragma once
#include <occ/solvent/cosmors.h>
#include <string>
#include <vector>

namespace occ::solvent::cosmors {

/// A solvent's segment ensemble as stored on disk, in the `.rsseg` layout: a
/// `# meta:` JSON header followed by
/// `sigma sigma_orth area atomic_number` rows.
///
/// The kernel needs each segment's σ and σ⊥ together, so the two are stored
/// per segment rather than as separate histograms: binning them apart would
/// decorrelate the pair the misfit term multiplies. The atomic number rides
/// along for the van der Waals term.
struct ComponentFile {
  std::string name;
  Component component;
  /// Averaging radii the descriptors were built on, checked on load against
  /// the parameters in use.
  double r_av{0.0};
  double r_corr{0.0};
  double sigma_orth_factor{0.0};
  /// What the cavity was computed with. Mixing a solute and solvent built at
  /// different levels is silently wrong, so the provenance travels with the
  /// ensemble rather than being assumed.
  std::string method;
  std::string basis;
};

ComponentFile read_segments(const std::string &path);

void write_segments(const std::string &path, const std::string &name,
                    const Component &component, const Parameters &params,
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

/// Load `name`'s ensemble and check it against the parameters in use.
///
/// The descriptors are meaningless across averaging radii and only
/// comparable at one level of theory, so both are checked here rather than
/// left to each caller: a mismatched radius throws, a mismatched basis warns.
ComponentFile load_solvent(const SegmentStore &store, const std::string &name,
                           const Parameters &params,
                           const std::string &method = {},
                           const std::string &basis = {});

} // namespace occ::solvent::cosmors
