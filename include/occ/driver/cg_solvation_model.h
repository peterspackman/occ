#pragma once
#include <memory>
#include <occ/cg/solvation_data.h>
#include <occ/core/molecule.h>
#include <occ/qm/wavefunction.h>
#include <string>
#include <vector>

namespace occ::driver {

/// Which wavefunction a solvation model can supply for solution-phase work.
enum class WavefunctionChoice;

/// A solvent, possibly a mixture, at a temperature.
///
/// Parsed from `water`, or `water:0.7,ethanol:0.3` for a mixture. Only the
/// σ-potential model accepts mixtures; the others reject them explicitly
/// rather than silently using the first component.
struct SolventSpec {
  std::vector<std::string> components{"water"};
  Vec mole_fractions{Vec::Ones(1)};
  double temperature{298.15};

  [[nodiscard]] bool is_mixture() const { return components.size() > 1; }

  /// The single component name; throws for a mixture.
  [[nodiscard]] const std::string &single() const;

  /// `water` or `water:0.7,ethanol:0.3`. Fractions are normalised; omitting
  /// them for a multi-component spec is an error.
  [[nodiscard]] static SolventSpec parse(const std::string &text);

  [[nodiscard]] std::string to_string() const;

  /// Filesystem-safe label for output filenames: `water`, or
  /// `water0.75-ethanol0.25` for a mixture. Colons and commas are not
  /// portable in filenames.
  [[nodiscard]] std::string filename_tag() const;
};

enum class SolvationModelKind {
  None,  ///< gas phase, no solvation surfaces
  Smd,   ///< SMD via a solvated DFT SCF
  Sigma, ///< COSMO-SAC σ-potential on an ideal-conductor cavity
};

[[nodiscard]] SolvationModelKind parse_solvation_model(const std::string &);
[[nodiscard]] std::string solvation_model_name(SolvationModelKind);

/// Per-monomer solvation surfaces, plus the wavefunctions the model can offer
/// for solution-phase monomer and interaction energies.
struct CGSolvationResult {
  std::vector<cg::SolvationData> surfaces;
  std::vector<qm::Wavefunction> wavefunctions;
};

/// Produces solvation surfaces for the cg pipeline.
///
/// Implementations differ only in how the per-element energies are obtained;
/// everything downstream — partitioning, attribution, the result records —
/// is shared.
class CGSolvationModel {
public:
  virtual ~CGSolvationModel() = default;

  [[nodiscard]] virtual std::string name() const = 0;

  /// Whether the model can supply a wavefunction polarised by the real
  /// solvent. The σ model cannot: its wavefunction is the ε=∞ conductor, so
  /// asking for solvated interaction energies is an error rather than a
  /// silent degradation.
  [[nodiscard]] virtual bool supports_solvated_wavefunctions() const = 0;

  /// Rejects a mixture unless the model handles one.
  virtual void validate(const SolventSpec &) const;

  [[nodiscard]] virtual CGSolvationResult
  compute(const std::string &basename,
          const std::vector<core::Molecule> &molecules,
          const std::vector<qm::Wavefunction> &gas_wavefunctions,
          const SolventSpec &solvent) = 0;
};

/// Settings the concrete models need; a superset, each uses what applies.
struct CGSolvationSettings {
  std::string method{"b3lyp"};
  std::string basis{"6-31g**"};
  bool pure_spherical{false};
  int angular_points{590};
};

[[nodiscard]] std::unique_ptr<CGSolvationModel>
make_cg_solvation_model(SolvationModelKind kind,
                        const CGSolvationSettings &settings);

} // namespace occ::driver
