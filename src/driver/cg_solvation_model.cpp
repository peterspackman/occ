#include <occ/cg/smd_solvation.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/driver/cg_solvation_model.h>
#include <occ/driver/cosmors_solvation.h>
#include <sstream>
#include <stdexcept>

namespace occ::driver {

const std::string &SolventSpec::single() const {
  if (is_mixture())
    throw std::runtime_error(fmt::format(
        "expected a single solvent but got the mixture '{}'", to_string()));
  return components.front();
}

SolventSpec SolventSpec::parse(const std::string &text) {
  SolventSpec spec;
  spec.components.clear();
  std::vector<double> fractions;

  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    token = occ::util::trim_copy(token);
    if (token.empty())
      continue;
    const auto colon = token.find(':');
    if (colon == std::string::npos) {
      spec.components.push_back(token);
      fractions.push_back(-1.0);
    } else {
      spec.components.push_back(occ::util::trim_copy(token.substr(0, colon)));
      fractions.push_back(std::stod(token.substr(colon + 1)));
    }
  }

  if (spec.components.empty())
    throw std::runtime_error(fmt::format("could not parse solvent '{}'", text));

  const bool any_missing =
      std::any_of(fractions.begin(), fractions.end(),
                  [](double f) { return f < 0.0; });
  if (spec.components.size() == 1) {
    spec.mole_fractions = Vec::Ones(1);
    return spec;
  }
  if (any_missing)
    throw std::runtime_error(fmt::format(
        "solvent mixture '{}' needs a mole fraction for every component, "
        "e.g. water:0.7,ethanol:0.3",
        text));

  spec.mole_fractions =
      Eigen::Map<Vec>(fractions.data(), fractions.size());
  const double sum = spec.mole_fractions.sum();
  if (sum <= 0.0)
    throw std::runtime_error(
        fmt::format("solvent mixture '{}' has non-positive mole fractions",
                    text));
  spec.mole_fractions /= sum;
  return spec;
}

std::string SolventSpec::to_string() const {
  if (!is_mixture())
    return components.front();
  std::string out;
  for (size_t i = 0; i < components.size(); i++) {
    if (i > 0)
      out += ",";
    out += fmt::format("{}:{:.4g}", components[i], mole_fractions(i));
  }
  return out;
}

std::string SolventSpec::filename_tag() const {
  if (!is_mixture())
    return components.front();
  std::string out;
  for (size_t i = 0; i < components.size(); i++) {
    if (i > 0)
      out += "-";
    out += fmt::format("{}{:.3g}", components[i], mole_fractions(i));
  }
  return out;
}

SolvationModelKind parse_solvation_model(const std::string &name) {
  const auto lowered = occ::util::to_lower_copy(name);
  if (lowered == "none" || lowered == "gas")
    return SolvationModelKind::None;
  if (lowered == "smd")
    return SolvationModelKind::Smd;
  if (lowered == "cosmo-rs" || lowered == "cosmors" ||
      lowered == "opencosmo-rs" || lowered == "cosmors")
    return SolvationModelKind::CosmoRS;
  throw std::runtime_error(fmt::format(
      "unknown solvation model '{}' (none, smd, cosmo-rs)", name));
}

std::string solvation_model_name(SolvationModelKind kind) {
  switch (kind) {
  case SolvationModelKind::None:
    return "none";
  case SolvationModelKind::Smd:
    return "smd";
  case SolvationModelKind::CosmoRS:
    return "cosmo-rs";
  }
  return "unknown";
}

void CGSolvationModel::validate(const SolventSpec &solvent) const {
  if (solvent.is_mixture())
    throw std::runtime_error(
        fmt::format("{} does not support solvent mixtures (got '{}'); only "
                    "cosmo-rs does",
                    name(), solvent.to_string()));
}

namespace {

/// RT ln(24.46) at 298 K, in Hartree: the ideal gas at 1 atm against the same
/// gas at 1 mol/L. Owed by any model whose ΔG_solv starts from a 1 mol/L gas.
constexpr double gas_concentration_shift = 1.89 / occ::units::AU_TO_KCAL_PER_MOL;

class SmdCGSolvationModel final : public CGSolvationModel {
public:
  explicit SmdCGSolvationModel(CGSolvationSettings settings)
      : m_settings(std::move(settings)) {}

  std::string name() const override { return "SMD"; }

  /// SMD is parameterised against the Minnesota solvation database, whose
  /// ΔG_solv runs from a 1 mol/L gas to a 1 mol/L solution.
  double standard_state_shift() const override {
    return gas_concentration_shift;
  }
  bool supports_solvated_wavefunctions() const override { return true; }

  CGSolvationResult
  compute(const std::string &basename,
          const std::vector<core::Molecule> &molecules,
          const std::vector<qm::Wavefunction> &gas_wavefunctions,
          const SolventSpec &solvent) override {
    validate(solvent);
    cg::SMDSettings smd;
    smd.method = m_settings.method;
    smd.basis = m_settings.basis;
    smd.pure_spherical = m_settings.pure_spherical;
    smd.temperature = solvent.temperature;

    cg::SMDCalculator calculator(basename, molecules, gas_wavefunctions,
                                 solvent.single(), smd);
    auto result = calculator.calculate();
    return {std::move(result.surfaces), std::move(result.wavefunctions)};
  }

private:
  CGSolvationSettings m_settings;
};

class CosmoRSSolvationModel final : public CGSolvationModel {
public:
  explicit CosmoRSSolvationModel(CGSolvationSettings settings)
      : m_settings(std::move(settings)) {}

  std::string name() const override { return "openCOSMO-RS"; }

  /// The same convention as SMD: `[cosmors]` checks the assembled total
  /// against the same experimental hydration free energies, and it lands on
  /// them with a mean deviation of -0.09 kJ/mol, so it is a 1 mol/L to
  /// 1 mol/L quantity too.
  double standard_state_shift() const override {
    return gas_concentration_shift;
  }
  /// The ideal-conductor wavefunction is over-polarised for interaction
  /// energies, so cg must use the gas-phase one.
  bool supports_solvated_wavefunctions() const override { return false; }
  void validate(const SolventSpec &) const override {} // mixtures welcome

  CGSolvationResult
  compute(const std::string &basename,
          const std::vector<core::Molecule> &molecules,
          const std::vector<qm::Wavefunction> &gas_wavefunctions,
          const SolventSpec &solvent) override {
    CosmoRSSettings settings;
    settings.method = m_settings.method;
    settings.basis = m_settings.basis;
    settings.pure_spherical = m_settings.pure_spherical;
    settings.angular_points = m_settings.angular_points;
    settings.temperature = solvent.temperature;
    settings.volume_per_molecule = m_settings.volume_per_molecule;
    return cosmors_solvation(basename, molecules, gas_wavefunctions,
                                 solvent, settings);
  }

private:
  CGSolvationSettings m_settings;
};

class NoSolvationModel final : public CGSolvationModel {
public:
  std::string name() const override { return "none"; }
  bool supports_solvated_wavefunctions() const override { return false; }
  void validate(const SolventSpec &) const override {}

  CGSolvationResult
  compute(const std::string &, const std::vector<core::Molecule> &,
          const std::vector<qm::Wavefunction> &gas_wavefunctions,
          const SolventSpec &) override {
    CGSolvationResult result;
    result.surfaces.resize(gas_wavefunctions.size());
    result.wavefunctions = gas_wavefunctions;
    return result;
  }
};

} // namespace

std::unique_ptr<CGSolvationModel>
make_cg_solvation_model(SolvationModelKind kind,
                        const CGSolvationSettings &settings) {
  switch (kind) {
  case SolvationModelKind::None:
    return std::make_unique<NoSolvationModel>();
  case SolvationModelKind::Smd:
    return std::make_unique<SmdCGSolvationModel>(settings);
  case SolvationModelKind::CosmoRS:
    return std::make_unique<CosmoRSSolvationModel>(settings);
  }
  throw std::runtime_error("make_cg_solvation_model: unknown kind");
}

} // namespace occ::driver
