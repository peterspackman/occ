#include <CLI/App.hpp>
#include <fmt/format.h>
#include <occ/core/element.h>
#include <occ/core/format_matrix.h>
#include <occ/core/linear_algebra.h>
#include <occ/core/units.h>
#include <occ/core/util.h>
#include <occ/driver/dma_driver.h>
#include <occ/io/structure_format.h>
#include <occ/main/occ_dma.h>
#include <occ/main/version.h>
#include <occ/mults/dimer_interaction.h>
#include <occ/mults/dma_force_field.h>

namespace occ::main {

namespace {

// Mult stores a fixed 121-component vector, so rank 10 is the hard ceiling.
constexpr int MAX_SUPPORTED_RANK = 10;

/**
 * Split the token stream of --atom-radius / --atom-limit into element/value
 * pairs. Two spellings are accepted and may be mixed:
 *
 *   --atom-radius H=0.35 --atom-radius C=0.65   one pair per occurrence
 *   --atom-radius H 0.35 C 0.65                 variadic, as documented
 *
 * CLI11 accumulates every occurrence into one token vector, so both arrive
 * here together.
 */
std::vector<std::pair<std::string, std::string>>
split_element_value_pairs(const std::string &flag,
                          const std::vector<std::string> &tokens) {
  std::vector<std::pair<std::string, std::string>> pairs;
  for (size_t i = 0; i < tokens.size();) {
    const auto eq = tokens[i].find('=');
    if (eq != std::string::npos) {
      pairs.emplace_back(tokens[i].substr(0, eq), tokens[i].substr(eq + 1));
      i += 1;
      continue;
    }
    if (i + 1 >= tokens.size()) {
      throw CLI::ValidationError(
          flag, fmt::format("'{}' has no value; expected '{}=<value>' or "
                            "'{} <element> <value>'",
                            tokens[i], tokens[i], flag));
    }
    pairs.emplace_back(tokens[i], tokens[i + 1]);
    i += 2;
  }
  return pairs;
}

/// Canonical element symbol, or a ValidationError naming the bad token.
std::string require_element(const std::string &flag, const std::string &token) {
  const auto symbol = occ::util::capitalize_copy(occ::util::trim_copy(token));
  const occ::core::Element element(symbol, /* exact_match = */ true);
  if (element.atomic_number() == 0) {
    throw CLI::ValidationError(
        flag, fmt::format("unknown element symbol '{}'", token));
  }
  return element.symbol();
}

double require_double(const std::string &flag, const std::string &symbol,
                      const std::string &text) {
  size_t consumed = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &consumed);
  } catch (const std::exception &) {
    consumed = 0;
  }
  if (text.empty() || consumed != text.size()) {
    throw CLI::ValidationError(
        flag, fmt::format("value for {} is not a number: '{}'", symbol, text));
  }
  return value;
}

int require_int(const std::string &flag, const std::string &symbol,
                const std::string &text) {
  size_t consumed = 0;
  int value = 0;
  try {
    value = std::stoi(text, &consumed);
  } catch (const std::exception &) {
    consumed = 0;
  }
  if (text.empty() || consumed != text.size()) {
    throw CLI::ValidationError(
        flag, fmt::format("value for {} is not an integer: '{}'", symbol, text));
  }
  return value;
}

std::string force_field_help() {
  std::string help = "short-range parameter set written with --write-csp-input:";
  for (const auto &model : occ::mults::short_range_model_registry()) {
    help += fmt::format("\n  {:<12s} {}", model.name, model.description);
  }
  return help;
}

} // namespace

CLI::App *add_dma_subcommand(CLI::App &app) {

  CLI::App *dma = app.add_subcommand(
      "dma", "compute distributed multipoles using DMA algorithm");
  auto config = std::make_shared<DMAConfig>();

  dma->add_option("wavefunction", config->wavefunction_filename,
                  "wavefunction file input")
      ->required();

  // --- output ---------------------------------------------------------
  dma->add_option("-p,--punch", config->punch_filename,
                  "punch file output (default: dma.punch)");
  dma->add_flag("--no-punch", [config](int64_t) { config->write_punch = false; },
                "disable punch file output");
  dma->add_option("--json", config->json_filename,
                  "JSON file with the settings, frame and effective per-site "
                  "radii/limits alongside the multipoles");

  // --- site model -----------------------------------------------------
  dma->add_option("--max-rank,--limit", config->settings.max_rank,
                  "maximum angular momenta (l_max) for multipoles (default: 4)")
      ->check(CLI::Range(0, MAX_SUPPORTED_RANK));
  dma->add_option("--switch,--big-exponent", config->settings.big_exponent,
                  "switch parameter for DMA algorithm (default: 4.0)");
  dma->add_flag("--include-nuclei,!--no-nuclei", config->settings.include_nuclei,
                "include the nuclear contribution to the multipoles "
                "(default: on)");

  dma->add_option_function<std::vector<std::string>>(
         "--atom-radius",
         [config](const std::vector<std::string> &vals) {
           for (const auto &[token, text] :
                split_element_value_pairs("--atom-radius", vals)) {
             const auto symbol = require_element("--atom-radius", token);
             const double radius = require_double("--atom-radius", symbol, text);
             if (!(radius > 0.0)) {
               throw CLI::ValidationError(
                   "--atom-radius",
                   fmt::format("radius for {} must be positive, got {}", symbol,
                               radius));
             }
             config->atom_radii[symbol] = radius;
           }
         },
         "site radius in Angstrom for an element, repeatable: "
         "--atom-radius H=0.35 --atom-radius C=0.65 "
         "(--atom-radius H 0.35 C 0.65 also accepted)")
      ->expected(-1)
      ->allow_extra_args()
      ->type_name("<El>=<Angstrom>");

  dma->add_option_function<std::vector<std::string>>(
         "--atom-limit",
         [config](const std::vector<std::string> &vals) {
           for (const auto &[token, text] :
                split_element_value_pairs("--atom-limit", vals)) {
             const auto symbol = require_element("--atom-limit", token);
             const int limit = require_int("--atom-limit", symbol, text);
             if (limit < 0 || limit > MAX_SUPPORTED_RANK) {
               throw CLI::ValidationError(
                   "--atom-limit",
                   fmt::format("rank limit for {} must be in [0, {}], got {}",
                               symbol, MAX_SUPPORTED_RANK, limit));
             }
             config->atom_limits[symbol] = limit;
           }
         },
         "maximum rank for an element, repeatable: --atom-limit H=1 "
         "--atom-limit C=4 (--atom-limit H 1 C 4 also accepted). Clamped to "
         "--max-rank")
      ->expected(-1)
      ->allow_extra_args()
      ->type_name("<El>=<rank>");

  // --- input transform ------------------------------------------------
  dma->add_option_function<std::vector<double>>(
         "--wfn-rotation,--wfn_rotation",
         [config](const std::vector<double> &vals) {
           config->wfn_rotation =
               Eigen::Map<const occ::Mat3RM>(vals.data());
         },
         "rotation applied to the wavefunction before analysis (row major)")
      ->expected(9);

  dma->add_option_function<std::vector<double>>(
         "--wfn-translation,--wfn_translation",
         [config](const std::vector<double> &vals) {
           config->wfn_translation = Eigen::Map<const occ::Vec3>(vals.data());
         },
         "translation applied to the wavefunction before analysis (Angstrom, "
         "after the rotation)")
      ->expected(3);

  // --- molecular orientation ------------------------------------------
  dma->add_option("--axis-method", config->axis_method,
                  "molecular axis method: none, nc, pca, moi (default: none)")
      ->check(CLI::IsMember({"none", "nc", "pca", "moi"}));

  dma->add_option("--axis-atoms", config->axis_atoms,
                  "atom indices for nc axis method (0-based, e.g., --axis-atoms 0 1 2)")
      ->expected(3);

  dma->add_option("--oriented-xyz", config->oriented_xyz_filename,
                  "output filename for the analysed (oriented) geometry");

  dma->add_flag("--write-oriented-xyz",
                [config](int64_t) {
                  config->write_oriented_xyz = true;
                  if (config->oriented_xyz_filename.empty()) {
                    config->oriented_xyz_filename = "oriented.xyz";
                  }
                },
                "write the analysed geometry to oriented.xyz");

  dma->add_option("--axis-file", config->axis_filename,
                  "output filename for neighcrys-compatible axis file");

  dma->add_flag("--write-axis-file",
                [config](int64_t) {
                  config->write_axis_file = true;
                  if (config->axis_filename.empty()) {
                    config->axis_filename = "molecule.mols";
                  }
                },
                "write neighcrys-compatible molecular axis file");

  // --- CSP force-field output -----------------------------------------
  dma->add_option("--write-csp-input", config->csp_input_filename,
                  "write force-field JSON (molecules + multipoles + pair potentials + settings) for CSP programs");

  dma->add_option("--csp-force-field", config->csp_force_field,
                  force_field_help())
      ->check(CLI::IsMember(occ::mults::short_range_model_names()))
      ->capture_default_str();

  // Configuration files are read by the root command, which understands a
  // [dma] section: `occ --config run.toml dma wfn.owf.json`. A set_config()
  // here would shadow that option and silently do nothing -- CLI11 only
  // processes config files on the root App.

  dma->fallthrough();
  dma->callback([config]() { run_dma_subcommand(*config); });
  return dma;
}

void run_dma_subcommand(const DMAConfig &config) {
  occ::main::print_header();

  occ::driver::DMADriver driver(config);
  auto output = driver.run();

  const auto &result = output.result.multipoles;
  const auto &sites = output.sites;

  log::info("{:-<72s}", "DMA multipole moments (au)  ");
  for (int site_index = 0; site_index < result.size(); site_index++) {
    const auto &m = result[site_index];
    const auto pos =
        sites.positions.col(site_index) * occ::units::BOHR_TO_ANGSTROM;
    occ::log::info("{:8s}   x ={:10.6f}  y ={:10.6f}  z ={:10.6f} angstrom",
                   sites.name[site_index], pos.x(), pos.y(), pos.z());
    occ::log::info(
        "           Maximum rank = {:2d}   Radius =  {:4.3f} angstrom",
        m.max_rank, sites.radii(site_index) * occ::units::BOHR_TO_ANGSTROM);
    occ::log::info("{}", m.to_string(m.max_rank));
  }

  occ::Vec3 origin(0, 0, 0);
  occ::log::info("Total multipoles referred to origin at:");
  occ::log::info("{}", format_matrix(origin * occ::units::BOHR_TO_ANGSTROM));
  occ::log::info("{}", output.total.to_string(output.total.max_rank));

  if (!config.csp_input_filename.empty()) {
    occ::mults::DMAForceFieldOptions options;
    options.force_field = config.csp_force_field;
    options.molecule_name = "mol";
    auto basis = occ::mults::build_dma_force_field_basis(sites, result, options);

    occ::io::write_force_field_json(config.csp_input_filename, basis, "mol");
    occ::log::info("Wrote CSP input (basis JSON, {} potential) to {}",
                   basis.potentials.force_field, config.csp_input_filename);
  }
}

} // namespace occ::main
