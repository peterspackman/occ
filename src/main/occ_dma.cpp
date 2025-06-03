#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <fmt/os.h>
#include <occ/main/occ_dma.h>
#include <occ/qm/wavefunction.h>
#include <occ/core/timings.h>

namespace occ::main {

CLI::App *add_dma_subcommand(CLI::App &app) {

  CLI::App *dma = app.add_subcommand(
      "dma", "compute distributed multipoles using DMA algorithm");
  auto config = std::make_shared<DMAConfig>();

  dma->add_option("wavefunction", config->wavefunction_filename,
                  "wavefunction file input")
      ->required();
  dma->add_option("--max-rank", config->settings.max_rank,
                  "maximum angular momenta (l_max) for multipoles");
  dma->fallthrough();
  dma->callback([config]() { run_dma_subcommand(*config); });
  return dma;
}

void run_dma_subcommand(const DMAConfig &config) {
  auto wfn = occ::qm::Wavefunction::load(config.wavefunction_filename);

  occ::dma::DMACalculator calc(wfn);

  calc.update_settings(config.settings);
  calc.set_radius_for_element(1, 0.35);
  calc.set_limit_for_element(1, 1);

  occ::timing::start(occ::timing::category::global);

  auto dma_result = calc.compute_multipoles();
  const auto &result = dma_result.multipoles;

  for (int site = 0; site < result.size(); site++) {
    const auto &m = result[site];
    fmt::print("Site: {}\n{}\n", site, m.to_string(config.settings.max_rank));
  }

  occ::timing::stop(occ::timing::category::global);
  occ::timing::print_timings();
}

} // namespace occ::main
