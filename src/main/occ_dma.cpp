#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <fmt/os.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/main/occ_dma.h>
#include <occ/main/version.h>
#include <occ/qm/wavefunction.h>

namespace occ::main {

CLI::App *add_dma_subcommand(CLI::App &app) {

  CLI::App *dma = app.add_subcommand(
      "dma", "compute distributed multipoles using DMA algorithm");
  auto config = std::make_shared<DMAConfig>();

  dma->add_option("wavefunction", config->wavefunction_filename,
                  "wavefunction file input")
      ->required();
  dma->add_option("-p,--punch-file", config->punch_filename,
                  "punch file input");
  dma->add_option("--max-rank", config->settings.max_rank,
                  "maximum angular momenta (l_max) for multipoles");
  dma->add_option("--dma-exponent", config->settings.big_exponent,
                  "corresponds to the SWITCH flag in gdma (default=4.0)");
  dma->fallthrough();
  dma->callback([config]() { run_dma_subcommand(*config); });
  return dma;
}

void run_dma_subcommand(const DMAConfig &config) {
  occ::main::print_header();
  auto wfn = occ::qm::Wavefunction::load(config.wavefunction_filename);

  log::info("Loading wavefunction from file: {}", config.wavefunction_filename);
  occ::dma::DMACalculator calc(wfn);

  calc.update_settings(config.settings);
  calc.set_radius_for_element(1, 0.35);
  calc.set_limit_for_element(1, 1);

  auto dma_result = calc.compute_multipoles();
  const auto &result = dma_result.multipoles;
  const auto &sites = calc.sites();

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
  auto total = calc.compute_total_multipoles(dma_result);
  occ::log::info("{}", total.to_string(total.max_rank));
}

} // namespace occ::main
