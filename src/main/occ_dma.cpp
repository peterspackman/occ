#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <fmt/os.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/core/element.h>
#include <occ/core/linear_algebra.h>
#include <occ/driver/dma_driver.h>
#include <occ/main/occ_dma.h>
#include <occ/main/version.h>
#include <occ/qm/wavefunction.h>
#include <fstream>

namespace occ::main {

CLI::App *add_dma_subcommand(CLI::App &app) {

  CLI::App *dma = app.add_subcommand(
      "dma", "compute distributed multipoles using DMA algorithm");
  auto config = std::make_shared<DMAConfig>();

  dma->add_option("wavefunction", config->wavefunction_filename,
                  "wavefunction file input")
      ->required();
  dma->add_option("-p,--punch", config->punch_filename,
                  "punch file output (default: dma.punch)");
  dma->add_option("--max-rank,--limit", config->settings.max_rank,
                  "maximum angular momenta (l_max) for multipoles (default: 4)");
  dma->add_option("--switch", config->settings.big_exponent,
                  "switch parameter for DMA algorithm (default: 4.0)");
  dma->add_flag("--no-punch", [config](int64_t) { config->write_punch = false; },
                  "disable punch file output");
  
  // Atom-specific options
  dma->add_option("--atom-radius", 
                  [config](const std::vector<std::string> &vals) {
                    for (size_t i = 0; i < vals.size(); i += 2) {
                      if (i + 1 < vals.size()) {
                        config->atom_radii[vals[i]] = std::stod(vals[i + 1]);
                      }
                    }
                    return true;
                  },
                  "set radius for specific atoms (e.g., --atom-radius H 0.35 C 0.65)")
      ->expected(-1);
      
  dma->add_option("--atom-limit",
                  [config](const std::vector<std::string> &vals) {
                    for (size_t i = 0; i < vals.size(); i += 2) {
                      if (i + 1 < vals.size()) {
                        config->atom_limits[vals[i]] = std::stoi(vals[i + 1]);
                      }
                    }
                    return true;
                  },
                  "set max rank for specific atoms (e.g., --atom-limit H 1 C 4)")
      ->expected(-1);
  
  // TOML configuration support
  dma->set_config("--config", "dma.toml", "Read TOML configuration", false);
  
  dma->fallthrough();
  dma->callback([config]() { run_dma_subcommand(*config); });
  return dma;
}


void run_dma_subcommand(const DMAConfig &config) {
  occ::main::print_header();
  
  // Use the driver
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
  
  // Need to load wavefunction again to compute total multipoles
  auto wfn = occ::qm::Wavefunction::load(config.wavefunction_filename);
  occ::dma::DMACalculator calc(wfn);
  auto total = calc.compute_total_multipoles(output.result);
  occ::log::info("{}", total.to_string(total.max_rank));
}

} // namespace occ::main
