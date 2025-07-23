#include <CLI/App.hpp>
#include <CLI/Config.hpp>
#include <CLI/Formatter.hpp>
#include <fmt/os.h>
#include <occ/core/timings.h>
#include <occ/core/units.h>
#include <occ/core/element.h>
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
  dma->add_flag("-v,--verbose", config->verbose,
                  "verbose output");
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

void write_punch_file(const std::string &filename, const DMAConfig &config,
                      const dma::DMAResult &result, const dma::DMASites &sites,
                      const dma::Mult &total) {
  std::ofstream punch(filename);
  if (!punch.is_open()) {
    log::error("Failed to open punch file: {}", filename);
    return;
  }
  
  punch << fmt::format("! Distributed multipoles from occ dma\n");
  punch << fmt::format("! Wavefunction: {}\n", config.wavefunction_filename);
  punch << fmt::format("! Switch parameter: {}\n", config.settings.big_exponent);
  punch << fmt::format("! Max rank: {}\n", config.settings.max_rank);
  punch << fmt::format("\n");
  punch << fmt::format("Units angstrom\n\n");
  
  // Write individual site multipoles
  for (int i = 0; i < result.multipoles.size(); i++) {
    const auto &m = result.multipoles[i];
    const auto pos = sites.positions.col(i) * occ::units::BOHR_TO_ANGSTROM;
    
    punch << fmt::format("{:<8s} {:12.8f} {:12.8f} {:12.8f}\n",
                         sites.name[i], pos.x(), pos.y(), pos.z());
    punch << fmt::format("Rank {}\n", m.max_rank);
    
    // Write multipoles in order: Q00, Q10, Q11c, Q11s, Q20, Q21c, Q21s, Q22c, Q22s, etc.
    int idx = 0;
    for (int rank = 0; rank <= m.max_rank; rank++) {
      int num_components = 2 * rank + 1;
      for (int comp = 0; comp < num_components; comp++) {
        punch << fmt::format(" {:16.10f}", m.q(idx++));
        if ((comp + 1) % 3 == 0 || comp == num_components - 1) {
          punch << "\n";
        }
      }
    }
    punch << "\n";
  }
  
  // GDMA doesn't include total multipoles in punch files
  
  punch.close();
  log::info("Punch file written to: {}", filename);
}

void run_dma_subcommand(const DMAConfig &config) {
  occ::main::print_header();
  auto wfn = occ::qm::Wavefunction::load(config.wavefunction_filename);

  log::info("Loading wavefunction from file: {}", config.wavefunction_filename);
  occ::dma::DMACalculator calc(wfn);

  calc.update_settings(config.settings);
  
  // Apply atom-specific settings
  for (const auto &[element, radius] : config.atom_radii) {
    int atomic_number = occ::core::Element(element).atomic_number();
    calc.set_radius_for_element(atomic_number, radius);
    if (config.verbose) {
      log::info("Setting radius for {} to {:.3f} Angstrom", element, radius);
    }
  }
  
  for (const auto &[element, limit] : config.atom_limits) {
    int atomic_number = occ::core::Element(element).atomic_number();
    calc.set_limit_for_element(atomic_number, limit);
    if (config.verbose) {
      log::info("Setting max rank for {} to {}", element, limit);
    }
  }
  
  // Set default H settings if not specified
  if (config.atom_radii.find("H") == config.atom_radii.end()) {
    calc.set_radius_for_element(1, 0.35);
  }
  if (config.atom_limits.find("H") == config.atom_limits.end()) {
    calc.set_limit_for_element(1, 1);
  }

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
    if (config.verbose) {
      occ::log::info("{}", m.to_string(m.max_rank));
    }
  }

  occ::Vec3 origin(0, 0, 0);
  occ::log::info("Total multipoles referred to origin at:");
  occ::log::info("{}", format_matrix(origin * occ::units::BOHR_TO_ANGSTROM));
  auto total = calc.compute_total_multipoles(dma_result);
  if (config.verbose) {
    occ::log::info("{}", total.to_string(total.max_rank));
  }
  
  // Write punch file if requested
  if (config.write_punch) {
    write_punch_file(config.punch_filename, config, dma_result, sites, total);
  }
}

} // namespace occ::main
