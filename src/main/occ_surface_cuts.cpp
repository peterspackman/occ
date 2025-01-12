#include <occ/core/log.h>
#include <occ/crystal/crystal.h>
#include <occ/crystal/surface.h>
#include <occ/io/load_geometry.h>
#include <occ/main/occ_surface_cuts.h>

using occ::crystal::Crystal;

namespace occ::main {

CLI::App *add_surface_cuts_subcommand(CLI::App &app) {
  CLI::App *cuts =
      app.add_subcommand("surface_cuts", "compute surface cuts for a crystal");
  auto config = std::make_shared<SurfaceCutsConfig>();

  cuts->add_option("crystal", config->filename, "input geometry file (crystal)")
      ->required();

  cuts->add_option("--dmin", config->dmin, "Minimum interplanar spacing");
  cuts->add_option("--count", config->count, "Number of cuts");
  cuts->fallthrough();
  cuts->callback([config]() { run_surface_cuts_subcommand(*config); });
  return cuts;
}

void run_surface_cuts_subcommand(SurfaceCutsConfig config) {
  Crystal crystal = occ::io::load_crystal(config.filename);
  crystal::CrystalSurfaceGenerationParameters params;
  params.d_min = config.dmin;
  params.unique = true;
  auto surfaces = crystal::generate_surfaces(crystal, params);
  log::debug("Top {} surfaces", config.count);
  int number_of_surfaces = 0;
  constexpr double tolerance{1e-5};

  // find unique positions to consider
  Mat3N unique_positions(3, crystal.unit_cell_molecules().size());
  log::debug("Unique positions to check: {}", unique_positions.cols());
  {
    int i = 0;
    for (const auto &mol : crystal.unit_cell_molecules()) {
      unique_positions.col(i) = mol.centroid();
      i++;
    }
  }

  for (auto &surf : surfaces) {
    const auto hkl = surf.hkl();
    log::debug("{:-^72s}",
               fmt::format("  {} {} {} surface  ", hkl.h, hkl.k, hkl.l));
    surf.print();
    auto cuts = surf.possible_cuts(unique_positions);
    log::debug("{} unique cuts", cuts.size());

    for (const double &cut : cuts) {
      log::debug("\nCut @ {:.6f} * depth", cut);
    }

    number_of_surfaces++;
    if (number_of_surfaces >= config.count)
      break;
  }
}

} // namespace occ::main
