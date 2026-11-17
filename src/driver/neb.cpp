#include <filesystem>
#include <fmt/os.h>
#include <occ/core/log.h>
#include <occ/core/units.h>
#include <occ/driver/geometry_optimization.h>
#include <occ/driver/neb.h>
#include <occ/io/xyz.h>
#include <occ/opt/neb.h>
#include <occ/xtb/xtb_calculator.h>

namespace occ::driver {

namespace {

std::string basename(const io::OccInput &config) {
  return config.filename.empty()
             ? std::string("neb")
             : std::filesystem::path(config.filename).stem().string();
}

void write_path(const std::string &filename,
                const std::vector<core::Molecule> &path, const Vec &energies) {
  auto out = fmt::output_file(filename);
  for (size_t k = 0; k < path.size(); k++) {
    const auto &m = path[k];
    out.print("{}\nimage {} Energy={:.9f}\n", m.size(), k, energies(k));
    const auto &el = m.elements();
    const auto &x = m.positions();
    for (int i = 0; i < m.size(); i++)
      out.print("{:2s} {:14.8f} {:14.8f} {:14.8f}\n", el[i].symbol(), x(0, i),
                x(1, i), x(2, i));
  }
}

} // namespace

NebResult nudged_elastic_band(const io::OccInput &config) {
  const auto &opt = config.optimization;
  if (opt.neb_product.empty())
    throw std::runtime_error("NEB: no product geometry (--product)");
  const core::Molecule reactant = config.geometry.molecule();
  const core::Molecule product = io::molecule_from_xyz_file(opt.neb_product);
  log::info("{:=^72s}", "  Nudged Elastic Band  ");
  log::info("Reactant: input geometry; product: {}", opt.neb_product);

  opt::NebSettings settings;
  settings.images = opt.neb_images;
  settings.spring = opt.neb_spring;
  settings.force_max = opt.neb_force_max;
  settings.climbing_force_max = opt.neb_climbing_force_max;
  // the saddle-point search does the rest: a rough band is enough
  if (opt.neb_ts) {
    settings.force_max = std::max(settings.force_max, 2e-2);
    settings.climbing_force_max = std::max(settings.climbing_force_max, 2e-3);
  }
  settings.climbing_image = opt.neb_climbing_image;
  settings.project_net_force_torque = config.geometry.point_charges.empty();
  opt::NudgedElasticBand band(reactant, product, settings);
  log::info("{} images between the endpoints, interpolated by IDPP",
            settings.images);

  // the endpoints, which should be minima
  double endpoint_energy[2];
  {
    const core::Molecule ends[2] = {reactant, product};
    for (int k = 0; k < 2; k++) {
      auto [w, g] = energy_and_gradient_at(config, ends[k]);
      endpoint_energy[k] = w.energy.total;
      const double gmax = g.colwise().norm().maxCoeff();
      if (gmax > 1e-3)
        log::warn("NEB: the {} is not a minimum (largest gradient {:.2e} "
                  "Ha/Bohr); optimise it first, or the band may not settle",
                  k == 0 ? "reactant" : "product", gmax);
    }
    band.set_endpoint_energies(endpoint_energy[0], endpoint_energy[1]);
  }

  // one warm start per image
  const int n = settings.images;
  std::vector<std::optional<qm::Wavefunction>> previous(n);
  std::vector<std::optional<xtb::XtbCalculator>> gfn2(n);
  const std::string path_file = basename(config) + "_neb.xyz";

  NebResult result;
  for (int iteration = 0; iteration < opt.neb_max_iterations; iteration++) {
    const auto images = band.images();
    std::vector<double> energies(n);
    std::vector<Mat3N> gradients(n);
    for (int k = 0; k < n; k++) {
      auto [w, g] = energy_and_gradient_at(
          config, images[k], previous[k] ? &*previous[k] : nullptr, &gfn2[k]);
      energies[k] = w.energy.total;
      gradients[k] = g;
      previous[k] = std::move(w);
    }
    band.update(energies, gradients);
    const bool done = band.step();
    log::info("NEB iteration {:3d}: largest force {:.3e} Ha/Bohr, barrier "
              "{:.6f} Ha{}",
              iteration, band.max_force(),
              band.energies()(band.highest_image()) - endpoint_energy[0],
              band.climbing() ? " (climbing)" : "");
    if (done) {
      result.converged = true;
      break;
    }
    write_path(path_file, band.path(), band.energies());
  }
  if (!result.converged)
    log::warn("NEB did not converge in {} iterations", opt.neb_max_iterations);

  result.path = band.path();
  result.energies = band.energies();
  result.highest = band.highest_image();
  write_path(path_file, result.path, result.energies);

  log::info("Energy profile (kJ/mol from the reactant):");
  double arc = 0.0;
  for (size_t k = 0; k < result.path.size(); k++) {
    if (k > 0)
      arc += (result.path[k].positions() - result.path[k - 1].positions())
                 .norm();
    log::info("  {:3d} {:10.3f} Angstrom {:12.3f}{}", k, arc,
              (result.energies(k) - result.energies(0)) *
                  occ::units::AU_TO_KJ_PER_MOL,
              int(k) == result.highest ? "  <- highest" : "");
  }
  log::info("Path written to {}", path_file);

  if (opt.neb_ts) {
    log::info("{:=^72s}", "  Transition state from the highest image  ");
    io::OccInput ts_config = config;
    ts_config.geometry.set_molecule(result.path[result.highest]);
    ts_config.optimization.transition_state = true;
    ts_config.filename = basename(config) + "_ts.xyz";
    const Vec tangent = band.tangent(result.highest);
    auto [wfn, modes] =
        geometry_optimization_with_frequencies(ts_config, true, tangent);
    result.transition_state = std::move(wfn);
    result.ts_modes = std::move(modes);
  }
  return result;
}

} // namespace occ::driver
