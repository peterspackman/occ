#include <CLI/App.hpp>
#include <occ/core/element.h>
#include <occ/core/units.h>
#include <occ/core/linear_algebra.h>
#include <occ/driver/dma_driver.h>
#include <occ/io/structure_format.h>
#include <occ/main/occ_dma.h>
#include <occ/main/version.h>
#include <occ/mults/force_field_params.h>
#include <occ/qm/wavefunction.h>

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
  
  // Molecular orientation options
  dma->add_option("--axis-method", config->axis_method,
                  "molecular axis method: none, nc, pca, moi (default: none)")
      ->check(CLI::IsMember({"none", "nc", "pca", "moi"}));
  
  dma->add_option("--axis-atoms", config->axis_atoms,
                  "atom indices for nc axis method (0-based, e.g., --axis-atoms 0 1 2)")
      ->expected(3);
  
  dma->add_option("--oriented-xyz", config->oriented_xyz_filename,
                  "output filename for oriented molecule XYZ coordinates");
  
  dma->add_flag("--write-oriented-xyz", 
                [config](int64_t) { 
                  config->write_oriented_xyz = true; 
                  if (config->oriented_xyz_filename.empty()) {
                    config->oriented_xyz_filename = "oriented.xyz";
                  }
                },
                "write oriented molecule coordinates to XYZ file");
  
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
  
  dma->add_option("--write-csp-input", config->csp_input_filename,
                  "write basis JSON file (molecules + multipoles + potentials) for CSP programs");

  // TOML configuration support
  dma->set_config("--config", "dma.toml", "Read TOML configuration", false);
  
  dma->fallthrough();
  dma->callback([config]() { run_dma_subcommand(*config); });
  return dma;
}


void run_dma_subcommand(const DMAConfig &config) {
  occ::main::print_header();

  // When writing CSP input, default to MOI orientation so multipoles
  // are computed in the principal axis frame.
  DMAConfig effective_config = config;
  if (!config.csp_input_filename.empty() && config.axis_method == "none") {
    effective_config.axis_method = "moi";
    occ::log::info("CSP output requested: orienting molecule on principal axes");
  }

  occ::driver::DMADriver driver(effective_config);
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

  // Write CSP input (basis JSON) if requested
  if (!config.csp_input_filename.empty()) {
    occ::io::Basis basis;

    // Build molecule type from DMA output
    occ::io::MoleculeType mt;
    mt.name = "mol";

    // Collect atomic numbers and positions for Williams typing
    std::vector<int> atomic_numbers;
    std::vector<occ::Vec3> body_positions;

    for (int i = 0; i < static_cast<int>(result.size()); ++i) {
      const auto &m = result[i];
      occ::io::MoleculeSite ms;
      ms.label = sites.name[i];

      // Get element from site name (first 1-2 chars)
      std::string name = sites.name[i];
      // Try 2-char element first, then 1-char
      int z = 0;
      if (name.size() >= 2) {
        std::string sym2;
        sym2 += static_cast<char>(std::toupper(static_cast<unsigned char>(name[0])));
        sym2 += static_cast<char>(std::tolower(static_cast<unsigned char>(name[1])));
        z = occ::core::Element(sym2).atomic_number();
      }
      if (z == 0 && !name.empty()) {
        std::string sym1(1, static_cast<char>(std::toupper(static_cast<unsigned char>(name[0]))));
        z = occ::core::Element(sym1).atomic_number();
      }
      ms.element = occ::core::Element(z).symbol();

      // Position in Angstrom (body frame, relative to COM)
      occ::Vec3 pos_ang =
          sites.positions.col(i) * occ::units::BOHR_TO_ANGSTROM;
      ms.position = {pos_ang.x(), pos_ang.y(), pos_ang.z()};

      // Multipoles
      int n_comp = m.num_components();
      std::vector<double> flat(n_comp);
      for (int j = 0; j < n_comp; ++j) {
        flat[j] = m.q(j);
      }
      ms.multipoles = occ::io::SiteMultipoles::from_flat(flat);

      mt.sites.push_back(std::move(ms));

      atomic_numbers.push_back(z);
      body_positions.push_back(pos_ang);
    }

    // Center positions on mass-weighted COM
    occ::Vec3 com = occ::Vec3::Zero();
    double total_mass = 0.0;
    for (int i = 0; i < static_cast<int>(atomic_numbers.size()); ++i) {
      double mass = occ::core::Element(atomic_numbers[i]).mass();
      com += mass * body_positions[i];
      total_mass += mass;
    }
    if (total_mass > 0) com /= total_mass;
    for (auto &s : mt.sites) {
      s.position[0] -= com.x();
      s.position[1] -= com.y();
      s.position[2] -= com.z();
    }
    for (auto &p : body_positions) {
      p -= com;
    }

    // Williams atom typing
    auto neighbors = occ::mults::ForceFieldParams::bonded_neighbors(
        atomic_numbers, body_positions);
    for (int i = 0; i < static_cast<int>(atomic_numbers.size()); ++i) {
      int type_code = occ::mults::ForceFieldParams::classify_williams_type(
          i, neighbors, atomic_numbers);
      const char *label =
          occ::mults::ForceFieldParams::short_range_type_label(type_code);
      if (label && label[0] != '\0') {
        mt.sites[i].type = label;
      } else {
        mt.sites[i].type = mt.sites[i].element;
      }
    }

    // Collect type codes present in this molecule
    std::set<int> present_types;
    for (const auto &s : mt.sites) {
      // Look up type code from label
      for (const auto &[key, params] :
           occ::mults::ForceFieldParams::williams_typed_params()) {
        const char *l = occ::mults::ForceFieldParams::short_range_type_label(
            key.first);
        if (l && s.type == l) present_types.insert(key.first);
        l = occ::mults::ForceFieldParams::short_range_type_label(key.second);
        if (l && s.type == l) present_types.insert(key.second);
      }
    }

    basis.molecule_types.push_back(std::move(mt));

    // Williams DE typed Buckingham potentials — only pairs relevant to
    // types present in the molecule
    for (const auto &[key, params] :
         occ::mults::ForceFieldParams::williams_typed_params()) {
      if (key.first > key.second) continue;
      if (!present_types.count(key.first) ||
          !present_types.count(key.second))
        continue;
      occ::io::BuckinghamPair bp;
      const char *l1 =
          occ::mults::ForceFieldParams::short_range_type_label(key.first);
      const char *l2 =
          occ::mults::ForceFieldParams::short_range_type_label(key.second);
      bp.types = {l1 ? l1 : "", l2 ? l2 : ""};
      bp.elements = {
          occ::core::Element(
              occ::mults::ForceFieldParams::short_range_type_atomic_number(
                  key.first))
              .symbol(),
          occ::core::Element(
              occ::mults::ForceFieldParams::short_range_type_atomic_number(
                  key.second))
              .symbol()};
      bp.A = params.A * occ::units::KJ_PER_MOL_TO_EV;
      bp.rho = 1.0 / params.B;
      bp.C6 = params.C * occ::units::KJ_PER_MOL_TO_EV;
      basis.potentials.buckingham.push_back(std::move(bp));
    }

    occ::io::write_basis_json(config.csp_input_filename, basis, "mol");
    occ::log::info("Wrote CSP input (basis JSON) to {}",
                   config.csp_input_filename);
  }
}

} // namespace occ::main
