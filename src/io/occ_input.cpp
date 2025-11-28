#include <fstream>
#include <occ/core/atom.h>
#include <occ/core/units.h>
#include <occ/io/occ_input.h>
#include <filesystem>
#include <toml++/toml.hpp>

namespace occ::io {
occ::core::Molecule GeometryInput::molecule() const {
  return occ::core::Molecule(elements, positions);
}

void GeometryInput::set_molecule(const occ::core::Molecule &mol) {
  elements = mol.elements();
  positions.clear();
  const auto &pos = mol.positions();
  positions.reserve(pos.cols());
  for (size_t i = 0; i < elements.size(); i++) {
    positions.push_back(std::array<double, 3>{pos(0, i), pos(1, i), pos(2, i)});
  }
}

OccInput read_occ_input_file(const std::string path) {
  // for now only SCF works.
  OccInput result;
  auto config = toml::parse_file(path);
  auto scf = config["scf"];
  if ( config["cg"] || config["cube"]|| config["describe"]|| config["dimers"]||
     config["dma"]|| config["elastic"]|| config["elat"]|| config["elastic_fit"]||
     config["embed"]|| config["iso"]|| config["pair"]|| config["cuts"])
    throw std::runtime_error("Not yet implemented, only SCF has direct reading implemented.");

  result.runtime.threads = config["threads"].value_or(result.runtime.threads);
  result.verbosity = config["verbosity"].value_or(result.verbosity);
  result.filename = scf["input"].value_or(result.filename);
  result.method.name = scf["method"].value_or(result.method.name);
  result.basis.name = scf["basis"].value_or(result.basis.name);
  result.electronic.charge = scf["charge"].value_or(result.electronic.charge);
  if (auto arr = scf["output"].as_array()) {
    result.output.formats.clear();
    result.output.formats.reserve(arr->size());

    for (auto&& elem : *arr) {
        if (auto val = elem.value<std::string>()) {
            result.output.formats.push_back(*val);
        }
    }
  }
  result.electronic.multiplicity = scf["multiplicity"].value_or(result.electronic.multiplicity);
  result.electronic.spinorbital_kind =  scf["unrestricted"].value_or(result.electronic.spinorbital_kind) ? SpinorbitalKind::Unrestricted : SpinorbitalKind::Restricted;
  result.driver.driver = scf["driver"].value_or(result.driver.driver);
  result.basis.basis_set_directory = scf["basis_set_directory"].value_or(result.basis.basis_set_directory);
  result.method.integral_precision = scf["integral_precision"].value_or(result.method.integral_precision);
  result.method.use_direct_df_kernels = scf["use_direct_df_kernels"].value_or(result.method.use_direct_df_kernels);

  // DFT grid settings
  result.method.dft_grid.max_angular_points = scf["dft_grid_max_angular"].value_or(result.method.dft_grid.max_angular_points);
  result.method.dft_grid.min_angular_points = scf["dft_grid_min_angular"].value_or(result.method.dft_grid.min_angular_points);
  result.method.dft_grid.radial_precision = scf["dft_grid_radial_precision"].value_or(result.method.dft_grid.radial_precision);
  result.method.dft_grid.reduced_first_row_element_grid = scf["dft_grid_reduce_light_elements"].value_or(result.method.dft_grid.reduced_first_row_element_grid);

  // basis set
  result.basis.df_name = scf["df-basis"].value_or(result.basis.df_name);
  result.basis.ri_basis = scf["ri-basis"].value_or(result.basis.ri_basis);
  result.basis.spherical = scf["spherical"].value_or(result.basis.spherical);

  result.method.orbital_smearing_sigma = scf["orbital_smearing_sigma"].value_or(result.method.orbital_smearing_sigma);

  // point charges
  result.geometry.point_charge_filename = scf["point_charge_file"].value_or(result.geometry.point_charge_filename);

  // Solvation
  result.solvent.solvent_name = scf["solvent_name"].value_or(result.solvent.solvent_name);
  result.solvent.output_surface_filename = scf["solvent_file"].value_or(result.solvent.output_surface_filename);
  result.solvent.radii_scaling = scf["solvent_radii_scaling"].value_or(result.solvent.radii_scaling);

  // XDM
  result.dispersion.evaluate_correction = scf["xdm"].value_or(result.dispersion.evaluate_correction);
  result.dispersion.xdm_a1 = scf["xdm_a1"].value_or(result.dispersion.xdm_a1);
  result.dispersion.xdm_a2 = scf["xdm_a2"].value_or(result.dispersion.xdm_a2);

  result.chelpg_filename = scf["chelpg"].value_or(result.chelpg_filename);

  // Optimization convergence criteria
  result.optimization.gradient_max = scf["opt_gradient_max"].value_or(result.optimization.gradient_max);
  result.optimization.gradient_rms = scf["opt_gradient_rms"].value_or(result.optimization.gradient_rms);
  result.optimization.step_max = scf["opt_step_max"].value_or(result.optimization.step_max);
  result.optimization.step_rms = scf["opt_step_rms"].value_or(result.optimization.step_rms);
  result.optimization.energy_change = scf["opt_energy_change"].value_or(result.optimization.energy_change);
  result.optimization.use_energy_criterion = scf["opt_use_energy"].value_or(result.optimization.use_energy_criterion);
  result.optimization.max_iterations = scf["opt_max_iterations"].value_or(result.optimization.max_iterations);
  result.optimization.gradient_integral_precision = scf["opt_gradient_precision"].value_or(result.optimization.gradient_integral_precision);
  result.optimization.early_gradient_integral_precision = scf["opt_early_gradient_precision"].value_or(result.optimization.early_gradient_integral_precision);
  result.optimization.tight_gradient_threshold = scf["opt_tight_threshold"].value_or(result.optimization.tight_gradient_threshold);
  result.optimization.write_wavefunction_steps = scf["opt_write_wavefunctions"].value_or(result.optimization.write_wavefunction_steps);
  result.optimization.compute_frequencies = scf["frequencies"].value_or(result.optimization.compute_frequencies);

  return result;
}

} // namespace occ::io
