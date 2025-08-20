#include <fmt/os.h>
#include <fstream>
#include <iomanip>
#include <filesystem>
#include <occ/core/data_directory.h>
#include <occ/core/units.h>
#include <occ/dft/dft.h>
#include <occ/driver/geometry_optimization.h>
#include <occ/driver/vibrational_analysis.h>
#include <occ/driver/method_parser.h>
#include <occ/opt/berny_optimizer.h>
#include <occ/qm/gradients.h>
#include <occ/qm/hf.h>
#include <occ/qm/scf.h>

using occ::core::Molecule;
using occ::dft::DFT;
using occ::io::OccInput;
using occ::qm::HartreeFock;
using occ::qm::SCF;
using occ::qm::SpinorbitalKind;
using occ::qm::Wavefunction;

namespace occ::driver {

occ::qm::AOBasis load_basis_set_opt(const Molecule &m, const std::string &name,
                                    bool spherical) {
  auto basis = occ::qm::AOBasis::load(m.atoms(), name);
  basis.set_pure(spherical);
  log::info("Loaded basis set: {}", spherical ? "spherical" : "cartesian");
  log::info("Number of shells:            {}", basis.size());
  log::info("Number of  basis functions:  {}", basis.nbf());
  log::info("Maximum angular momentum:    {}", basis.l_max());
  return basis;
}

template <typename T, SpinorbitalKind SK>
std::pair<Wavefunction, Mat3N>
run_method_for_optimization(const Molecule &m, const occ::qm::AOBasis &basis,
                            const OccInput &config,
                            const Wavefunction *prev_wfn = nullptr,
                            double energy_change = 1.0) {

  T proc = [&]() {
    if constexpr (std::is_same<T, DFT>::value)
      return T(config.method.name, basis, config.method.dft_grid);
    else
      return T(basis);
  }();

  if (!config.basis.df_name.empty())
    proc.set_density_fitting_basis(config.basis.df_name);

  occ::log::info("Spinorbital kind: {}", spinorbital_kind_to_string(SK));

  // Use adaptive integral precision based on energy change
  double integral_precision = config.method.integral_precision;
  double gradient_precision = config.optimization.gradient_integral_precision;
  
  if (std::abs(energy_change) > config.optimization.tight_gradient_threshold) {
    gradient_precision = config.optimization.early_gradient_integral_precision;
    occ::log::debug("Using looser gradient precision ({:.1e}) for early optimization", 
                    gradient_precision);
  } else {
    occ::log::debug("Using tight gradient precision ({:.1e}) near convergence", 
                    gradient_precision);
  }
  
  occ::log::trace("Setting integral precision: {}", integral_precision);
  proc.set_precision(integral_precision);

  SCF<T> scf(proc, SK);
  occ::log::trace("Setting system charge: {}", config.electronic.charge);
  occ::log::trace("Setting system multiplicity: {}",
                  config.electronic.multiplicity);
  scf.set_charge_multiplicity(config.electronic.charge,
                              config.electronic.multiplicity);
  scf.set_point_charges(config.geometry.point_charges);
  if (!config.basis.df_name.empty()) {
    scf.convergence_settings.incremental_fock_threshold = 0.0;
  }

  if (config.method.orbital_smearing_sigma != 0.0) {
    scf.ctx.mo.smearing.kind = occ::qm::OrbitalSmearing::Kind::Fermi;
    scf.ctx.mo.smearing.sigma = config.method.orbital_smearing_sigma;
  }

  // Use previous wavefunction's MO coefficients as initial guess if available
  if (prev_wfn != nullptr) {
    scf.set_initial_guess_from_wfn(*prev_wfn);
  }

  double e = scf.compute_scf_energy();
  if constexpr (std::is_same<T, DFT>::value) {
    double enlc = proc.post_scf_nlc_correction(scf.ctx.mo);
    if (enlc != 0.0) {
      log::info("Post SCF NLC correction:         {: 20.12f}", enlc);
      e += enlc;
      log::info("Corrected total energy:          {: 20.12f}", e);
    }
  }

  if (config.method.orbital_smearing_sigma != 0.0) {
    log::info("Correlation entropy approx.      {: 20.12f}",
              scf.ctx.mo.smearing.ec_entropy());
    log::info("Free energy                      {: 20.12f}",
              e + scf.ctx.mo.smearing.ec_entropy());
    log::info("Energy (zero point)              {: 20.12f}",
              e + 0.5 * scf.ctx.mo.smearing.ec_entropy());
  }

  auto wfn = scf.wavefunction();
  
  // Set gradient-specific precision if different from SCF precision
  if (gradient_precision != integral_precision) {
    proc.set_precision(gradient_precision);
  }
  
  occ::qm::GradientEvaluator eval(scf.m_procedure);
  Mat3N gradient = eval(wfn.mo);
  // Convert gradients from Hartree/Bohr to Hartree/Angstrom for optimizer
  gradient /= occ::units::ANGSTROM_TO_BOHR;
  return {wfn, gradient};
}

std::pair<Wavefunction, Mat3N> optimization_step_driver(const OccInput &config,
                                                        const Molecule &m,
                                                        const Wavefunction *prev_wfn = nullptr,
                                                        double energy_change = 1.0) {
  constexpr auto R = SpinorbitalKind::Restricted;
  constexpr auto U = SpinorbitalKind::Unrestricted;
  constexpr auto G = SpinorbitalKind::General;

  if (!config.basis.basis_set_directory.empty()) {
    occ::log::info("Overriding environment basis set directory with: '{}'",
                   config.basis.basis_set_directory);
    occ::set_data_directory(config.basis.basis_set_directory);
  }
  auto basis = load_basis_set_opt(m, config.basis.name, config.basis.spherical);
  auto method_kind = method_kind_from_string(config.method.name);
  auto guess_sk = determine_spinorbital_kind(
      config.method.name, config.electronic.multiplicity, method_kind);
  auto conf_sk = config.electronic.spinorbital_kind;

  if (config.solvent.solvent_name.empty()) {
    switch (method_kind) {
    case MethodKind::HF: {
      if (guess_sk == U || conf_sk == U)
        return run_method_for_optimization<HartreeFock, U>(m, basis, config, prev_wfn, energy_change);
      else if (guess_sk == G || conf_sk == G)
        return run_method_for_optimization<HartreeFock, G>(m, basis, config, prev_wfn, energy_change);
      else
        return run_method_for_optimization<HartreeFock, R>(m, basis, config, prev_wfn, energy_change);
      break;
    }
    case MethodKind::DFT: {
      if (guess_sk == U || conf_sk == U)
        return run_method_for_optimization<DFT, U>(m, basis, config, prev_wfn, energy_change);
      else if (guess_sk == G || conf_sk == G)
        throw std::runtime_error(
            "Not implemented: DFT general spinorbital gradients");
      else
        return run_method_for_optimization<DFT, R>(m, basis, config, prev_wfn, energy_change);

      break;
    }
    case MethodKind::MP2: {
      throw std::runtime_error(
          "Not implemented: MP2 gradients for geometry optimization");
      break;
    }
    }
  } else {
    throw std::runtime_error("Not implemented: Solvated gradients");
  }
}

Wavefunction geometry_optimization(const OccInput &config) {

  Wavefunction wfn;
  Mat3N gradient;
  Molecule m = config.geometry.molecule();
  int current_step{0};
  
  // Create trajectory filename based on input filename
  std::string traj_filename;
  if (!config.filename.empty()) {
    std::filesystem::path input_path(config.filename);
    std::string basename = input_path.stem().string();
    traj_filename = basename + "_trj.xyz";
  } else {
    traj_filename = "opt_traj.xyz"; // Fallback to original name
  }
  auto traj = fmt::output_file(traj_filename);
  occ::log::info("Writing optimization trajectory to: {}", traj_filename);

  // Set up convergence criteria from config
  occ::opt::ConvergenceCriteria criteria;
  criteria.gradient_rms = config.optimization.gradient_rms;
  criteria.gradient_max = config.optimization.gradient_max;
  criteria.step_rms = config.optimization.step_rms;
  criteria.step_max = config.optimization.step_max;
  criteria.energy_change = config.optimization.energy_change;
  criteria.use_energy_criterion = config.optimization.use_energy_criterion;
  criteria.max_iterations = config.optimization.max_iterations;
  
  // Initialize Berny optimizer
  occ::opt::BernyOptimizer optimizer(m, criteria);
  occ::log::info("Geometry optimization convergence criteria (internal coordinates):");
  occ::log::info("  Max gradient:    {:.2e}", criteria.gradient_max);
  occ::log::info("  RMS gradient:    {:.2e}", criteria.gradient_rms);
  occ::log::info("  Max step:        {:.2e}", criteria.step_max);
  occ::log::info("  RMS step:        {:.2e}", criteria.step_rms);
  if (criteria.use_energy_criterion) {
    occ::log::info("  Energy change:   {:.2e} Hartree", criteria.energy_change);
  }
  occ::log::info("  Max iterations:  {}", criteria.max_iterations);

  // Open detailed step log
  auto step_log = fmt::output_file("occ_step_log.txt");
  step_log.print("OCC BernyOptimizer Step-by-Step Log\n");
  step_log.print("===================================\n\n");
  
  // Step 0: Compute initial energy and gradient
  occ::log::info("Computing initial wavefunction for optimization");
  std::tie(wfn, gradient) = optimization_step_driver(config, m);
  
  // Write initial geometry to trajectory
  const auto &el = m.elements();
  const auto &positions = m.positions();
  traj.print("{}\nStep {} Energy={:.9f}\n", el.size(), current_step, wfn.energy.total);
  for (int i = 0; i < el.size(); i++) {
    traj.print("{} {:12.6f} {:12.6f} {:12.6f}\n", el[i].symbol(),
               positions(0, i), positions(1, i), positions(2, i));
  }
  traj.flush();

  // Log initial step details
  step_log.print("STEP {} INPUT:\n", current_step);
  step_log.print("Geometry (Angstrom):\n");
  for (int i = 0; i < el.size(); i++) {
    step_log.print("  {} {:2d}:  {:12.8f}  {:12.8f}  {:12.8f}\n", el[i].symbol(), i,
                   positions(0, i), positions(1, i), positions(2, i));
  }
  step_log.print("\n");
  
  step_log.print("STEP {} RESULTS:\n", current_step);
  step_log.print("Energy: {:.12f} Ha\n", wfn.energy.total);
  step_log.print("|gradient|: {:.8f}\n", gradient.norm());
  step_log.print("RMS gradient: {:.8f}\n", gradient.norm() / std::sqrt(gradient.size()));
  step_log.print("Max gradient: {:.8f}\n", gradient.cwiseAbs().maxCoeff());
  step_log.print("Gradient (Hartree/Angstrom):\n");
  for (int i = 0; i < el.size(); i++) {
    step_log.print("  {} {:2d}:  {:11.8f}  {:11.8f}  {:11.8f}\n", el[i].symbol(), i,
                   gradient(0, i), gradient(1, i), gradient(2, i));
  }
  step_log.print("\n");
  step_log.flush();

  occ::log::info("Initial energy:          {:.12f} Ha", wfn.energy.total);
  occ::log::info("Initial RMS gradient:    {:.5e} Ha/Angstrom", 
                 gradient.norm() / std::sqrt(gradient.size()));
  occ::log::info("Initial max gradient:    {:.5e} Ha/Angstrom", 
                 gradient.cwiseAbs().maxCoeff());

  // Write initial wavefunction if requested
  if (config.optimization.write_wavefunction_steps) {
    std::string wfn_filename = "opt_step_000.owf.json";
    wfn.save(wfn_filename);
    occ::log::debug("Wrote initial wavefunction to {}", wfn_filename);
  }

  // Update optimizer with initial data
  optimizer.update(wfn.energy.total, gradient);
  
  // Main optimization loop
  bool converged = false;
  
  while (!converged && current_step < criteria.max_iterations) {
    current_step++;
    
    // Take optimization step
    converged = optimizer.step();
    if (converged) {
      occ::log::info("Optimization converged in step {}", current_step);
      break;
    }
    
    // Get new geometry from optimizer
    Molecule m_new = optimizer.get_next_geometry();
    const auto &new_positions = m_new.positions();
    
    // Log input geometry for this step
    step_log.print("STEP {} INPUT:\n", current_step);
    step_log.print("Geometry (Angstrom):\n");
    for (int i = 0; i < el.size(); i++) {
      step_log.print("  {} {:2d}:  {:12.8f}  {:12.8f}  {:12.8f}\n", el[i].symbol(), i,
                     new_positions(0, i), new_positions(1, i), new_positions(2, i));
    }
    step_log.print("\n");
    
    occ::log::info("Computing wavefunction for optimization step {}", current_step);
    
    // Compute energy and gradient at new geometry, using previous wavefunction as guess
    double prev_energy = wfn.energy.total;
    // For step 1, use large energy change; for later steps, use previous step's change
    static double last_energy_change = 1.0;
    std::tie(wfn, gradient) = optimization_step_driver(config, m_new, &wfn, last_energy_change);
    
    // Write geometry to trajectory
    traj.print("{}\nStep {} Energy={:.9f}\n", el.size(), current_step, wfn.energy.total);
    for (int i = 0; i < el.size(); i++) {
      traj.print("{} {:12.6f} {:12.6f} {:12.6f}\n", el[i].symbol(),
                 new_positions(0, i), new_positions(1, i), new_positions(2, i));
    }
    traj.flush();
    
    // Log results for this step
    double energy_change = wfn.energy.total - prev_energy;
    last_energy_change = std::abs(energy_change);
    double rms_gradient = gradient.norm() / std::sqrt(gradient.size());
    double max_gradient = gradient.cwiseAbs().maxCoeff();
    
    step_log.print("STEP {} RESULTS:\n", current_step);
    step_log.print("Energy: {:.12f} Ha\n", wfn.energy.total);
    step_log.print("|gradient|: {:.8f}\n", gradient.norm());
    step_log.print("RMS gradient: {:.8f}\n", rms_gradient);
    step_log.print("Max gradient: {:.8f}\n", max_gradient);
    step_log.print("Gradient (Hartree/Angstrom):\n");
    for (int i = 0; i < el.size(); i++) {
      step_log.print("  {} {:2d}:  {:11.8f}  {:11.8f}  {:11.8f}\n", el[i].symbol(), i,
                     gradient(0, i), gradient(1, i), gradient(2, i));
    }
    step_log.print("\n");
    step_log.flush();
    
    occ::log::info("Step {:2d} (Cartesian coordinates):", current_step);
    occ::log::info("  Energy:           {:.12f} Ha", wfn.energy.total);
    occ::log::info("  Energy change:    {:.5e} Ha", energy_change);
    occ::log::info("  RMS gradient:     {:.5e} Ha/Angstrom", rms_gradient);
    occ::log::info("  Max gradient:     {:.5e} Ha/Angstrom", max_gradient);
    
    // Write wavefunction if requested
    if (config.optimization.write_wavefunction_steps) {
      std::string wfn_filename = fmt::format("opt_step_{:03d}.owf.json", current_step);
      wfn.save(wfn_filename);
      occ::log::debug("Wrote wavefunction for step {} to {}", current_step, wfn_filename);
    }
    
    // Update optimizer with new energy and gradient
    optimizer.update(wfn.energy.total, gradient);
  }

  if (converged) {
    occ::log::info("Optimization converged successfully after {} steps", current_step);
    step_log.print("Optimization converged after {} steps!\n", current_step);
  } else {
    occ::log::warn("Optimization did not converge after {} steps", current_step);
    step_log.print("Optimization did not converge after {} steps.\n", current_step);
  }
  
  step_log.print("Final energy: {:20.12f} Ha\n", wfn.energy.total);
  occ::log::info("Final energy: {:20.12f}", wfn.energy.total);

  // Write final optimized geometry
  std::string opt_filename;
  if (!config.filename.empty()) {
    std::filesystem::path input_path(config.filename);
    std::string basename = input_path.stem().string();
    opt_filename = basename + "_opt.xyz";
  } else {
    opt_filename = "optimized.xyz"; // Fallback name
  }
  
  auto opt_file = fmt::output_file(opt_filename);
  const auto &final_mol = optimizer.get_next_geometry();
  const auto &final_el = final_mol.elements();
  const auto &final_pos = final_mol.positions();
  
  opt_file.print("{}\nOptimized geometry - Final energy: {:.12f} Ha\n", final_el.size(), wfn.energy.total);
  for (int i = 0; i < final_el.size(); i++) {
    opt_file.print("{} {:12.6f} {:12.6f} {:12.6f}\n", final_el[i].symbol(),
                   final_pos(0, i), final_pos(1, i), final_pos(2, i));
  }
  occ::log::info("Final optimized geometry written to: {}", opt_filename);

  return wfn;
}

std::pair<qm::Wavefunction, core::VibrationalModes> 
geometry_optimization_with_frequencies(const io::OccInput &config, bool run_frequencies) {
  
  // Perform geometry optimization
  auto optimized_wfn = geometry_optimization(config);
  
  core::VibrationalModes vib_modes;
  
  if (run_frequencies) {
    occ::log::info("\n{:=^72s}", "  Post-Optimization Frequency Analysis  ");
    occ::log::info("Geometry optimization completed successfully");
    occ::log::info("Computing vibrational frequencies at optimized geometry...");
    
    // Configure vibrational analysis
    VibrationalAnalysisConfig vib_config;
    vib_config.compute_frequencies = true;
    vib_config.project_tr_rot = false;  // Default: don't project modes
    vib_config.step_size = 0.005;       // ORCA-like default
    vib_config.use_acoustic_sum_rule = true;  // Efficient computation
    vib_config.save_hessian = true;     // Save for later analysis
    vib_config.save_results = true;     // Save frequency analysis
    
    // Set filenames based on input basename
    if (!config.filename.empty()) {
      std::filesystem::path input_path(config.filename);
      std::string basename = input_path.stem().string();
      vib_config.hessian_filename = basename + "_hess.json";
      vib_config.results_filename = basename + "_freq.json";
    } else {
      vib_config.hessian_filename = "optimized_hessian.json";
      vib_config.results_filename = "optimized_frequencies.json";
    }
    
    try {
      vib_modes = vibrational_analysis(config, optimized_wfn, vib_config);
      
      // Check if this is a minimum (no imaginary frequencies)
      auto sorted_freqs = vib_modes.get_all_frequencies();
      int n_imaginary = 0;
      for (int i = 0; i < sorted_freqs.size(); i++) {
        if (sorted_freqs[i] < -50.0) {  // Consider frequencies < -50 cm⁻¹ as truly imaginary
          n_imaginary++;
        }
      }
      
      if (n_imaginary == 0) {
        occ::log::info("SUCCESS: Optimized geometry is a local minimum (no imaginary frequencies)");
      } else {
        occ::log::warn("WARNING: Found {} imaginary frequencies - may be a transition state or saddle point", n_imaginary);
      }
      
    } catch (const std::exception &e) {
      occ::log::error("Failed to compute vibrational frequencies: {}", e.what());
      occ::log::warn("Continuing with optimization result only");
    }
  }
  
  return {optimized_wfn, vib_modes};
}

} // namespace occ::driver
